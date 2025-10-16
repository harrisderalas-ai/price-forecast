from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import pandas as pd
from ..config import TimezoneConfig, Naming, DAMConfig, WeatherConfig, WeatherSite
from ..data_sources.dam import DAMAdapter
from ..data_sources.entsoe import EntsoeAdapter
from ..data_sources.weather import WeatherAdapter
from ..utils.merge import TimeSeriesMerger

class DatasetBuilder:
    """
    High-level orchestrator to produce a single, model-ready dataset.

    Responsibilities
    ----------------
    - Normalize time across sources (DAM, ENTSO-E, Open-Meteo) via the provided adapters.
    - Merge data on a UTC hourly axis (robust against DST).
    - Convert back to tz-aware local time for analysis/feature engineering.
    - Create derived features (e.g., previous-day price).
    - Return the final DataFrame.

    Design
    ------
    - Dependency injection: pass in your config and client objects rather than creating them internally.
      This keeps the class easily testable and configurable.
    - Single entrypoint: `create_dataset(...)`.
    - Readability-first: explicit steps + comments; clear variable names; minimal magic.

    Parameters
    ----------
    tz_cfg : TimezoneConfig
        Time configuration (tz names, DST handling, hourly freq).
    naming : Naming
        Column names configuration, e.g. local datetime column and DAM price column.
    dam_cfg : DAMConfig
        Where/how to find local DAM files.
    weather_sites : list[WeatherSite]
        Which Open-Meteo sites to load, e.g. Athens/Thessaloniki/Heraklion.
    weather_cfg : WeatherConfig
        Which Open-Meteo hourly variables to request.
    entsoe_client : EntsoePandasClient
        A ready-to-use ENTSO-E client (already authenticated).
    entsoe_prefix : str
        Prefix applied to ENTSO-E columns to avoid name collisions (default: "entsoe_").
    """

    def __init__(
        self,
        tz_cfg: TimezoneConfig,
        naming: Naming,
        dam_cfg: DAMConfig,
        weather_sites: List[WeatherSite],
        weather_cfg: WeatherConfig,
        entsoe_client,
        entsoe_prefix: str = "entsoe_",
    ) -> None:
        self.tz_cfg = tz_cfg
        self.naming = naming

        # Adapters
        self._dam = DAMAdapter(tz_cfg, dam_cfg, naming=naming)
        self._entsoe = EntsoeAdapter(tz_cfg, prefix=entsoe_prefix, naming=naming)
        self._weather = WeatherAdapter(tz_cfg, weather_sites, weather_cfg, naming=naming)

        # Merger
        self._merger = TimeSeriesMerger(tz_cfg, naming=naming)

        # External client
        self._entsoe_client = entsoe_client

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_entsoe_window(
        self, start_date_str: str, end_date_str: str
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        ENTSO-E API requires tz-aware timestamps; the end is *exclusive*.
        We take local calendar days, convert to UTC, then pad end by +1h.

        Example:
            start="2020-11-01", end="2023-12-31"  -->
            start_local = 2020-11-01 00:00 (Athens)
            end_local   = 2023-12-31 23:00 (Athens)
            start_utc   = start_local in UTC
            end_utc     = (end_local + 1h) in UTC   # exclusive
        """
        start_local = pd.Timestamp(start_date_str, tz=self.tz_cfg.tz_local)
        end_local = pd.Timestamp(f"{end_date_str} 23:00", tz=self.tz_cfg.tz_local)

        start_utc = start_local.tz_convert(self.tz_cfg.tz_canon)
        end_utc = (end_local + pd.Timedelta(hours=1)).tz_convert(self.tz_cfg.tz_canon)

        return start_utc, end_utc

    def _build_dam(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        (dam_local, dam_utc)
            dam_local: columns [naming.dt_local, naming.dam_price]
            dam_utc:   UTC-indexed, column [naming.dam_price]
        """
        dam_local = self._dam.load_local_hourly()
        dam_utc = self._dam.to_utc_hourly(dam_local)
        return dam_local, dam_utc

    def _build_entsoe(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries ENTSO-E and normalizes to (local, UTC) using EntsoeAdapter.

        Returns
        -------
        (entsoe_local, entsoe_utc)
        """
        res_fc = self._entsoe_client.query_wind_and_solar_forecast(
            "GR", start=start_utc, end=end_utc
        )
        entsoe_local, entsoe_utc = self._entsoe.normalize(res_fc)
        return entsoe_local, entsoe_utc

    def _build_weather(self, start_date_str: str, end_date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Open-Meteo archive expects string dates ('YYYY-MM-DD') and a timezone keyword.

        Returns
        -------
        (wx_local, wx_utc)
        """
        wx_local, wx_utc = self._weather.build_weather(start_date=start_date_str, end_date=end_date_str)
        return wx_local, wx_utc

    def _merge_all_utc(self, dam_utc: pd.DataFrame, entsoe_utc: pd.DataFrame, wx_utc: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all UTC-indexed frames on the UTC hourly index.
        """
        merged_utc = self._merger.merge_utc([dam_utc, entsoe_utc, wx_utc], how="outer")
        return merged_utc

    def _back_to_local(self, df_utc: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a UTC-indexed frame back to local tz with a named datetime column.
        """
        return self._merger.to_local_with_col(df_utc, out_col=self.naming.dt_local)

    def _add_previous_day_feature(self, df_local: pd.DataFrame, hours_back: int = 24, prev_day_col: str = "previous_day_dam") -> pd.DataFrame:
        """
        Add 'previous_day_dam' by shifting the configured DAM price column back by 24 hours (default).

        Parameters
        ----------
        df_local : DataFrame
            Local-time dataset with the configured dam price column.
        hours_back : int
            Shift window in hours (24 → previous day).

        Returns
        -------
        DataFrame
            Same as input with a new column 'previous_day_dam'.
        """
        out = df_local.copy()
        if self.naming.dam_price not in out.columns:
            raise KeyError(
                f"Expected price column '{self.naming.dam_price}' not found. "
                "Ensure DAMAdapter naming matches the dataset."
            )
        prev_day = out[self.naming.dam_price].shift(hours_back)
        out.insert(2, prev_day_col, prev_day)
        out.dropna(subset=[prev_day_col], inplace=True)
        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_dataset(self, start_date: str, end_date: str, add_prev_day: bool = True) -> pd.DataFrame:
        """
        Build the final, model-ready dataset.

        Steps
        -----
        1) Compute the (start_utc, end_utc) window for ENTSO-E (end exclusive).
        2) Build DAM (local & UTC).
        3) Build ENTSO-E (local & UTC).
        4) Build Weather (local & UTC).
        5) Merge all *UTC* frames on the hourly index.
        6) Convert back to local (tz-aware column).
        7) Add previous-day feature (optional).

        Parameters
        ----------
        start_date : str
            Calendar start day in 'YYYY-MM-DD' (interpreted in local tz).
        end_date : str
            Calendar end day in 'YYYY-MM-DD' (interpreted in local tz).
        add_prev_day : bool
            If True, adds 'previous_day_dam' = dam_price shifted by 24 hours.

        Returns
        -------
        DataFrame
            tz-aware local dataset with aligned columns from DAM, ENTSO-E and weather,
            plus the optional previous-day feature.
        """
        # 1) ENTSO-E window
        start_utc, end_utc = self._compute_entsoe_window(start_date, end_date)

        # 2) Sources
        dam_local, dam_utc = self._build_dam()
        print("Finished building DAM data.")
        entsoe_local, entsoe_utc = self._build_entsoe(start_utc, end_utc)
        print("Finished building ENTSO-E data.")
        wx_local, wx_utc = self._build_weather(start_date, end_date)
        print("Finished building Weather data.")
        # 3) Merge on UTC
        merged_utc = self._merge_all_utc(dam_utc, entsoe_utc, wx_utc)
        print("Finished merging all data on UTC axis.")
        # 4) Back to local
        merged_local = self._back_to_local(merged_utc)
        print("Converted merged data back to local timezone.")
        # 5) Feature engineering
        if add_prev_day:
            merged_local = self._add_previous_day_feature(merged_local, hours_back=24)
            print("Added previous-day DAM price feature.")
        return merged_local
