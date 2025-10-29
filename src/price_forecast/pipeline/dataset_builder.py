from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..config import DAMConfig, Naming, TimezoneConfig, WeatherConfig, WeatherSite
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
        prev_day_col: str = "previous_day_dam",
    ) -> None:
        self.tz_cfg = tz_cfg
        self.naming = naming

        # Adapters
        self._dam = DAMAdapter(tz_cfg, dam_cfg, naming=naming)
        self._entsoe = EntsoeAdapter(tz_cfg, prefix=entsoe_prefix, naming=naming)
        self._weather = WeatherAdapter(
            tz_cfg, weather_sites, weather_cfg, naming=naming
        )

        # Merger
        self._merger = TimeSeriesMerger(tz_cfg, naming=naming)

        # External client
        self._entsoe_client = entsoe_client

        self.prev_day_col: str = prev_day_col

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

    def _build_entsoe(
        self, start_utc: pd.Timestamp, end_utc: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def _build_weather(
        self, start_date_str: str, end_date_str: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Open-Meteo archive expects string dates ('YYYY-MM-DD') and a timezone keyword.

        Returns
        -------
        (wx_local, wx_utc)
        """
        wx_local, wx_utc = self._weather.build_weather(
            start_date=start_date_str, end_date=end_date_str
        )
        return wx_local, wx_utc

    def _merge_all_utc(
        self, dam_utc: pd.DataFrame, entsoe_utc: pd.DataFrame, wx_utc: pd.DataFrame
    ) -> pd.DataFrame:
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

    def _add_previous_day_feature(
        self, df_local: pd.DataFrame, hours_back: int = 24
    ) -> pd.DataFrame:
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
        out.insert(2, self.prev_day_col, prev_day)

        return out

    def _add_diff_features(
        self,
        df: pd.DataFrame,
        *,
        as_relative: bool = False,
        prev_day_diff_name: Optional[str] = None,
        same_day_suffix: Optional[str] = None,
        epsilon: float = 1e-6,
        min_denominator: float = 0.0,
    ) -> pd.DataFrame:
        """
        Add first-order difference features for the DAM price and the 'previous-day' price.

        By default this computes **absolute differences** (x_t - x_{t-1}).
        If `as_relative=True`, it computes **relative differences** w.r.t. the previous value:
            rel_t = (x_t - x_{t-1}) / denom_t
        where:
            denom_t = x_{t-1}                             (if min_denominator == 0)
                    = sign(x_{t-1}) * max(|x_{t-1}|, m)  (if min_denominator > 0)
        and a small `epsilon` is added to the denominator for numerical safety.

        Parameters
        ----------
        df : pd.DataFrame
            Input frame in local time that already contains:
            - DAM price column (self.naming.dam_price)
            - previous-day price column (self.prev_day_col) created by `_add_previous_day_feature`.
        as_relative : bool, default False
            If True, compute relative differences; else absolute differences.
        prev_day_diff_name : str, optional
            Column name for the previous-day difference/relative feature.
            If None, we derive it automatically from `self.prev_day_col`:
            - '<prev_day_col>_diff'      when `as_relative=False`
            - '<prev_day_col>_rel'       when `as_relative=True`
        same_day_suffix : str, optional
            Suffix for the same-day price diff column. If None, we use:
            - '_diff' when `as_relative=False`
            - '_rel'  when `as_relative=True`
            The final column name will be '<dam_price><suffix>'.
        epsilon : float, default 1e-6
            Added to the denominator in relative mode for numerical safety.
        min_denominator : float, default 0.0
            If > 0, apply a sign-preserving absolute floor to the denominator in relative mode:
            denom = sign(prev) * max(|prev|, min_denominator)
            Use a domain-appropriate value (e.g., 5–20 for EUR/MWh) if zeros/near-zeros occur.

        Returns
        -------
        pd.DataFrame
            A copy of `df` with two new columns:
            - same-day diff/relative for `self.naming.dam_price`
            - diff/relative for `self.prev_day_col`
            Rows with NaN in `self.prev_day_col` are dropped (these arise from the 24h shift).

        Notes
        -----
        - The first hourly row naturally has no previous value, so the new columns
        will start with NaN (and additional NaNs where appropriate).
        - We *do not* fill these NaNs; your downstream dataset split will handle them,
        and we already drop rows where `previous_day_dam` is NaN to align samples.
        """
        out = df.copy()

        # --------- Validate required columns exist ---------
        dam_col = self.naming.dam_price
        prev_day_col = self.prev_day_col
        if dam_col not in out.columns:
            raise KeyError(
                f"Expected price column '{dam_col}' not found. "
                "Ensure DAMAdapter naming matches the dataset."
            )
        if prev_day_col not in out.columns:
            raise KeyError(
                f"Expected previous-day column '{prev_day_col}' not found. "
                "Run `_add_previous_day_feature` before `_add_diff_features`."
            )

        # --------- Decide output column names ---------
        if same_day_suffix is None:
            same_day_suffix = "_rel" if as_relative else "_diff"
        same_day_name = f"{dam_col}{same_day_suffix}"

        if prev_day_diff_name is None:
            prev_day_diff_name = f"{prev_day_col}{('_rel' if as_relative else '_diff')}"

        # --------- Fetch series (as float) and their previous-hour values ---------
        x = out[dam_col].astype(float)
        x_prev = x.shift(1)

        p = out[prev_day_col].astype(float)
        p_prev = p.shift(1)

        # --------- Define helpers for relative mode (safe denominator) ---------
        def _safe_rel(curr: pd.Series, prev: pd.Series) -> pd.Series:
            """
            Compute relative difference with optional sign-preserving denom floor
            and epsilon safety term. Returns a float Series matching 'curr' index.
            """
            if min_denominator > 0.0:
                denom = np.sign(prev) * np.maximum(np.abs(prev), min_denominator)
            else:
                denom = prev
            return (curr - prev) / (denom + epsilon)

        # --------- Compute diffs (absolute or relative) ---------
        if as_relative:
            out[same_day_name] = _safe_rel(x, x_prev)
            out[prev_day_diff_name] = _safe_rel(p, p_prev)
        else:
            out[same_day_name] = x - x_prev
            out[prev_day_diff_name] = p - p_prev

        # --------- Keep only rows where previous-day base exists ---------
        # This mirrors your previous behavior: rows before the 24h shift are removed.
        out.dropna(subset=[prev_day_col], inplace=True)

        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_dataset(
        self, start_date: str, end_date: str, add_prev_day: bool = True, add_diff: bool = True, as_relative: bool = False
    ) -> pd.DataFrame:
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
            if add_diff:
                merged_local = self._add_diff_features(merged_local, as_relative=as_relative)
                print("Added difference features for DAM price and previous-day DAM price.")
            merged_local.fillna(0, inplace=True)
        return merged_local
