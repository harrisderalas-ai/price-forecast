from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import pandas as pd
from ..config import TimezoneConfig, Naming
from ..utils.time_axis import TimeAxisService
from ..utils.interpolate import interpolate_local, interpolate_utc

# ============================================================================
# Weather (Open-Meteo) Adapter
# ============================================================================

class WeatherAdapter:
    """
    Fetches Open-Meteo Archive hourly data for multiple sites,
    returns:
      - weather (local): tz-aware local time column + per-site columns + wx_* aggregates
      - wx_utc (UTC):    UTC-indexed hourly copy with same value columns
    """
    OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, tz: TimezoneConfig, sites: List[WeatherSite], wx_cfg: WeatherConfig, naming: Naming = Naming()):
        self.tz = tz
        self.axis = TimeAxisService(tz)
        self.sites = sites
        self.wx_cfg = wx_cfg
        self.naming = naming

    # ----- internals -----

    def _fetch_site(self, site: WeatherSite, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch one site's hourly archive as a tz-aware (local) series, and regularize to exact hourly.
        Open-Meteo expects 'YYYY-MM-DD' strings for start/end and a named timezone.
        """
        params = {
            "latitude": site.lat,
            "longitude": site.lon,
            "hourly": self.wx_cfg.hourly_vars,
            "timezone": self.tz.tz_local,     # Interpret calendar days in local tz
            "start_date": start_date,         # inclusive
            "end_date": end_date,             # inclusive
        }
        r = requests.get(self.OPEN_METEO_ARCHIVE, params=params, timeout=180)
        r.raise_for_status()
        js = r.json()
        if "hourly" not in js:
            raise ValueError(f"No 'hourly' in response for {site.name}: {js}")

        h = pd.DataFrame(js["hourly"])

        # Localize immediately to avoid downstream AmbiguousTimeError
        h[self.naming.dt_local] = pd.to_datetime(h["time"], errors="coerce")
        h[self.naming.dt_local] = self.axis.ensure_localized(h[self.naming.dt_local])

        # Clean & rename per-site columns
        h = h.drop(columns=["time"])
        rename = {c: f"{c}_{site.name}" for c in h.columns if c != self.naming.dt_local}
        h = h.rename(columns=rename)

        # Exact local hourly grid (handles duplicates)
        h = self.axis.to_exact_hourly(h, self.naming.dt_local).reset_index()
        return h

    # ----- public API -----

    def build_weather(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download/merge all sites → (weather_local, weather_utc)
        - 'weather' has tz-aware local column and per-site values
        - 'wx_utc' is UTC-indexed with same value columns
        Both are interpolated (time-wise).
        """
        # Fetch each site (polite pacing)
        frames: List[pd.DataFrame] = []
        for s in self.sites:
            frames.append(self._fetch_site(s, start_date, end_date))
            time.sleep(2)

        # Outer-merge on local datetime, then sort
        weather = frames[0]
        for df in frames[1:]:
            weather = weather.merge(df, on=self.naming.dt_local, how="outer")
        weather = weather.sort_values(self.naming.dt_local)

        # Aggregated features across sites (prefix wx_)
        def _avg(prefixes: List[str]) -> pd.Series:
            cols = [c for c in weather.columns if any(c.startswith(f"{p}_") for p in prefixes)]
            return weather[cols].mean(axis=1) if cols else pd.Series(pd.NA, index=weather.index)

        if any(c.startswith("temperature_2m_") for c in weather.columns):
            weather["wx_temp_avg_c"] = _avg(["temperature_2m"])
        if any(c.startswith("cloudcover_") for c in weather.columns):
            weather["wx_cloud_avg_pct"] = _avg(["cloudcover"])
        if any(c.startswith("wind_speed_100m_") for c in weather.columns):
            weather["wx_wind100_avg"] = _avg(["wind_speed_100m"])
        if any(c.startswith("shortwave_radiation_") for c in weather.columns):
            weather["wx_swr_avg"] = _avg(["shortwave_radiation"])

        # Simple power proxies
        if "wx_wind100_avg" in weather:
            weather["wx_wind_power_idx"] = weather["wx_wind100_avg"].clip(lower=0) ** 3
        if "wx_swr_avg" in weather:
            weather["wx_solar_power_idx"] = weather["wx_swr_avg"].clip(lower=0)

        weather = weather.dropna(axis=1, how="all")

        # Build UTC-hourly from local time column
        tmp = weather.copy()
        tmp["dt_utc"] = self.axis.to_utc(tmp[self.naming.dt_local])
        value_cols = [c for c in tmp.columns if c not in (self.naming.dt_local, "dt_utc")]
        wx_utc = (
            tmp.set_index("dt_utc")[value_cols]
               .sort_index()
               .asfreq(self.tz.hourly_freq)
        )

        # Interpolate both representations
        weather = interpolate_local(weather, time_col=self.naming.dt_local)
        wx_utc = interpolate_utc(wx_utc)
        return weather, wx_utc