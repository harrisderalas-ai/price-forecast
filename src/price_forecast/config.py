from __future__ import annotations
import os, glob, logging, time, requests
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============ Shared time-axis config/services ============

@dataclass(frozen=True)
class TimezoneConfig:
    tz_local: str = "Europe/Athens"
    tz_canon: str = "UTC"
    nonexistent: str = "shift_forward"   # spring gap
    ambiguous_localize: str = "infer"    # fall overlap
    hourly_freq: str = "h"


@dataclass(frozen=True)
class Naming:
    """
    Central place for column names used across the codebase.
    """
    dt_local: str = "datetime_local"
    dam_price: str = "dam_price_eur_mwh"


@dataclass(frozen=True)
class DAMConfig:
    """
    DAM (Day-Ahead Market) loader configuration.
    - base_dir:   root directory that contains *_EL-DAM_Results subfolders
    - pattern:    glob pattern to find Excel files
    - datetime_candidates: possible column names for delivery start
    - price_candidates:    possible column names for MCP
    """
    base_dir: str
    pattern: str = "*_EL-DAM_Results/*.xlsx"
    datetime_candidates: Tuple[str, ...] = ("DELIVERY_MTU", "DELIVERY_START", "DELIVERY_DATE_TIME")
    price_candidates: Tuple[str, ...] = ("MCP", "MARKET_CLEARING_PRICE")


@dataclass(frozen=True)
class WeatherSite:
    """A single weather site with name and coordinates."""
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class WeatherConfig:
    """
    Open-Meteo archive request configuration.
    NOTE: The archive API uses 'cloudcover' (no underscore). 'cloud_cover' will error.
    """
    hourly_vars: str = "temperature_2m,cloudcover,wind_speed_100m,shortwave_radiation"

    # Optional long form, retained here for reference; not used by default.
    hourly_vars_all: str = (
        "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "precipitation_probability,precipitation,rain,showers,snowfall,snow_depth,freezing_level_height,"
        "weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
        "visibility,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wind_speed_80m,wind_direction_80m,wind_speed_100m"
        "wind_direction_100m,wind_speed_120m,wind_direction_120m,wind_speed_180m,wind_direction_180m,shortwave_radiation,"
        "direct_radiation,diffuse_radiation,direct_normal_irradiance,et0_fao_evapotranspiration,vapour_pressure_deficit,is_day,sunshine_duration"
    )


from sklearn.preprocessing import StandardScaler
import numpy as np

class SafeStandardScaler(StandardScaler):
    """
    Like StandardScaler but clamps extremely small std during inverse_transform
    to avoid giant multipliers when mapping back to original space.
    """
    def __init__(self, min_scale: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = float(min_scale)

    def inverse_transform(self, X, copy=None):
        orig_scale = self.scale_
        try:
            self.scale_ = np.maximum(self.scale_, self.min_scale)
            return super().inverse_transform(X, copy=copy)
        finally:
            self.scale_ = orig_scale



@dataclass(frozen=True)
class DatasetCfg:
    datetime_col: str = "datetime_local"
    target_col: str = "dam_price_eur_mwh"
    n_lookback_days: int = 2
    test_size: float = 0.2
    main_series: str = "previous_day_dam"
    put_main_first: bool = True
    scale_features: bool = True
    scale_target: bool = True
    feature_scaler: StandardScaler = SafeStandardScaler()
    target_scaler: StandardScaler = SafeStandardScaler()
    use_relative_features: bool = False
    relative_feature_cols: Tuple[str, ...] = ("previous_day_dam",)
    target_as_relative: bool = False
    relative_epsilon: float = 1e-5



