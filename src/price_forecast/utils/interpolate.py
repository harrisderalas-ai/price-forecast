from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import pandas as pd




# ============================================================================
# Small, focused interpolation helpers
# ============================================================================

def interpolate_local(df: pd.DataFrame, time_col: str = "datetime_local") -> pd.DataFrame:
    """
    Time-based interpolation (linear in time) for a DataFrame with a local time column.
    - Sorts by time
    - Interpolates numeric columns using method='time'
    - Fills both ends (limit_direction='both')
    - Returns the same shape, with time as a column.
    """
    out = df.copy() if time_col in df.columns else df.reset_index()
    out = out.set_index(time_col).sort_index()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].interpolate(method="time", limit_direction="both")
    return out.reset_index()


def interpolate_utc(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based interpolation (linear in time) for a UTC-indexed DataFrame.
    - Sorts by index
    - Interpolates numeric columns using method='time'
    - Fills both ends (limit_direction='both')
    """
    out = df_utc.sort_index().copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].interpolate(method="time", limit_direction="both")
    return out