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
from .time_axis import TimeAxisService

# ============================================================================
# Merger
# ============================================================================

class TimeSeriesMerger:
    """
    Joins multiple UTC-hourly frames (by index) and can convert back to local with a named column.
    """

    def __init__(self, tz: TimezoneConfig, naming: Naming = Naming()):
        self.axis = TimeAxisService(tz)
        self.naming = naming

    def merge_utc(self, dfs: List[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
        """
        Successively join frames on UTC hourly index.
        Assumes each df is:
          - indexed by UTC
          - already on an exact hourly grid
        """
        base = dfs[0].copy()
        for d in dfs[1:]:
            base = base.join(d, how=how)
        return base.sort_index().asfreq("h")

    def to_local_with_col(self, df_utc: pd.DataFrame, out_col: Optional[str] = None) -> pd.DataFrame:
        """
        Convert a UTC-indexed frame back to local, exposing a named time column.
        Default column name comes from Naming.dt_local.
        """
        col = out_col or self.naming.dt_local
        return self.axis.utc_index_to_local_col(df_utc, out_col=col)