from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import pandas as pd

from ..config import TimezoneConfig

# ============================================================================
# TimeAxisService: one place for all timezone & frequency logic
# ============================================================================

class TimeAxisService:
    """
    Centralizes tz-localization, conversion to UTC, duplicate-hour collapsing,
    and regularizing to an exact hourly grid.
    """

    def __init__(self, cfg: TimezoneConfig):
        self.cfg = cfg

    # ---------- tz-localization / conversion ----------

    def ensure_localized(self, s: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
        """
        Ensure a tz-aware *local* (cfg.tz_local) datetime-like object.
        Accepts either a Series or a DatetimeIndex.
        """
        # DatetimeIndex path
        if isinstance(s, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(pd.to_datetime(s, errors="coerce"))
            if idx.tz is None:
                try:
                    return idx.tz_localize(
                        self.cfg.tz_local,
                        nonexistent=self.cfg.nonexistent,
                        ambiguous=self.cfg.ambiguous_localize,
                    )
                except Exception:
                    # Fallback: drop ambiguous repeats to NaT if inference fails
                    return idx.tz_localize(
                        self.cfg.tz_local,
                        nonexistent=self.cfg.nonexistent,
                        ambiguous="NaT",
                    )
            # Already tz-aware: convert to local
            return idx.tz_convert(self.cfg.tz_local)

        # Series path
        s = pd.to_datetime(s, errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            try:
                s = s.dt.tz_localize(
                    self.cfg.tz_local,
                    nonexistent=self.cfg.nonexistent,
                    ambiguous=self.cfg.ambiguous_localize,
                )
            except Exception:
                s = s.dt.tz_localize(
                    self.cfg.tz_local,
                    nonexistent=self.cfg.nonexistent,
                    ambiguous="NaT",
                )
        else:
            s = s.dt.tz_convert(self.cfg.tz_local)
        return s

    def to_utc(self, s_local: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
        """
        Convert a tz-aware *local* datetime-like to tz-aware UTC.
        Accepts either a Series or a DatetimeIndex.
        """
        if isinstance(s_local, pd.DatetimeIndex):
            return s_local.tz_convert(self.cfg.tz_canon)

        s_local = pd.to_datetime(s_local, errors="coerce")
        if getattr(s_local.dt, "tz", None) is None:
            s_local = s_local.dt.tz_localize(
                self.cfg.tz_local,
                nonexistent=self.cfg.nonexistent,
                ambiguous=self.cfg.ambiguous_localize,
            )
        return s_local.dt.tz_convert(self.cfg.tz_canon)

    # ---------- hourly regularization ----------

    def to_exact_hourly(self, df: pd.DataFrame, index_col: str) -> pd.DataFrame:
        """
        Put the series on a *local* exact hourly grid (cfg.hourly_freq).
        Steps:
          1) set `index_col` as index;
          2) collapse duplicate timestamps by **mean of numeric columns** (handles DST overlaps);
          3) sort and `.asfreq('h')` to regularize.
        Returns a DataFrame with a tz-aware DatetimeIndex.
        """
        d = df.copy()
        d = d.dropna(subset=[index_col])

        # Set index and sort
        d = d.set_index(index_col).sort_index()

        # Collapse duplicates across *numeric* columns only
        num_cols = d.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            d = d.groupby(d.index)[list(num_cols)].mean()
        else:
            # If no numeric data, keep first occurrence to ensure unique index
            d = d[~d.index.duplicated(keep="first")]

        # Exact hourly grid (no fills here; leave NaNs explicit)
        d = d.asfreq(self.cfg.hourly_freq)
        return d

    def utc_index_to_local_col(self, df_utc: pd.DataFrame, out_col: str = "datetime_local") -> pd.DataFrame:
        """
        Convert a UTC-indexed DataFrame's index to local time and expose it as a column.
        """
        out = df_utc.copy()
        out.index = out.index.tz_convert(self.cfg.tz_local)
        out.index.name = out_col
        return out.reset_index()