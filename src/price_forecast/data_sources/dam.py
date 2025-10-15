from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import pandas as pd
from ..config import TimezoneConfig, DAMConfig, Naming
from ..utils.time_axis import TimeAxisService



# ============================================================================
# DAM (Day-Ahead Market) Adapter
# ============================================================================

class DAMAdapter:
    """
    Loads DAM Excel files, normalizes timestamps to local time,
    collapses duplicates, regularizes to exact hourly, and exposes:
      - a local hourly DataFrame with a configurable datetime column name
      - a UTC hourly DataFrame with the price column configurable
    """

    def __init__(self, tz: TimezoneConfig, cfg: DAMConfig, naming: Naming = Naming()):
        self.tz = tz
        self.axis = TimeAxisService(tz)
        self.cfg = cfg
        self.naming = naming  # holds dt_local and dam_price names

    # ----- internals -----

    def _detect_cols(self, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """
        Find the (datetime, mcp) columns in a case-insensitive manner,
        trying candidates from the config in order.
        """
        up = {c.upper(): c for c in df.columns}
        dt = next((up.get(c.upper()) for c in self.cfg.datetime_candidates if up.get(c.upper())), None)
        pr = next((up.get(c.upper()) for c in self.cfg.price_candidates if up.get(c.upper())), None)
        return (dt, pr) if dt and pr else None

    def _read_one(self, path: str) -> Optional[pd.DataFrame]:
        """
        Read one Excel and return a small frame with [dt_local, 'mcp'] (collapsed within file).
        """
        try:
            raw = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            logging.warning("[DAM] Failed to read %s: %s", os.path.basename(path), e)
            return None

        cols = self._detect_cols(raw)
        if not cols:
            logging.info("[DAM] Skip (missing expected columns) -> %s", os.path.basename(path))
            return None

        dt_col, mcp_col = cols
        d = raw[[dt_col, mcp_col]].rename(columns={dt_col: self.naming.dt_local, mcp_col: "mcp"})

        # Localize & clean
        d[self.naming.dt_local] = self.axis.ensure_localized(d[self.naming.dt_local])
        d["mcp"] = pd.to_numeric(d["mcp"], errors="coerce")
        d = d.dropna(subset=[self.naming.dt_local])

        # Collapse duplicate rows within the file for the same local timestamp
        d = d.groupby(self.naming.dt_local, as_index=False)["mcp"].mean()
        return d