from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import warnings
import requests
import pandas as pd
from ..config import TimezoneConfig, DAMConfig, Naming
from ..utils.time_axis import TimeAxisService
from ..utils.interpolate import interpolate_local, interpolate_utc



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

        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"openpyxl\.styles\.stylesheet",
                message=r".*Workbook contains no default style.*",
                )
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

     # ----- public API -----

    def load_local_hourly(self) -> pd.DataFrame:
        """
        Load and merge all DAM files → local hourly series:
        Columns: [<naming.dt_local>, <naming.dam_price>]; <naming.dt_local> tz-aware.
        """
        pattern = os.path.join(self.cfg.base_dir, self.cfg.pattern)
        files = glob.glob(pattern)
        if not files:
            raise ValueError(f"No XLSX found at {pattern}")

        parts = [d for f in files if (d := self._read_one(f)) is not None]
        if not parts:
            raise ValueError("No valid DAM rows parsed from any file.")

        # Merge parts and collapse duplicates across files
        dam = pd.concat(parts, ignore_index=True)
        dam = dam.groupby(self.naming.dt_local, as_index=False)["mcp"].mean()

        # Local exact hourly grid, rename price column, then interpolate
        dam = self.axis.to_exact_hourly(dam, self.naming.dt_local).rename(columns={"mcp": self.naming.dam_price})
        dam = dam.reset_index()  # expose local time column
        dam = interpolate_local(dam, time_col=self.naming.dt_local)
        return dam

    def to_utc_hourly(self, dam_local_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the local-hourly DAM series to a UTC-hourly frame.
        Index: UTC hourly; Columns: [<naming.dam_price>]
        """
        tmp = dam_local_hourly.copy()

        # Ensure the local datetime column exists
        if self.naming.dt_local not in tmp.columns:
            tmp = tmp.reset_index().rename(columns={"index": self.naming.dt_local})

        # Localize defensively and convert to UTC
        s_local = self.axis.ensure_localized(pd.to_datetime(tmp[self.naming.dt_local], errors="coerce"))
        tmp["dt_utc"] = self.axis.to_utc(s_local)

        out = (
            tmp.set_index("dt_utc")[[self.naming.dam_price]]
               .sort_index()
               .asfreq(self.tz.hourly_freq)
        )
        out = interpolate_utc(out)
        return out