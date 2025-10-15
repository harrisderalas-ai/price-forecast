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



# ============================================================================
# ENTSO-E Adapter
# ============================================================================

class EntsoeAdapter:
    """
    Normalizes ENTSO-E client outputs into:
      - a local-hourly DataFrame with a configurable local-datetime column name
      - a UTC-hourly DataFrame
    All value columns are prefixed (e.g., 'entsoe_*') to avoid collisions on merge.
    """

    def __init__(self, tz: TimezoneConfig, prefix: str = "entsoe_", naming: Naming = Naming()):
        self.tz = tz
        self.axis = TimeAxisService(tz)
        self.prefix = prefix
        self.naming = naming

    def normalize(self, res_fc) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Turn ENTSO-E Series/DataFrame into (local_hourly, utc_hourly).
        - Flattens MultiIndex columns if needed
        - Prefixes columns to avoid collisions
        - Localizes index to Athens, makes exact hourly
        - Builds UTC hourly
        - Interpolates both
        """
        # Ensure DataFrame
        df = res_fc.to_frame(name=res_fc.name) if isinstance(res_fc, pd.Series) else res_fc.copy()

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in t if str(x) != ""])
                for t in df.columns
            ]
        df.columns = [f"{self.prefix}{c}" for c in df.columns]

        # Localize index to Athens and put on exact hourly grid
        df.index = self.axis.ensure_localized(pd.to_datetime(df.index, errors="coerce"))
        local_hourly = df.sort_index().asfreq(self.tz.hourly_freq)
        local_hourly = local_hourly.reset_index().rename(columns={"index": self.naming.dt_local})

        # Build UTC-hourly (from the local datetime column)
        tmp = local_hourly.copy()
        tmp["dt_utc"] = self.axis.to_utc(pd.to_datetime(tmp[self.naming.dt_local]))
        value_cols = [c for c in tmp.columns if c not in (self.naming.dt_local, "dt_utc")]
        utc_hourly = (
            tmp.set_index("dt_utc")[value_cols]
               .sort_index()
               .asfreq(self.tz.hourly_freq)
        )

        # Interpolate both representations
        local_hourly = interpolate_local(local_hourly, time_col=self.naming.dt_local)
        utc_hourly = interpolate_utc(utc_hourly)
        return local_hourly, utc_hourly
