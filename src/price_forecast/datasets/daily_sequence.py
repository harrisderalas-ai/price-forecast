from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from ..config import DatasetCfg




class DailySequenceDataset:
    """
    Daily LSTM dataset (no intra-day sliding).

    Input X:  previous N full days  -> shape (N*24, n_features)
    Target y: 24 values of the LAST day inside that window -> shape (24, 1)
    Step:      exactly one day (00:00→23:00) per sample.

    Assumptions: dataframe is already hourly, contiguous, unique timestamps, and NaN-free.
    """

    # ---------- lifecycle ----------
    def __init__(self, df: pd.DataFrame, cfg: DatasetCfg):
        self.cfg = cfg
        self.df = df.copy().sort_values(cfg.datetime_col).reset_index(drop=True)

        # Sanity checks (no mutation)
        self._assert_hourly_unique(self.df[cfg.datetime_col])

        # Calendar features (deterministic)
        fdf = self._add_calendar_features(self.df, cfg.datetime_col)

        # Choose numeric feature columns (exclude target); enforce main series order
        self.feature_cols = self._choose_feature_cols(
            fdf, cfg.target_col, cfg.main_series, cfg.put_main_first
        )

        # Keep datetime for tidy DataFrame exports
        self.X_frame = fdf[[cfg.datetime_col] + self.feature_cols]
        self.y_series = self.df[cfg.target_col]

        # Reshape to days; do not alter values or timezone
        self.X_days, self.dt_days = self._reshape_days(
            self.X_frame[self.feature_cols].to_numpy(),
            self.X_frame[cfg.datetime_col],
        )
        self.y_days, _ = self._reshape_days(
            self.y_series.to_numpy().reshape(-1, 1),
            self.X_frame[cfg.datetime_col],
        )

        if self.X_days.shape[0] != self.y_days.shape[0]:
            raise RuntimeError("Mismatch between X and y day counts.")
        if self.X_days.shape[1] != 24 or self.y_days.shape[1] != 24:
            raise RuntimeError("Each day must have exactly 24 rows (00:00→23:00).")

    # ---------- public API ----------
    def build(self, return_dfs: bool = True) -> Dict[str, object]:
        """
        Assemble samples and split on the nearest whole-day boundary.
        Returns BOTH scaled and unscaled arrays (and DataFrames if requested).
        """
        X, y, Xdt, ydt = self._make_samples_same_day()

        S = X.shape[0]
        split_idx = int(round(S * (1 - self.cfg.test_size)))
        split_idx = max(self.cfg.n_lookback_days, min(S - 1, split_idx))

        # --- split ---
        X_tr, X_te = X[:split_idx], X[split_idx:]
        y_tr, y_te = y[:split_idx], y[split_idx:]
        Xdt_tr, Xdt_te = Xdt[:split_idx], Xdt[split_idx:]
        ydt_tr, ydt_te = ydt[:split_idx], ydt[split_idx:]

        # --- keep RAW (unscaled) copies ---
        X_tr_raw, X_te_raw = X_tr.copy(), X_te.copy()
        y_tr_raw, y_te_raw = y_tr.copy(), y_te.copy()

        # --- scale on TRAIN only (if enabled) ---
        if self.cfg.scale_features:
            F = X_tr.shape[2]
            self.cfg.feature_scaler.fit(X_tr.reshape(-1, F))
            X_tr = self.cfg.feature_scaler.transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
            X_te = self.cfg.feature_scaler.transform(X_te.reshape(-1, F)).reshape(X_te.shape)

        if self.cfg.scale_target:
            self.cfg.target_scaler.fit(y_tr.reshape(-1, 1))
            y_tr = self.cfg.target_scaler.transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
            y_te = self.cfg.target_scaler.transform(y_te.reshape(-1, 1)).reshape(y_te.shape)

        out: Dict[str, object] = {
            # Scaled
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,

            # Unscaled (raw)
            "X_train_raw": X_tr_raw, "X_test_raw": X_te_raw,
            "y_train_raw": y_tr_raw, "y_test_raw": y_te_raw,

            "meta": {
                "split_idx_days": split_idx,
                "total_days": S,
                "feature_cols": self.feature_cols,
                "lookback_hours": self.cfg.n_lookback_days * 24,
            },
        }

        if return_dfs:
            # Scaled DFs
            out["X_train_df"] = self._to_long_X_df(X_tr, Xdt_tr, self.feature_cols)
            out["X_test_df"]  = self._to_long_X_df(X_te, Xdt_te, self.feature_cols)
            out["y_train_df"] = self._to_long_y_df(y_tr, ydt_tr, self.cfg.target_col)
            out["y_test_df"]  = self._to_long_y_df(y_te, ydt_te, self.cfg.target_col)

            # Raw DFs
            out["X_train_df_raw"] = self._to_long_X_df(X_tr_raw, Xdt_tr, self.feature_cols)
            out["X_test_df_raw"]  = self._to_long_X_df(X_te_raw, Xdt_te, self.feature_cols)
            out["y_train_df_raw"] = self._to_long_y_df(y_tr_raw, ydt_tr, self.cfg.target_col)
            out["y_test_df_raw"]  = self._to_long_y_df(y_te_raw, ydt_te, self.cfg.target_col)

        return out

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform y from scaler space to original units (no-op if scaling disabled)."""
        if not self.cfg.scale_target:
            return y_scaled
        return self.cfg.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)

    # ---------- static/utility methods ----------
    @staticmethod
    def _assert_hourly_unique(dt: pd.Series | pd.Index) -> None:
        s = pd.to_datetime(dt)
        if s.duplicated().any():
            raise ValueError("Timestamps must be unique.")
        # Check 1-hour diffs except for the first row
        diffs = s.diff().iloc[1:]
        # Allow tz-aware or naive; compare to 1 hour
        if not np.all(diffs.view("i8") == pd.Timedelta(hours=1).value):
            raise ValueError("Datetime diffs must be exactly 1 hour everywhere.")

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
        out = df.copy()
        dt = pd.to_datetime(out[dt_col])
        out["hour_sin"]   = np.sin(2 * np.pi * dt.dt.hour / 24.0)
        out["hour_cos"]   = np.cos(2 * np.pi * dt.dt.hour / 24.0)
        out["day_sin"]    = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
        out["day_cos"]    = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
        out["month_sin"]  = np.sin(2 * np.pi * dt.dt.month / 12.0)
        out["month_cos"]  = np.cos(2 * np.pi * dt.dt.month / 12.0)
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        return out

    @staticmethod
    def _choose_feature_cols(df: pd.DataFrame, target_col: str,
                             main_series: str, put_main_first: bool) -> List[str]:
        cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
        if main_series not in cols:
            raise ValueError(f"'{main_series}' is missing among numeric features.")
        cols.remove(main_series)
        return ([main_series] + cols) if put_main_first else (cols + [main_series])

    @staticmethod
    def _reshape_days(values_2d: np.ndarray, dt_like: pd.Series | pd.Index
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape flat hourly arrays into per-day blocks.
        Returns:
          data_days: (num_days, 24, n_feat)
          dt_days:   (num_days, 24) ndarray of datetimes (tz preserved)
        """
        n_rows = values_2d.shape[0]
        dt_arr = pd.to_datetime(dt_like).to_numpy(copy=False)  # works for Series/Index, tz-safe

        if dt_arr.shape[0] != n_rows:
            raise ValueError("Datetime and values length mismatch.")

        # Align to full days: trim the head so length is a multiple of 24 (keeps order intact)
        rem = n_rows % 24
        if rem != 0:
            values_2d = values_2d[rem:]
            dt_arr = dt_arr[rem:]
            n_rows = values_2d.shape[0]

        num_days = n_rows // 24
        data_days = values_2d.reshape(num_days, 24, -1)
        dt_days   = dt_arr.reshape(num_days, 24)
        return data_days, dt_days

    def _make_samples_same_day(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each day d (starting at n_lookback_days-1):
          X = concat days [d - N + 1 .. d]  -> (N*24, F)
          y = prices of day d               -> (24, 1)
        """
        N = self.cfg.n_lookback_days
        num_days, _, n_feat = self.X_days.shape

        X_list, y_list, Xdt_list, ydt_list = [], [], [], []
        for d in range(N - 1, num_days):
            start_d = d - N + 1
            X_block = self.X_days[start_d:d+1].reshape(N * 24, n_feat)
            y_block = self.y_days[d]                         # (24, 1)
            Xdt     = self.dt_days[start_d:d+1].reshape(N * 24)
            ydt     = self.dt_days[d]

            X_list.append(X_block)
            y_list.append(y_block)
            Xdt_list.append(Xdt)
            ydt_list.append(ydt)

        X   = np.stack(X_list, axis=0)     # (S, N*24, F)
        y   = np.stack(y_list, axis=0)     # (S, 24, 1)
        Xdt = np.stack(Xdt_list, axis=0)   # (S, N*24)
        ydt = np.stack(ydt_list, axis=0)   # (S, 24)
        return X, y, Xdt, ydt

    @staticmethod
    def _to_long_X_df(X: np.ndarray, dts: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
        S, T, F = X.shape
        idx = pd.MultiIndex.from_product([range(S), range(T)], names=["sample", "t"])
        df = pd.DataFrame(X.reshape(S*T, F), index=idx, columns=feature_cols)
        df["datetime"] = pd.to_datetime(dts.reshape(S*T))
        return df[["datetime"] + feature_cols]

    @staticmethod
    def _to_long_y_df(y: np.ndarray, dts: np.ndarray, target_col: str) -> pd.DataFrame:
        S, T, _ = y.shape
        idx = pd.MultiIndex.from_product([range(S), range(T)], names=["sample", "hour"])
        out = pd.DataFrame(
            {target_col: y.reshape(S*T), "datetime": pd.to_datetime(dts.reshape(S*T))},
            index=idx,
        )
        return out[["datetime", target_col]]