from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Sequence
from ..config import DatasetCfg
from ..utils.relative import RelativeDeltaTransformer


class DailySequenceDataset:
    """
    Daily LSTM dataset (no intra-day sliding).

    Input X:  previous N full days  -> shape (N*24, n_features)
    Target y: 24 values of the LAST day inside that window -> shape (24, 1)
    Step:      exactly one day (00:00→23:00) per sample.

    Assumptions: dataframe is already hourly, contiguous, unique timestamps, and NaN-free.

    New (opt-in, backwards compatible):
    - target_as_relative:   predict relative deltas for the target instead of absolute values
    - relative_feature_cols: turn selected feature columns into relative deltas (e.g., ["previous_day_dam"])
    """

    # ---------- lifecycle ----------
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DatasetCfg,
        *,
        target_as_relative: bool = False,
        relative_feature_cols: Optional[Sequence[str]] = None,
        epsilon: float = 1e-6,
        fill_value: float = 0.0,
    ):
        self.cfg = cfg
        self.df = df.copy().sort_values(cfg.datetime_col).reset_index(drop=True)

        # Relative-delta setup
        self.target_as_relative = target_as_relative
        self.relative_feature_cols = list(relative_feature_cols or [])
        self.rel_t = RelativeDeltaTransformer(epsilon=epsilon, fill_value=fill_value,mode='diff')

        # Sanity checks (no mutation)
        self._assert_hourly_unique(self.df[cfg.datetime_col])

        # Calendar features (deterministic; same as old impl)
        fdf = self._add_calendar_features(self.df, cfg.datetime_col)

        # Choose numeric feature columns (exclude target); enforce main series order
        self.feature_cols = self._choose_feature_cols(
            fdf, cfg.target_col, cfg.main_series, cfg.put_main_first
        )

        # Optionally convert selected *features* to relative differences
        if self.relative_feature_cols:
            # only transform those that are actually in feature_cols
            rel_cols = [c for c in self.relative_feature_cols if c in self.feature_cols]
            if rel_cols:
                # transform each column independently to relative deltas
                for c in rel_cols:
                    fdf[c] = self.rel_t.fit_transform(fdf[c].to_numpy())

        # Keep datetime for tidy DataFrame exports
        self.X_frame = fdf[[cfg.datetime_col] + self.feature_cols]
        y_series_abs = self.df[cfg.target_col].copy()

        # Optionally convert *target* to relative differences
        if self.target_as_relative:
            y_rel = self.rel_t.fit_transform(y_series_abs.to_numpy())
            y_rel = np.clip(y_rel, -3.0, 3.0)
            self.y_series = pd.Series(y_rel, index=y_series_abs.index, name=cfg.target_col)
            # Keep bases per day to invert later: base for day k is the absolute value at (k*24 - 1)
            self._y_base_days = self._compute_day_bases(y_series_abs)
        else:
            self.y_series = y_series_abs
            self._y_base_days = None

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

        # Will be filled by build()
        self._y_train_bases: Optional[np.ndarray] = None
        self._y_test_bases: Optional[np.ndarray] = None

    # ---------- public API ----------
    def build(self, return_dfs: bool = True) -> Dict[str, object]:
        """
        Assemble samples and split on the nearest whole-day boundary.
        Returns BOTH scaled and unscaled arrays (and DataFrames if requested).

        Backwards-compatible with the previous class (same keys in the dict).
        If target_as_relative=True, also returns `y_train_bases` and `y_test_bases`
        to help invert predictions day-by-day.
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

        # --- per-sample bases for relative target inversion ---
        if self.target_as_relative:
            # base for sample d is base of the TARGET's last-day in that sample (i.e., day d)
            # samples start at day index (N-1) .. (num_days-1)
            N = self.cfg.n_lookback_days
            sample_day_indices = np.arange(N - 1, N - 1 + S)
            y_bases_all = self._y_base_days[sample_day_indices]
            self._y_train_bases = y_bases_all[:split_idx]
            self._y_test_bases = y_bases_all[split_idx:]
        else:
            self._y_train_bases = None
            self._y_test_bases = None

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
                "total_days": self.X_days.shape[0],
                "feature_cols": self.feature_cols,
                "lookback_hours": self.cfg.n_lookback_days * 24,
                "target_mode": "relative" if self.target_as_relative else "absolute",
            },
        }

        if self.target_as_relative:
            out["y_train_bases"] = self._y_train_bases
            out["y_test_bases"]  = self._y_test_bases

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

    # src/price_forecast/datasets/daily_sequence.py

    def inverse_transform_target(
        self,
        y_scaled: np.ndarray,
        *,
        bases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert model outputs back to absolute target units (EUR/MWh).

        Pipeline
        --------
        1) If y was scaled: inverse-scale using `target_scaler`.
        2) If target_as_relative is False: return (S, 24, 1).
        3) If target_as_relative is True:
        Reconstruct per sample, hour-by-hour, using the SAME denominator as forward:
            denom = sign(prev) * max(|prev|, min_denominator)
            x_t   = r_t * (denom + epsilon) + prev
        """
        # Normalize to (S, 24)
        y_arr = y_scaled[..., 0] if (y_scaled.ndim == 3 and y_scaled.shape[-1] == 1) else y_scaled

        # Step 1: undo scaling (if any)
        if self.cfg.scale_target:
            y_arr = self.cfg.target_scaler.inverse_transform(y_arr.reshape(-1, 1)).reshape(y_arr.shape)

        # Absolute target case
        if not self.target_as_relative:
            return y_arr[..., None]

        # Relative target requires bases
        if bases is None:
            raise ValueError(
                "Relative target inversion requires per-sample bases. "
                "Pass bases=out['y_train_bases'] or out['y_test_bases'] returned by build()."
            )
        if len(bases) != y_arr.shape[0]:
            raise ValueError("Length of `bases` must match number of samples (S).")

        # Step 2: reconstruct per sample; MIRROR forward denom
        out_abs = np.empty_like(y_arr, dtype=float)
        eps = self.rel_t.epsilon
        min_den = getattr(self.rel_t, "min_denominator", 0.0)

        for s in range(y_arr.shape[0]):
            prev = bases[s]
            if not np.isfinite(prev):
                # This can happen for the very first sample when n_lookback_days == 1
                # Skip or set to NaN; caller can mask it out for metrics.
                out_abs[s, :] = np.nan
                continue

            for t in range(y_arr.shape[1]):
                if min_den > 0.0:
                    denom = np.sign(prev) * max(abs(prev), min_den)
                else:
                    denom = prev
                out_abs[s, t] = (y_arr[s, t] * (denom + eps)) + prev
                prev = out_abs[s, t]

        return out_abs[..., None]



    # ---------- static/utility methods ----------
    @staticmethod
    def _assert_hourly_unique(dt: pd.Series | pd.Index) -> None:
        s = pd.to_datetime(dt)
        if s.duplicated().any():
            raise ValueError("Timestamps must be unique.")
        diffs = s.diff().iloc[1:]
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
        dt_arr = pd.to_datetime(dt_like).to_numpy(copy=False)
        if dt_arr.shape[0] != n_rows:
            raise ValueError("Datetime and values length mismatch.")

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

    # --------- helpers for relative inversion ---------
    @staticmethod
    def _compute_day_bases(y_abs: pd.Series) -> np.ndarray:
        """
        Base for day k is y_abs at index (k*24 - 1) — the hour just before that day starts.
        For k=0, base is NaN (no prior hour).
        Returns shape: (num_days,)
        """
        n = len(y_abs)
        rem = n % 24
        if rem != 0:
            y_abs = y_abs.iloc[rem:].reset_index(drop=True)
        num_days = len(y_abs) // 24
        bases = np.full((num_days,), np.nan, dtype=float)
        # day k starts at idx k*24; its base is idx k*24 - 1
        for k in range(num_days):
            idx = k * 24 - 1
            if idx >= 0:
                bases[k] = float(y_abs.iloc[idx])
        return bases


# ---------------------------------------------------------------------- #
# Small utility: identity scaler (drop-in when scaling disabled)
# ---------------------------------------------------------------------- #
class _IdentityScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return np.asarray(X)
    def fit_transform(self, X):
        return np.asarray(X)
    def inverse_transform(self, X):
        return np.asarray(X)

