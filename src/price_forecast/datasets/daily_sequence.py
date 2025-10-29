from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..config import DatasetCfg  # your dataclass(frozen=True) with scalers etc.


class DailySequenceDataset:
    """
    Daily LSTM dataset (no intra-day sliding).

    Input X:  previous N full days  -> shape (N*24, n_features)
    Target y: 24 values of the LAST day inside that window -> shape (24, 1)
    Step:      exactly one day (00:00→23:00) per sample.

    Assumptions
    -----------
    - DataFrame is already hourly, contiguous, unique timestamps, and NaN-free.
    - Upstream builder has already created the *_diff columns (e.g.:
        - target_col + "_diff" (e.g., "dam_price_eur_mwh_diff")
        - main_series + "_diff" (e.g., "previous_day_dam_diff")
      If not, `keep_only_diff=True` will result in empty features.

    What changed (compared to your earlier class)
    ---------------------------------------------
    - No relative features are generated here.
    - `build()` now accepts two switches:
        * keep_only_diff: if True, drop absolute feature columns and keep only *_diff.
        * diff_as_target: if True, y = target_diff; else y = target_abs.
          In both cases we prevent leakage by dropping the "other" target flavor from X.
    - A helper `inverse_transform_target(...)` reconstructs the absolute series:
        * If diff_as_target=False: inverse-scale only (absolute output).
        * If diff_as_target=True: inverse-scale diffs, then cumulatively add a per-sample base
          (absolute value at the hour right before the target day starts).

    Backwards compatibility
    -----------------------
    - Output dictionary keys match your previous pipeline (X/y arrays + optional DataFrames).
    - Adds `y_train_bases` / `y_test_bases` only when diff_as_target=True (needed to invert).
    - Adds `full_df` for inspection.
    """

    # ---------- lifecycle ----------
    def __init__(self, df: pd.DataFrame, cfg: DatasetCfg):
        self.cfg = cfg
        # Keep a full copy for inspection; ensure deterministic ordering
        self.full_df = df.copy().sort_values(cfg.datetime_col).reset_index(drop=True)

        # Defensive: timestamps must be hourly & unique
        self._assert_hourly_unique(self.full_df[cfg.datetime_col])

        # Calendar features (deterministic; stays the same as before)
        with_cal = self._add_calendar_features(self.full_df, cfg.datetime_col)

        # Feature selection (numeric, excluding target) and main-series ordering
        self.feature_cols_all = self._choose_feature_cols(
            with_cal, cfg.target_col, cfg.main_series, cfg.put_main_first
        )

        # Store for later
        if cfg.target_col+"_diff" in self.full_df.columns:
            self._target_diff_col = f"{cfg.target_col}_diff"
        else:
            self._target_diff_col = f"{cfg.target_col}_rel" 
        self._frame_with_features = with_cal[[cfg.datetime_col] + self.feature_cols_all]
        self._dt_col = cfg.datetime_col
        self._target_abs_col = cfg.target_col
        

        # Shapes after reshape
        self.X_days: Optional[np.ndarray] = None
        self.y_days: Optional[np.ndarray] = None
        self.dt_days: Optional[np.ndarray] = None

        # Bases for diff-mode inversion
        self._y_train_bases: Optional[np.ndarray] = None
        self._y_test_bases: Optional[np.ndarray] = None

    # ---------- public API ----------
    def build(
        self,
        *,
        keep_only_diff: bool = False,
        diff_as_target: bool = False,
        return_dfs: bool = True,
    ) -> Dict[str, object]:
        """
        Assemble samples and split by whole days; fit scalers on TRAIN only.

        Parameters
        ----------
        keep_only_diff : bool, default False
            If True, keep only feature columns that end with "_diff".
            (Absolute feature columns are dropped.)
        diff_as_target : bool, default False
            If True, use the target *diff* column as y.
            - We will DROP the absolute target column from X to avoid leakage.
            If False, use the absolute target as y and DROP the target diff from X.
        return_dfs : bool, default True
            Whether to also return tidy DataFrames (scaled and raw).

        Returns
        -------
        dict
            {
              "X_train", "X_test", "y_train", "y_test",
              "X_train_raw", "X_test_raw", "y_train_raw", "y_test_raw",
              "meta": {...},
              # present in diff_as_target=True:
              "y_train_bases", "y_test_bases",
              # optional (if return_dfs=True)
              "X_train_df", "X_test_df", "y_train_df", "y_test_df",
              "X_train_df_raw", "X_test_df_raw", "y_train_df_raw", "y_test_df_raw",
              # always present:
              "full_df": <original full dataframe with calendar feats>
            }
        """
        cfg = self.cfg

        # ---------- 1) Choose feature set per switches ----------
        feature_cols = list(self.feature_cols_all)
        print("Initial feature columns:", feature_cols)

        # Maybe keep only *_diff columns in X
        if keep_only_diff:
            feature_cols = [c for c in feature_cols if c.endswith("_diff") or c.endswith("_rel")]

        # Prevent target leakage in X
        if diff_as_target:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # y will be target_diff → ensure absolute target not in X
            if self._target_diff_col in feature_cols:
                feature_cols.remove(self._target_diff_col)
                print(f"Removed absolute target '{self._target_diff_col}' from features to avoid leakage.")
        else:
            print("============================================================")
            # y will be target_abs → ensure target_diff not in X
            if self._target_abs_col in feature_cols:
                feature_cols.remove(self._target_abs_col)
                print(f"Removed target diff '{self._target_abs_col}' from features to avoid leakage.")

        # ---------- 2) Split X (features frame) and y (target series) ----------
        X_frame = self._frame_with_features[[self._dt_col] + feature_cols].copy()

        if diff_as_target:
            if self._target_diff_col not in self.full_df.columns:
                raise KeyError(
                    f"Missing '{self._target_diff_col}' in DataFrame. "
                    "Create target diff upstream (e.g., builder._add_diff_features)."
                )
            y_series = self.full_df[self._target_diff_col].copy()
        else:
            y_series = self.full_df[self._target_abs_col].copy()

        # ---------- 3) Reshape to days (24 per day) ----------
        X_days, dt_days = self._reshape_days(
            X_frame[feature_cols].to_numpy(),
            X_frame[self._dt_col],
        )
        y_days, _ = self._reshape_days(
            y_series.to_numpy().reshape(-1, 1),
            X_frame[self._dt_col],
        )
        if X_days.shape[0] != y_days.shape[0]:
            raise RuntimeError("Mismatch between X and y day counts.")
        if X_days.shape[1] != 24 or y_days.shape[1] != 24:
            raise RuntimeError("Each day must have exactly 24 rows (00:00→23:00).")

        # Keep for later
        self.X_days, self.y_days, self.dt_days = X_days, y_days, dt_days

        # ---------- 4) Assemble samples (sliding by *days*, step=1 day) ----------
        X, y, Xdt, ydt = self._make_samples_same_day()

        # ---------- 5) Make tail split (whole-sample) ----------
        S = X.shape[0]
        split_idx = int(round(S * (1 - cfg.test_size)))
        split_idx = max(cfg.n_lookback_days, min(S - 1, split_idx))

        X_tr, X_te = X[:split_idx], X[split_idx:]
        y_tr, y_te = y[:split_idx], y[split_idx:]
        Xdt_tr, Xdt_te = Xdt[:split_idx], Xdt[split_idx:]
        ydt_tr, ydt_te = ydt[:split_idx], ydt[split_idx:]

        # ---------- 6) RAW copies (pre-scaling) ----------
        X_tr_raw, X_te_raw = X_tr.copy(), X_te.copy()
        y_tr_raw, y_te_raw = y_tr.copy(), y_te.copy()

        # ---------- 7) Per-sample bases (for diff_as_target inversion) ----------
        self._y_train_bases, self._y_test_bases = None, None
        if diff_as_target:
            # Base per day = absolute target value at (k*24 - 1)
            bases_all_days = self._compute_day_bases(self.full_df[self._target_abs_col])
            N = cfg.n_lookback_days
            sample_day_indices = np.arange(N - 1, N - 1 + S)
            y_bases_all = bases_all_days[sample_day_indices]
            self._y_train_bases = y_bases_all[:split_idx]
            self._y_test_bases  = y_bases_all[split_idx:]

        # ---------- 8) Fit scalers on TRAIN only, transform both splits ----------
        if cfg.scale_features:
            F = X_tr.shape[2]
            cfg.feature_scaler.fit(X_tr.reshape(-1, F))
            X_tr = cfg.feature_scaler.transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
            X_te = cfg.feature_scaler.transform(X_te.reshape(-1, F)).reshape(X_te.shape)

        if cfg.scale_target:
            cfg.target_scaler.fit(y_tr.reshape(-1, 1))
            y_tr = cfg.target_scaler.transform(y_tr.reshape(-1, 1)).reshape(y_tr.shape)
            y_te = cfg.target_scaler.transform(y_te.reshape(-1, 1)).reshape(y_te.shape)

        # ---------- 9) Package outputs ----------
        out: Dict[str, object] = {
            # Scaled arrays
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,

            # Unscaled arrays
            "X_train_raw": X_tr_raw, "X_test_raw": X_te_raw,
            "y_train_raw": y_tr_raw, "y_test_raw": y_te_raw,

            "meta": {
                "split_idx_days": split_idx,
                "total_days": self.X_days.shape[0],
                "feature_cols": feature_cols,
                "lookback_hours": cfg.n_lookback_days * 24,
                "target_mode": "diff" if diff_as_target else "absolute",
            },

            # Always give the full (inspection) DataFrame that fed the pipeline
            "full_df": self._frame_with_features.assign(**{
                self._target_abs_col: self.full_df[self._target_abs_col].values,
                self._target_diff_col: self.full_df.get(self._target_diff_col, pd.Series(index=self.full_df.index, dtype=float)),
            }),
        }

        if diff_as_target:
            out["y_train_bases"] = self._y_train_bases
            out["y_test_bases"]  = self._y_test_bases

        if return_dfs:
            # Scaled DFs (tidy long)
            out["X_train_df"] = self._to_long_X_df(X_tr, Xdt_tr, feature_cols)
            out["X_test_df"]  = self._to_long_X_df(X_te, Xdt_te, feature_cols)
            out["y_train_df"] = self._to_long_y_df(y_tr, ydt_tr, self._target_diff_col if diff_as_target else self._target_abs_col)
            out["y_test_df"]  = self._to_long_y_df(y_te, ydt_te, self._target_diff_col if diff_as_target else self._target_abs_col)

            # Raw DFs (tidy long)
            out["X_train_df_raw"] = self._to_long_X_df(X_tr_raw, Xdt_tr, feature_cols)
            out["X_test_df_raw"]  = self._to_long_X_df(X_te_raw, Xdt_te, feature_cols)
            out["y_train_df_raw"] = self._to_long_y_df(y_tr_raw, ydt_tr, self._target_diff_col if diff_as_target else self._target_abs_col)
            out["y_test_df_raw"]  = self._to_long_y_df(y_te_raw, ydt_te, self._target_diff_col if diff_as_target else self._target_abs_col)

        return out

    # ------------------------------------------------------------------ #
    # Inversion helper
    # ------------------------------------------------------------------ #
    def inverse_transform_target(
        self,
        y_scaled: np.ndarray,
        *,
        diff_as_target: bool,
        bases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert model outputs back to absolute target units (EUR/MWh).

        Parameters
        ----------
        y_scaled : np.ndarray
            Predicted sequences in model output space; shape (S, 24, 1) or (S, 24).
        diff_as_target : bool
            Must match what you used in `build()`.
            - True  → y is (scaled) diffs; we inverse-scale then cumulative-sum with `bases`.
            - False → y is (scaled) absolute; we inverse-scale only.
        bases : np.ndarray, optional (required if diff_as_target=True)
            Shape (S,). Per-sample base = absolute value at the hour just
            before the target day starts. Use `out["y_train_bases"]` / `out["y_test_bases"]`.

        Returns
        -------
        np.ndarray
            Absolute predictions with shape (S, 24, 1).
        """
        arr = y_scaled[..., 0] if (y_scaled.ndim == 3 and y_scaled.shape[-1] == 1) else y_scaled

        # Undo scaling
        if self.cfg.scale_target:
            arr = self.cfg.target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)

        if not diff_as_target:
            # We already have absolute units
            return arr[..., None]

        # Diff mode → Need bases to reconstruct
        if bases is None:
            raise ValueError("`bases` is required when diff_as_target=True.")
        if len(bases) != arr.shape[0]:
            raise ValueError("Length of `bases` must equal number of samples (S).")

        out_abs = np.empty_like(arr, dtype=float)
        for s in range(arr.shape[0]):
            prev = bases[s]
            if not np.isfinite(prev):
                out_abs[s, :] = np.nan  # typically only the very first sample if N=1
                continue
            for t in range(arr.shape[1]):
                # x_t = prev + d_t
                out_abs[s, t] = prev + arr[s, t]
                prev = out_abs[s, t]
        return out_abs[..., None]

    # ------------------------------------------------------------------ #
    # Static / utility methods (unchanged except comments)
    # ------------------------------------------------------------------ #
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
        """Add standard cyclical calendar features (hour/day/month) + weekend flag."""
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
        """
        Pick numeric columns excluding the target; optionally put `main_series` first.
        """
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

        Returns
        -------
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
          y = target values of day d        -> (24, 1)
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

    # --------- bases for diff inversion ---------
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
        for k in range(num_days):
            idx = k * 24 - 1
            if idx >= 0:
                bases[k] = float(y_abs.iloc[idx])
        return bases
