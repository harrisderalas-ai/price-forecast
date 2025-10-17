from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, Literal
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series]
Mode = Literal["relative", "diff"]


@dataclass
class RelativeDeltaTransformer:
    """
    Transform a 1D value series into either:
      - RELATIVE deltas: r_t = (x_t - x_{t-1}) / denom_t
      - DIFF (absolute)  : d_t = (x_t - x_{t-1})

    Backwards compatibility
    -----------------------
    - Default `mode="relative"` preserves previous behavior.
    - Method signatures stay the same; only new optional args were added.

    Robustness for relative mode
    ----------------------------
    - Pure epsilon protection can still explode when x_{t-1} ≈ 0.
      We therefore support a **sign-preserving denominator floor**:
         denom_t = sign(prev) * max(|prev|, min_denominator)
      (applied only in `mode="relative"`).
    - If previous value is not finite, output is set to `fill_value`.

    Inversion
    ---------
    - RELATIVE:  x_t = r_t * (prev + epsilon) + prev
    - DIFF:      x_t = d_t + prev

    Parameters
    ----------
    epsilon : float
        Small constant for numerical safety (used in relative forward/inverse).
    fill_value : float
        Value to place when the previous value is invalid (e.g., first element).
    min_denominator : float
        Absolute floor for |prev| in relative mode (units of the series).
        Use a domain-appropriate value (e.g., 5–20 for EUR/MWh). Set 0.0 to disable.
    mode : {"relative","diff"}
        Output type produced by `transform()`. Default "relative".
    """

    epsilon: float = 1e-6
    fill_value: float = 0.0
    min_denominator: float = 10.0
    mode: Mode = "relative"

    # learned state (kept minimal): immediate previous absolute value
    _prev_abs_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # sklearn-like API
    # ------------------------------------------------------------------ #
    def fit(self, x: ArrayLike) -> "RelativeDeltaTransformer":
        """
        Store the previous-step absolute values for potential inverse use.
        No parameters are learned beyond this cached array.
        """
        arr = self._as_1d_array(x)
        self._prev_abs_ = np.roll(arr, 1)
        self._prev_abs_[0] = np.nan  # no previous value for the first element
        return self

    def transform(self, x: ArrayLike, *, mode: Optional[Mode] = None) -> np.ndarray:
        """
        Transform a series to RELATIVE deltas (default) or DIFFs.

        Parameters
        ----------
        x : array-like
            1D absolute series (e.g., price).
        mode : {"relative","diff"}, optional
            Override the instance's default mode for this call.

        Returns
        -------
        np.ndarray
            RELATIVE: r_t, or DIFF: d_t, same length as input.
        """
        mode = self.mode if mode is None else mode
        arr = self._as_1d_array(x).astype(float, copy=False)

        # (1) previous values (first prev is NaN)
        prev = np.roll(arr, 1)
        prev[0] = np.nan
        prev_finite = np.isfinite(prev)

        if mode == "diff":
            # ---- absolute differences: d_t = x_t - x_{t-1} ----
            out = arr - prev
            # invalid prev -> fill_value
            out[~prev_finite] = self.fill_value
            return out.astype(float)

        # ---- relative mode: r_t = (x_t - prev) / denom ----
        if self.min_denominator > 0.0:
            # sign-preserving absolute floor to avoid huge ratios when prev≈0
            denom = np.where(
                prev_finite,
                np.sign(prev) * np.maximum(np.abs(prev), self.min_denominator),
                np.nan,
            )
        else:
            # legacy behavior (not recommended when prev can be ~0)
            denom = np.where(prev_finite, prev, np.nan)

        rel = (arr - prev) / (denom + self.epsilon)
        rel[~prev_finite] = self.fill_value
        return rel.astype(float)

    def fit_transform(self, x: ArrayLike, *, mode: Optional[Mode] = None) -> np.ndarray:
        """Convenience: fit() then transform()."""
        self.fit(x)
        return self.transform(x, mode=mode)

    def inverse_transform(
        self,
        values: ArrayLike,
        x0: Optional[float] = None,
        *,
        mode: Optional[Literal["relative", "diff"]] = None,
    ) -> np.ndarray:
        """
        Invert RELATIVE or DIFF values back to absolute series.

        RELATIVE mode mirrors the forward transform exactly:
            forward:  denom = sign(prev) * max(|prev|, min_denominator)
                    r_t   = (x_t - prev) / (denom + epsilon)
            inverse:  x_t   = r_t * (denom + epsilon) + prev
        DIFF mode:
            x_t = d_t + prev

        Parameters
        ----------
        values : array-like
            RELATIVE deltas (r_t) or DIFFs (d_t), depending on `mode`.
        x0 : float, optional
            Base absolute value at t=-1 (previous hour) to seed the first step.
            If omitted, we try to use the first `prev` cached by `fit()`.
        mode : {"relative","diff"}, optional
            Override instance default for this call.

        Returns
        -------
        np.ndarray
            Reconstructed absolute series (same length as `values`).
        """
        mode = self.mode if mode is None else mode
        v = self._as_1d_array(values)
        out = np.empty_like(v, dtype=float)

        # Choose initial base
        prev = x0
        if prev is None and self._prev_abs_ is not None and not np.isnan(self._prev_abs_[0]):
            prev = float(self._prev_abs_[0])

        for i in range(len(v)):
            if prev is None or not np.isfinite(prev):
                out[i] = np.nan  # cannot proceed without a valid base
            else:
                if mode == "diff":
                    # x_t = d_t + prev
                    out[i] = v[i] + prev
                else:
                    # --- RELATIVE: mirror forward denominator exactly ---
                    if self.min_denominator > 0.0:
                        denom = np.sign(prev) * max(abs(prev), self.min_denominator)
                    else:
                        denom = prev  # legacy behavior
                    out[i] = v[i] * (denom + self.epsilon) + prev
            prev = out[i]
        return out


    # ------------------------------------------------------------------ #
    # DataFrame helpers
    # ------------------------------------------------------------------ #
    def transform_df(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        *,
        mode: Optional[Mode] = None,
        add_as_new_cols: bool = False,
        suffix_rel: str = "_rel",
        suffix_diff: str = "_diff",
    ) -> pd.DataFrame:
        """
        Apply the transform to multiple DataFrame columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input frame with absolute series.
        columns : Sequence[str]
            Columns to transform.
        mode : {"relative","diff"}, optional
            Override the instance's mode for this call.
        add_as_new_cols : bool, default False
            If True, append transformed data as new columns using the provided suffix.
            If False, replace the original columns in-place.
        suffix_rel, suffix_diff : str
            Suffixes used when `add_as_new_cols=True`.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        mode = self.mode if mode is None else mode
        out = df.copy()

        for c in columns:
            vals = self.transform(out[c].to_numpy(), mode=mode)
            if add_as_new_cols:
                suff = suffix_rel if mode == "relative" else suffix_diff
                out[c + suff] = vals
            else:
                out[c] = vals
        return out

    def inverse_transform_df(
        self,
        df_values: pd.DataFrame,
        columns: Sequence[str],
        bases: Dict[str, float],
        *,
        mode: Optional[Mode] = None,
    ) -> pd.DataFrame:
        """
        Invert DataFrame columns from RELATIVE/DIFF back to absolute values.

        Parameters
        ----------
        df_values : pd.DataFrame
            Frame containing the transformed values (relative or diff).
        columns : Sequence[str]
            Columns to invert.
        bases : Dict[str, float]
            Map column -> base absolute value to start reconstruction.
        mode : {"relative","diff"}, optional
            Override the instance's mode for this call.

        Returns
        -------
        pd.DataFrame
            DataFrame with selected columns inverted to absolute.
        """
        mode = self.mode if mode is None else mode
        out = df_values.copy()
        for c in columns:
            base = bases.get(c)
            out[c] = self.inverse_transform(out[c].to_numpy(), x0=base, mode=mode)
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_1d_array(x: ArrayLike) -> np.ndarray:
        """Cast Series/array to a contiguous 1D NumPy array."""
        if isinstance(x, pd.Series):
            return x.to_numpy()
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError("RelativeDeltaTransformer expects a 1D array/Series.")
        return arr
