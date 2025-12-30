"""
Policy artifacts for the Readiness Adjustment Layer (RAL).

This module defines portable, immutable policy objects produced by offline
optimization and consumed by deterministic evaluation and production workflows.

Responsibilities:
- Represent learned RAL parameters (global and optional segment-level uplifts)
- Provide a stable, serializable contract between optimization and evaluation
- Support audit and governance workflows

Non-responsibilities:
- Learning or tuning parameters
- Applying policies to data
- Defining metric or loss functions

Design philosophy:
Policies are artifacts, not algorithms. They encode *decisions* derived from
optimization, not the optimization process itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RALPolicy:
    r"""Portable policy artifact for the Readiness Adjustment Layer (RAL).

    A :class:`~eb_optimization.policies.ral_policy.RALPolicy` is the *output* of an
    offline tuning process (e.g., grid search or evolutionary optimization) and the
    *input* to deterministic evaluation / production application.

    Conceptually, RAL applies a multiplicative uplift to a baseline forecast:

    $$ \hat{y}^{(r)} = u \cdot \hat{y} $$

    where `u` can be either:

    - a **global uplift** (`global_uplift`), applied to all rows, and/or
    - **segment-level** uplifts stored in `uplift_table`, keyed by `segment_cols`

    Segment-level uplifts must fall back to the global uplift for unseen segment
    combinations at application time.

    Attributes
    ----------
    global_uplift
        The global multiplicative uplift used as a fallback and baseline readiness adjustment.
    segment_cols
        The segmentation columns used to key `uplift_table`. Empty means "global-only".
    uplift_table
        Optional DataFrame with columns `[*segment_cols, "uplift"]` containing
        segment-level uplifts. If `None` or empty, the policy is global-only.

    Notes
    -----
    This dataclass is intentionally simple and serializable. It is meant to be:

    - produced offline in `eb-optimization`
    - applied deterministically in `eb-evaluation`
    - loggable/auditable as part of operational governance

    The policy does *not* encode metric definitions or optimization state—only the
    artifacts needed to execute the adjustment.
    """

    global_uplift: float = 1.0
    segment_cols: list[str] = field(default_factory=list)
    uplift_table: pd.DataFrame | None = None

    def is_segmented(self) -> bool:
        """Return True if the policy contains segment-level uplifts."""
        return (
            bool(self.segment_cols)
            and self.uplift_table is not None
            and not self.uplift_table.empty
        )

    def adjust_forecast(self, df: pd.DataFrame, forecast_col: str) -> pd.Series:
        """Apply the RAL policy to adjust the forecast values.

        This method applies the global uplift to all rows, and applies segment-level uplifts
        if the policy is segmented and matching segments exist in the `uplift_table`.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the forecast to adjust.
        forecast_col : str
            The name of the column in `df` containing the forecast values to adjust.

        Returns
        -------
        pd.Series
            A series with the adjusted forecast values.
        """
        # Ensure we treat the slice as a Series for arithmetic
        baseline = cast(pd.Series, df[forecast_col])
        adjusted_forecast = cast(pd.Series, baseline * float(self.global_uplift))

        if self.is_segmented():
            # Explicitly cast to DataFrame to resolve Pyright's None-safety check
            table = cast(pd.DataFrame, self.uplift_table)

            # Use pd.Index for merge keys to satisfy Axes protocol
            merge_on = pd.Index(self.segment_cols).tolist()

            # Merge uplift_table with the DataFrame based on segment columns.
            uplift_df = df.merge(table, on=merge_on, how="left")

            # Extract the uplift column and fill missing segments with 1.0
            multiplier_ser = cast(pd.Series, uplift_df["uplift"]).fillna(1.0)

            # Solve the reportOperatorIssue by casting to concrete numpy arrays of floats.
            # This satisfies Pyright that the '*' operator is valid.
            arr_baseline = cast("np.ndarray[Any, np.dtype[np.float64]]", adjusted_forecast.values)
            arr_multiplier = cast("np.ndarray[Any, np.dtype[np.float64]]", multiplier_ser.values)

            result_raw = arr_baseline * arr_multiplier
            return pd.Series(result_raw, index=df.index, name="readiness_forecast")

        return adjusted_forecast

    def transform(self, df: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
        """Transform the input DataFrame by applying the forecast adjustment."""
        df_copy = df.copy()
        df_copy["readiness_forecast"] = self.adjust_forecast(df_copy, forecast_col)
        return df_copy


# Convenience default policy instance
DEFAULT_RAL_POLICY = RALPolicy()


def apply_ral_policy(
    df: pd.DataFrame,
    forecast_col: str,
    policy: RALPolicy = DEFAULT_RAL_POLICY,
) -> pd.DataFrame:
    """Convenience functional wrapper to apply a RALPolicy."""
    return policy.transform(df=df, forecast_col=forecast_col)


@dataclass(frozen=True)
class RALBands:
    """Risk-region thresholds for a two-band additive RAL policy.

    mid
        Lower bound for the mid-risk region (inclusive).
    high
        Lower bound for the high-risk region (inclusive).

    The two-band transform is:
      - add d_high when yhat >= high
      - add d_mid  when mid <= yhat < high
    """

    mid: float = 0.75
    high: float = 0.85

    def __post_init__(self) -> None:
        if self.mid < 0.0:
            raise ValueError("bands.mid must be non-negative.")
        if self.high < 0.0:
            raise ValueError("bands.high must be non-negative.")
        if self.high < self.mid:
            raise ValueError("bands.high must be >= bands.mid.")


@dataclass(frozen=True)
class RALDeltas:
    """Two-band additive deltas for a two-band RAL policy."""

    d_mid: float = 0.0
    d_high: float = 0.0

    def __post_init__(self) -> None:
        if self.d_mid < 0.0:
            raise ValueError("d_mid must be non-negative.")
        if self.d_high < 0.0:
            raise ValueError("d_high must be non-negative.")


@dataclass(frozen=True)
class RALTwoBandPolicy:
    r"""Portable policy artifact for two-band *additive* RAL.

    This policy encodes the exact "two-band" additive RAL used in the ISO-NE
    example notebook:

    - If baseline forecast $\hat{y}$ is in the mid-risk band:
      $$ \hat{y}^{(r)} = \hat{y} + d_{\text{mid}} $$
    - If baseline forecast $\hat{y}$ is in the high-risk band:
      $$ \hat{y}^{(r)} = \hat{y} + d_{\text{high}} $$

    Deltas can be:

    - global (fallback) via `global_deltas`, and/or
    - per-key overrides via `per_key_deltas`, keyed by a segment key column
      (e.g., `interface`).

    Notes
    -----
    This class is intentionally a *policy artifact* (parameters + deterministic
    application). It does not learn deltas; it only stores and applies them.
    """

    bands: RALBands = field(default_factory=RALBands)
    global_deltas: RALDeltas = field(default_factory=RALDeltas)
    per_key_deltas: dict[str, RALDeltas] | None = None

    def get_deltas(self, key: str | None = None) -> RALDeltas:
        """Return deltas for a key (or the global deltas if none/unknown)."""
        if key is None or self.per_key_deltas is None:
            return self.global_deltas
        return self.per_key_deltas.get(key, self.global_deltas)

    def adjust_forecast(
        self,
        df: pd.DataFrame,
        forecast_col: str,
        *,
        key_col: str | None = None,
    ) -> pd.Series:
        """Apply the two-band additive RAL policy to a forecast column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the forecast to adjust.
        forecast_col : str
            Column name containing baseline forecast values.
        key_col : str, optional
            Column name containing keys for per-key deltas (e.g., "interface").
            If omitted, the global deltas are applied.

        Returns
        -------
        pd.Series
            Adjusted forecast values as a series named "readiness_forecast".
        """
        baseline = cast(pd.Series, df[forecast_col])
        yhat = cast("np.ndarray[Any, np.dtype[np.float64]]", baseline.astype(float).values)

        if key_col is None:
            d = self.global_deltas
            out = _apply_two_band_additive(yhat, self.bands.mid, self.bands.high, d.d_mid, d.d_high)
            return pd.Series(out, index=df.index, name="readiness_forecast")

        if key_col not in df.columns:
            raise ValueError(f"key_col '{key_col}' not found in DataFrame.")

        keys = cast(pd.Series, df[key_col]).astype(str).to_numpy()
        out_all = yhat.copy()

        # Apply per-key overrides where present; fallback to global deltas otherwise.
        uniq = np.unique(keys)
        for k in uniq:
            mask = keys == k
            d = self.get_deltas(str(k))
            out_all[mask] = _apply_two_band_additive(
                out_all[mask],
                self.bands.mid,
                self.bands.high,
                d.d_mid,
                d.d_high,
            )

        return pd.Series(out_all, index=df.index, name="readiness_forecast")

    def transform(
        self,
        df: pd.DataFrame,
        forecast_col: str,
        *,
        key_col: str | None = None,
    ) -> pd.DataFrame:
        """Transform the input DataFrame by applying the forecast adjustment."""
        df_copy = df.copy()
        df_copy["readiness_forecast"] = self.adjust_forecast(df_copy, forecast_col, key_col=key_col)
        return df_copy

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        per: dict[str, dict[str, float]] | None = None
        if self.per_key_deltas is not None:
            per = {
                k: {"d_mid": v.d_mid, "d_high": v.d_high} for k, v in self.per_key_deltas.items()
            }
        return {
            "bands": {"mid": float(self.bands.mid), "high": float(self.bands.high)},
            "global_deltas": {
                "d_mid": float(self.global_deltas.d_mid),
                "d_high": float(self.global_deltas.d_high),
            },
            "per_key_deltas": per,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RALTwoBandPolicy:
        """Deserialize from a dict produced by `to_dict()`."""
        bands_d = cast(dict[str, Any], d.get("bands", {}))
        global_d = cast(dict[str, Any], d.get("global_deltas", {}))

        bands = RALBands(
            mid=float(bands_d.get("mid", 0.75)),
            high=float(bands_d.get("high", 0.85)),
        )
        global_deltas = RALDeltas(
            d_mid=float(global_d.get("d_mid", 0.0)),
            d_high=float(global_d.get("d_high", 0.0)),
        )

        per_in = d.get("per_key_deltas")
        per_out: dict[str, RALDeltas] | None
        if per_in is None:
            per_out = None
        else:
            per_tbl = cast(dict[str, Any], per_in)
            per_out = {
                str(k): RALDeltas(
                    d_mid=float(cast(dict[str, Any], v)["d_mid"]),
                    d_high=float(cast(dict[str, Any], v)["d_high"]),
                )
                for k, v in per_tbl.items()
            }

        return cls(bands=bands, global_deltas=global_deltas, per_key_deltas=per_out)


def _apply_two_band_additive(
    yhat: np.ndarray,
    mid: float,
    high: float,
    d_mid: float,
    d_high: float,
) -> np.ndarray:
    """Vectorized two-band additive RAL transform on a 1D float array."""
    out = yhat.copy()
    out = np.where(out >= high, out + d_high, out)
    out = np.where((out >= mid) & (out < high), out + d_mid, out)
    return out
