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
        adjusted_forecast = df[forecast_col] * float(self.global_uplift)

        if self.is_segmented():
            # Merge uplift_table with the DataFrame based on segment columns
            uplift_df = df.merge(
                self.uplift_table, on=list(self.segment_cols), how="left"
            )

            # Apply the segment-level uplift (if available) to the forecast.
            # NOTE: `uplift` here is interpreted as a multiplicative factor relative to the
            # global uplift. Missing segments default to 1.0 (no additional uplift).
            return adjusted_forecast * uplift_df["uplift"].fillna(1.0)

        return adjusted_forecast

    def transform(self, df: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
        """Transform the input DataFrame by applying the forecast adjustment.

        This method applies the RAL policy to adjust the forecast column and adds a new column
        with the adjusted forecast.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the forecast to adjust.
        forecast_col : str
            The name of the column in `df` containing the forecast values to adjust.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with the adjusted forecast values added.
        """
        df_copy = df.copy()
        df_copy["readiness_forecast"] = self.adjust_forecast(df_copy, forecast_col)
        return df_copy


# Convenience default policy instance (for the policies package API)
DEFAULT_RAL_POLICY = RALPolicy()


def apply_ral_policy(
    df: pd.DataFrame,
    forecast_col: str,
    policy: RALPolicy = DEFAULT_RAL_POLICY,
) -> pd.DataFrame:
    """Convenience functional wrapper to apply a RALPolicy.

    This is a thin wrapper around :meth:`RALPolicy.transform` used by callers/tests
    that prefer a functional interface.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the forecast column.
    forecast_col : str
        Name of the forecast column to adjust.
    policy : RALPolicy
        Policy artifact to apply. Defaults to :data:`DEFAULT_RAL_POLICY`.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with `readiness_forecast` added.
    """
    return policy.transform(df=df, forecast_col=forecast_col)
