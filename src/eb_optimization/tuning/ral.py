"""
Offline tuning for the Readiness Adjustment Layer (RAL).

This module contains optimization logic for selecting RAL policy parameters
by minimizing Electric Barometer objectives (primarily Cost-Weighted Service
Loss) over historical data.

Responsibilities:
- Search bounded uplift grids to select optimal RAL parameters
- Produce portable RALPolicy artifacts
- Emit audit-ready diagnostics for governance and analysis

Non-responsibilities:
- Applying policies to forecasts
- Defining metric math (delegated to `eb-metrics`)
- Production-time inference or real-time decisioning
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from eb_optimization.policies.ral_policy import RALPolicy
from eb_optimization.search.grid import make_float_grid


def tune_ral_policy(
    df: pd.DataFrame,
    *,
    forecast_col: str,
    actual_col: str,
    cu: float = 2.0,
    co: float = 1.0,
    uplift_min: float = 1.0,
    uplift_max: float = 1.15,
    grid_step: float = 0.01,
    segment_cols: Sequence[str] | None = None,
    sample_weight_col: str | None = None,
) -> tuple[RALPolicy, pd.DataFrame]:
    """Tune a Readiness Adjustment Layer (RAL) policy via discrete grid search.

    This function performs *offline* tuning to select multiplicative uplift factors
    that convert a baseline forecast into an operationally conservative readiness forecast.

    The optimization objective is **Cost-Weighted Service Loss (CWSL)**.
    """
    # Ensure necessary columns are present
    missing = [c for c in (forecast_col, actual_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing in the DataFrame: {missing}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise ValueError(f"sample_weight_col {sample_weight_col!r} not found in df")

    # Initialize grid for candidate ratios
    uplift_grid = make_float_grid(uplift_min, uplift_max, grid_step)

    # Initialize variables for best uplift (global best across segments)
    best_uplift: float | None = None
    best_diff: float | None = None

    diagnostics: list[dict[str, Any]] = []

    # Tune the policy globally (and by segment if needed)
    grouped = df.groupby(list(segment_cols)) if segment_cols else [("", df)]

    for segment_key, group in grouped:
        y_true = group[actual_col].to_numpy(dtype=float)
        y_pred = group[forecast_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            weights = group[sample_weight_col].to_numpy(dtype=float)
        else:
            weights = np.ones_like(y_true, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"For segment {segment_key!r}, y_true and y_pred have different shapes: "
                f"{y_true.shape} vs {y_pred.shape}"
            )
        if np.any(weights < 0):
            raise ValueError(
                f"For segment {segment_key!r}, sample weights must be non-negative."
            )

        # Calculate shortfall and overbuild
        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)

        # Perform grid search to find the best uplift for the segment
        best_segment_uplift, best_segment_costs = _find_best_uplift(
            uplift_grid=uplift_grid,
            shortfall=shortfall,
            overbuild=overbuild,
            weights=weights,
            cu=cu,
            co=co,
        )

        # Store diagnostics
        diagnostics.append(
            {
                "segment": segment_key,
                "uplift": float(best_segment_uplift),
                "under_cost": float(best_segment_costs["under_cost"]),
                "over_cost": float(best_segment_costs["over_cost"]),
                "diff": float(best_segment_costs["diff"]),
            }
        )

        # Track the overall best uplift across segments
        seg_diff = float(best_segment_costs["diff"])
        if best_diff is None or seg_diff < best_diff:
            best_diff = seg_diff
            best_uplift = float(best_segment_uplift)

    if best_uplift is None:
        raise ValueError("Unable to tune RAL policy: no data available after grouping.")

    diagnostics_df = pd.DataFrame(diagnostics)

    # Create the RALPolicy object
    policy = RALPolicy(
        global_uplift=float(best_uplift),
        segment_cols=list(segment_cols) if segment_cols else [],
        uplift_table=diagnostics_df if not diagnostics_df.empty else None,
    )

    return policy, diagnostics_df


def _find_best_uplift(
    *,
    uplift_grid: np.ndarray,
    shortfall: np.ndarray,
    overbuild: np.ndarray,
    weights: np.ndarray,
    cu: float,
    co: float,
) -> tuple[float, dict[str, float]]:
    """Find the best uplift for a single segment by minimizing |under_cost - over_cost|."""
    best_uplift: float | None = None
    best_diff: float | None = None
    best_under_cost: float | None = None
    best_over_cost: float | None = None

    over_cost = float(np.sum(weights * float(co) * overbuild))

    for uplift in uplift_grid:
        uplift_f = float(uplift)
        cu_val = uplift_f * float(cu)
        under_cost = float(np.sum(weights * cu_val * shortfall))
        diff = abs(under_cost - over_cost)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_uplift = uplift_f
            best_under_cost = under_cost
            best_over_cost = over_cost

    if (
        best_uplift is None
        or best_diff is None
        or best_under_cost is None
        or best_over_cost is None
    ):
        raise ValueError("uplift_grid must be non-empty.")

    return best_uplift, {
        "under_cost": best_under_cost,
        "over_cost": best_over_cost,
        "diff": best_diff,
    }
