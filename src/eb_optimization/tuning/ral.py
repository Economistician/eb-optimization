from __future__ import annotations

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

from typing import List, Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from eb_metrics.metrics import cwsl, frs, nsl
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
    segment_cols: Optional[Sequence[str]] = None,
    sample_weight_col: Optional[str] = None,
) -> Tuple[RALPolicy, pd.DataFrame]:
    """Tune a Readiness Adjustment Layer (RAL) policy via discrete grid search.
    
    This function performs *offline* tuning to select multiplicative uplift factors
    that convert a baseline forecast into an operationally conservative readiness forecast.
    
    The optimization objective is **Cost-Weighted Service Loss (CWSL)**.
    """

    # Ensure necessary columns are present
    if forecast_col not in df.columns or actual_col not in df.columns:
        raise ValueError("Required columns missing in the DataFrame.")

    # Initialize grid for candidate ratios
    uplift_grid = make_float_grid(uplift_min, uplift_max, grid_step)
    
    # Prepare the DataFrame for tuning
    y_true_all = df[actual_col].to_numpy(dtype=float)
    y_pred_all = df[forecast_col].to_numpy(dtype=float)
    
    if sample_weight_col is not None:
        sample_weights = df[sample_weight_col].to_numpy(dtype=float)
    else:
        sample_weights = np.ones_like(y_true_all)
    
    # Initialize variables for best uplift and diagnostics
    best_uplift = None
    best_cu = None
    best_over_cost = None
    best_under_cost = None
    best_diff = None

    diagnostics = []

    # Tune the policy globally (and by segment if needed)
    if segment_cols:
        grouped = df.groupby(segment_cols)
    else:
        grouped = [('', df)]  # No segmentation, use the entire DataFrame

    for segment_key, group in grouped:
        y_true = group[actual_col].to_numpy(dtype=float)
        y_pred = group[forecast_col].to_numpy(dtype=float)
        weights = sample_weights[:len(y_true)]
        
        # Calculate shortfall and overbuild
        shortfall = np.maximum(0, y_true - y_pred)
        overbuild = np.maximum(0, y_pred - y_true)

        # Perform grid search to find the best uplift for the segment
        best_segment_uplift, best_segment_costs = _find_best_uplift(
            uplift_grid, shortfall, overbuild, weights, cu, co
        )

        # Store diagnostics
        diagnostics.append({
            "segment": segment_key,
            "uplift": best_segment_uplift,
            "under_cost": best_segment_costs['under_cost'],
            "over_cost": best_segment_costs['over_cost'],
            "diff": best_segment_costs['diff'],
        })

        # Track the overall best uplift
        if best_diff is None or best_segment_costs['diff'] < best_diff:
            best_diff = best_segment_costs['diff']
            best_uplift = best_segment_uplift
            best_cu = best_segment_costs['under_cost']
            best_over_cost = best_segment_costs['over_cost']

    # Create the RALPolicy object
    policy = RALPolicy(
        global_uplift=best_uplift,
        uplift_table=pd.DataFrame(diagnostics),
        segment_cols=segment_cols or [],
    )
    
    # Return the policy and diagnostics
    return policy, pd.DataFrame(diagnostics)

def _find_best_uplift(uplift_grid, shortfall, overbuild, weights, cu, co):
    """Helper function to find the best uplift for a single segment."""
    best_uplift = None
    best_cost = None
    best_under_cost = None
    best_over_cost = None

    for uplift in uplift_grid:
        # Calculate underbuild and overbuild costs for this uplift
        cu_val = uplift * cu
        under_cost = np.sum(weights * cu_val * shortfall)
        over_cost = np.sum(weights * co * overbuild)
        diff = abs(under_cost - over_cost)

        if best_cost is None or diff < best_cost:
            best_cost = diff
            best_uplift = uplift
            best_under_cost = under_cost
            best_over_cost = over_cost

    return best_uplift, {
        "under_cost": best_under_cost,
        "over_cost": best_over_cost,
        "diff": best_cost,
    }