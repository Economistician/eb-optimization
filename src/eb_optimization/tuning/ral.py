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

Usage:
This module is intended for offline experimentation, backtesting, and policy
calibration. The resulting policies can be applied deterministically via
`eb-evaluation` without requiring any optimization machinery at runtime.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl, frs, nsl

from eb_optimization.policies.ral_policy import RALPolicy
from eb_optimization.search.grid import make_float_grid


def _fit_segment_uplift(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid: np.ndarray,
    cu: float,
    co: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    r"""Fit the best uplift for a single scope by minimizing CWSL over a grid.

    This helper performs the core discrete optimization used by RAL tuning:

    $$
    u^* = \arg\min_{u \in \mathcal{U}} \mathrm{CWSL}(u \cdot \hat{y}, y)
    $$

    It also computes secondary diagnostics (FRS and underbuild-oriented rate) before
    and after applying the chosen uplift.

    Parameters
    ----------
    y_true
        Array of realized demand / actuals for the scope.
    y_pred
        Array of baseline forecasts for the scope.
    grid
        Candidate uplift multipliers (strictly positive).
    cu
        Underbuild cost coefficient used by CWSL/FRS.
    co
        Overbuild cost coefficient used by CWSL/FRS.
    sample_weight
        Optional non-negative weights aligned to `y_true`/`y_pred`.

    Returns
    -------
    tuple[float, dict[str, float]]
        The chosen uplift multiplier and a dict of diagnostics including:
        `cwsl_before`, `cwsl_after`, `cwsl_delta`, `frs_before`, `frs_after`,
        `frs_delta`, `ub_rate_before`, `ub_rate_after`, `ub_rate_delta`.

    Notes
    -----
    This function is internal to the tuner. The public API is `tune_ral_policy`.
    """
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    cwsl_before = float(cwsl(y_true=y_true_f, y_pred=y_pred_f, cu=cu, co=co, sample_weight=sw))
    frs_before = float(frs(y_true=y_true_f, y_pred=y_pred_f, cu=cu, co=co, sample_weight=sw))
    ub_rate_before = 1.0 - float(nsl(y_true=y_true_f, y_pred=y_pred_f, sample_weight=sw))

    best_uplift = float(grid[0])
    best_score = float("inf")
    y_best = y_pred_f

    for u in grid:
        y_adj = y_pred_f * float(u)
        score = float(cwsl(y_true=y_true_f, y_pred=y_adj, cu=cu, co=co, sample_weight=sw))
        if score < best_score:
            best_score = score
            best_uplift = float(u)
            y_best = y_adj

    cwsl_after = float(cwsl(y_true=y_true_f, y_pred=y_best, cu=cu, co=co, sample_weight=sw))
    frs_after = float(frs(y_true=y_true_f, y_pred=y_best, cu=cu, co=co, sample_weight=sw))
    ub_rate_after = 1.0 - float(nsl(y_true=y_true_f, y_pred=y_best, sample_weight=sw))

    diag = {
        "uplift": best_uplift,
        "cwsl_before": cwsl_before,
        "cwsl_after": cwsl_after,
        "cwsl_delta": cwsl_after - cwsl_before,
        "frs_before": frs_before,
        "frs_after": frs_after,
        "frs_delta": frs_after - frs_before,
        "ub_rate_before": ub_rate_before,
        "ub_rate_after": ub_rate_after,
        "ub_rate_delta": ub_rate_after - ub_rate_before,
    }
    return best_uplift, diag


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
    r"""Tune a Readiness Adjustment Layer (RAL) policy via discrete grid search.

    This function performs *offline* tuning to select multiplicative uplift factors
    that convert a baseline forecast into an operationally conservative readiness forecast.

    The optimization objective is **Cost-Weighted Service Loss (CWSL)**. Secondary
    diagnostics are tracked for governance and interpretability (FRS and an underbuild-
    oriented service rate derived from NSL).

    The tuner always learns a **global uplift** and, if `segment_cols` is provided,
    additionally learns **segment-level** uplifts (one per unique segment combination).

    Parameters
    ----------
    df
        Historical dataset containing forecasts, actuals, and optional segment and weight columns.
    forecast_col
        Column containing the baseline statistical forecast.
    actual_col
        Column containing realized demand / actual values.
    cu
        Underbuild cost coefficient passed to CWSL/FRS. Must be strictly positive.
    co
        Overbuild cost coefficient passed to CWSL/FRS. Must be strictly positive.
    uplift_min
        Minimum candidate uplift multiplier (inclusive). Must be strictly positive.
    uplift_max
        Maximum candidate uplift multiplier (inclusive). Must be >= `uplift_min`.
    grid_step
        Step size between uplift candidates. Must be strictly positive.
    segment_cols
        Optional segmentation columns used to learn per-segment uplifts. If `None` or empty,
        only a global uplift is learned.
    sample_weight_col
        Optional column containing non-negative weights aligned to the rows in `df`.
        Weights are passed through to EB metrics.

    Returns
    -------
    tuple[RALPolicy, pandas.DataFrame]
        - `policy`: a portable :class:`~eb_optimization.policies.ral_policy.RALPolicy`
          containing global and optional segment-level uplifts
        - `diagnostics`: a DataFrame of global and per-segment before/after metrics and deltas

    Raises
    ------
    ValueError
        If `df` is empty.
    KeyError
        If required columns are missing.

    Notes
    -----
    This function is designed to be called offline (experimentation/backtesting). The returned
    policy can be applied deterministically in `eb-evaluation` without requiring any of the tuning
    machinery at runtime.

    Future versions of `eb-optimization` may provide additional optimizers (e.g., evolutionary
    algorithms) that produce the same `RALPolicy` artifact, preserving downstream compatibility.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if forecast_col not in df.columns:
        raise KeyError(f"forecast_col {forecast_col!r} not found.")
    if actual_col not in df.columns:
        raise KeyError(f"actual_col {actual_col!r} not found.")

    if cu <= 0.0 or co <= 0.0:
        raise ValueError("cu and co must be strictly positive.")

    seg_cols = list(segment_cols) if segment_cols is not None else []
    for c in seg_cols:
        if c not in df.columns:
            raise KeyError(f"segment_col {c!r} not found.")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found.")

    grid = make_float_grid(uplift_min, uplift_max, grid_step)

    y_true_all = df[actual_col].to_numpy(dtype=float)
    y_pred_all = df[forecast_col].to_numpy(dtype=float)
    sw_all = df[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None

    # Global
    global_uplift, global_diag = _fit_segment_uplift(
        y_true=y_true_all, y_pred=y_pred_all, grid=grid, cu=cu, co=co, sample_weight=sw_all
    )
    global_diag = {"scope": "global", **global_diag}

    records: List[Dict[str, Any]] = [global_diag]
    uplift_table: Optional[pd.DataFrame] = None

    # Segment-level
    if seg_cols:
        seg_rows: List[Dict[str, Any]] = []
        grouped = df.groupby(seg_cols, dropna=False)
        for key, g in grouped:
            if not isinstance(key, tuple):
                key = (key,)

            y_true = g[actual_col].to_numpy(dtype=float)
            y_pred = g[forecast_col].to_numpy(dtype=float)
            sw = g[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None

            uplift, diag = _fit_segment_uplift(
                y_true=y_true, y_pred=y_pred, grid=grid, cu=cu, co=co, sample_weight=sw
            )

            row: Dict[str, Any] = {"scope": "segment", **diag}
            for col_name, value in zip(seg_cols, key):
                row[col_name] = value
            seg_rows.append(row)

        records.extend(seg_rows)
        uplift_table = pd.DataFrame(seg_rows)[[*seg_cols, "uplift"]].copy()

    policy = RALPolicy(
        global_uplift=float(global_uplift),
        segment_cols=tuple(seg_cols),
        uplift_table=uplift_table,
    )
    diagnostics = pd.DataFrame(records)
    return policy, diagnostics