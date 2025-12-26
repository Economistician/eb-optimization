from __future__ import annotations

r"""
Cost ratio (R) tuning utilities.

This module provides calibration helpers for selecting the underbuild-to-overbuild
cost ratio:

$$
    R = \frac{c_u}{c_o}
$$

These routines belong in **eb-optimization** because they *choose/govern* parameters
from data over a candidate set (grid search + calibration diagnostics). They are not
metric primitives (eb-metrics) and are not runtime policies (eb-optimization/policies).

Layering:
- search/ : reusable candidate-space utilities (grids, kernels)
- tuning/ : define candidate grids + objectives + return calibration artifacts
- policies/ : frozen artifacts that apply parameters deterministically at runtime
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .._utils import broadcast_param, handle_sample_weight, to_1d_array
from ..search.kernels import argmin_over_candidates

__all__ = [
    "estimate_R_cost_balance",
    "estimate_entity_R_from_balance",
]


# ---------------------------------------------------------------------
# Global calibration (array-like)
# ---------------------------------------------------------------------
def estimate_R_cost_balance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    R_grid: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, ArrayLike] = 1.0,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Estimate a global cost ratio $R = c_u / c_o$ via cost balance.

    This routine selects a single, global cost ratio $R$ by searching a
    candidate grid and choosing the value where the total weighted underbuild
    cost is closest to the total weighted overbuild cost.

    For each candidate $R$ in ``R_grid``:

    $$
    \begin{aligned}
    c_{u,i} &= R \cdot c_{o,i} \\
    s_i &= \max(0, y_i - \hat{y}_i) \\
    e_i &= \max(0, \hat{y}_i - y_i) \\
    C_u(R) &= \sum_i w_i \; c_{u,i} \; s_i \\
    C_o(R) &= \sum_i w_i \; c_{o,i} \; e_i
    \end{aligned}
    $$

    and the selected value is:

    $$
    R^* = \arg\min_R \; \left| C_u(R) - C_o(R) \right|.
    $$

    Parameters
    ----------
    y_true
        Realized demand (non-negative), shape (n_samples,).
    y_pred
        Forecast demand (non-negative), shape (n_samples,). Must match ``y_true``.
    R_grid
        Candidate ratios to search. Only strictly positive values are considered.
    co
        Overbuild cost coefficient $c_o$. May be scalar or 1D array of shape (n_samples,).
        Underbuild cost is implied as $c_{u,i} = R \cdot c_{o,i}$.
    sample_weight
        Optional non-negative weights. If None, all intervals receive weight 1.0.

    Returns
    -------
    float
        Selected cost ratio in ``R_grid`` minimizing |under_cost - over_cost|.

        Tie-breaking:
        - In the degenerate perfect-forecast case (zero error everywhere), returns
          the candidate closest to 1.0.
        - Otherwise, if multiple candidates yield the same minimal gap, the first
          encountered candidate (in filtered grid order) is returned.
    """
    y_true_arr = to_1d_array(y_true, "y_true")
    y_pred_arr = to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if np.any(y_true_arr < 0) or np.any(y_pred_arr < 0):
        raise ValueError("y_true and y_pred must be non-negative.")

    co_arr = broadcast_param(co, y_true_arr.shape, "co")
    if np.any(co_arr <= 0):
        raise ValueError("co must be strictly positive.")

    w = handle_sample_weight(sample_weight, y_true_arr.shape[0])

    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)
    overbuild = np.maximum(0.0, y_pred_arr - y_true_arr)

    R_grid_arr = np.asarray(R_grid, dtype=float)
    if R_grid_arr.ndim != 1 or R_grid_arr.size == 0:
        raise ValueError("R_grid must be a non-empty 1D sequence of floats.")

    positive_R = R_grid_arr[R_grid_arr > 0]
    if positive_R.size == 0:
        raise ValueError("R_grid must contain at least one positive value.")

    # Degenerate case: perfect forecast (no error anywhere)
    if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
        idx = int(np.argmin(np.abs(positive_R - 1.0)))
        return float(positive_R[idx])

    co_arr_f = co_arr.astype(float, copy=False)
    w_f = w.astype(float, copy=False)

    def _gap_for_R(R: float) -> float:
        cu_arr = float(R) * co_arr_f
        under_cost = float(np.sum(w_f * cu_arr * shortfall))
        over_cost = float(np.sum(w_f * co_arr_f * overbuild))
        return abs(under_cost - over_cost)

    best_R, _best_gap = argmin_over_candidates(
        candidates=positive_R,
        score_fn=_gap_for_R,
        tie_break="first",  # preserves prior behavior
    )

    return float(best_R)


# ---------------------------------------------------------------------
# Entity-level calibration (DataFrame)
# ---------------------------------------------------------------------
def estimate_entity_R_from_balance(
    df: pd.DataFrame,
    entity_col: str,
    y_true_col: str,
    y_pred_col: str,
    ratios: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float = 1.0,
    sample_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    Estimate an entity-level cost ratio via a cost-balance grid search.

    This function estimates a per-entity underbuild-to-overbuild cost ratio:

    $$
        R_e = \frac{c_{u,e}}{c_o}
    $$

    by searching over a user-provided grid of candidate ratios.

    Returns one row per entity with chosen R and supporting diagnostics.
    """
    required = {entity_col, y_true_col, y_pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    ratios_arr = np.asarray(list(ratios), dtype=float)
    if ratios_arr.ndim != 1 or ratios_arr.size == 0 or np.any(ratios_arr <= 0):
        raise ValueError("ratios must be a non-empty 1D sequence of positive floats.")

    if co <= 0:
        raise ValueError("co must be strictly positive.")

    results: list[dict] = []
    grouped = df.groupby(entity_col, sort=False)

    for entity_id, g in grouped:
        y_true = g[y_true_col].to_numpy(dtype=float)
        y_pred = g[y_pred_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            w = g[sample_weight_col].to_numpy(dtype=float)
        else:
            w = np.ones_like(y_true, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred have different shapes: "
                f"{y_true.shape} vs {y_pred.shape}"
            )
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred must be non-negative."
            )
        if np.any(w < 0):
            raise ValueError(
                f"For entity {entity_id!r}, sample weights must be non-negative."
            )

        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)

        # Degenerate case: no error at all for this entity
        if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
            idx = int(np.argmin(np.abs(ratios_arr - 1.0)))
            R_e = float(ratios_arr[idx])
            cu_e = R_e * float(co)
            results.append(
                {
                    entity_col: entity_id,
                    "R": R_e,
                    "cu": cu_e,
                    "co": float(co),
                    "under_cost": 0.0,
                    "over_cost": 0.0,
                    "diff": 0.0,
                }
            )
            continue

        w_f = w.astype(float, copy=False)
        co_f = float(co)

        def _diff_for_R(R: float) -> float:
            cu_val = float(R) * co_f
            under_cost = float(np.sum(w_f * cu_val * shortfall))
            over_cost = float(np.sum(w_f * co_f * overbuild))
            return abs(under_cost - over_cost)

        best_R, best_diff = argmin_over_candidates(
            candidates=ratios_arr,
            score_fn=_diff_for_R,
            tie_break="first",  # preserves prior behavior
        )

        # Recompute diagnostics for the chosen R (single pass)
        best_R_f = float(best_R)
        best_cu = best_R_f * co_f
        best_under_cost = float(np.sum(w_f * best_cu * shortfall))
        best_over_cost = float(np.sum(w_f * co_f * overbuild))

        results.append(
            {
                entity_col: entity_id,
                "R": best_R_f,
                "cu": float(best_cu),
                "co": float(co_f),
                "under_cost": float(best_under_cost),
                "over_cost": float(best_over_cost),
                "diff": float(best_diff),
            }
        )

    return pd.DataFrame(results)