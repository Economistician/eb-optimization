"""
Cost-ratio (R = c_u / c_o) policy artifacts for eb-optimization.

This module defines *frozen governance* for selecting and applying a cost ratio `R`
(and derived underbuild cost `c_u`) used by asymmetric cost metrics like CWSL.

Layering & responsibilities
---------------------------
- `tuning/cost_ratio.py`:
    Calibration logic (estimating R from residuals / cost balance).
- `policies/cost_ratio_policy.py`:
    Frozen configuration + deterministic application wrappers.

The policy layer exists so downstream consumers can:
- pin method + governance settings in a versioned, auditable object
- apply the same selection logic consistently across environments
- avoid re-encoding configuration details in orchestration code

Design intent
-------------
Policies should be:
- stable: schema changes are deliberate and versioned
- deterministic: same inputs + same policy -> same outputs
- safe: validated governance parameters + clear failure semantics

This module provides:
- `CostRatioPolicy`: frozen configuration
- `DEFAULT_COST_RATIO_POLICY`: exported default policy
- `apply_cost_ratio_policy`: estimate global R from arrays/Series
- `apply_entity_cost_ratio_policy`: estimate per-entity R from a DataFrame

Notes
-----
- This policy does *not* compute CWSL. It only governs parameter selection.
- `co` may be scalar or per-interval array for global estimation; for entity-level
  estimation, `co` is currently modeled as a scalar (consistent with tuning).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from eb_optimization.tuning.cost_ratio import (
    estimate_entity_R_from_balance,
    estimate_R_cost_balance,
)


@dataclass(frozen=True)
class CostRatioPolicy:
    """
    Frozen cost-ratio (R) policy configuration.

    Attributes
    ----------
    R_grid : Sequence[float]
        Candidate ratios to search. Only strictly positive values are considered.
        Order matters for tie-breaking (first-minimizer behavior).
    co : float
        Default overbuild cost coefficient used for entity-level estimation
        (and as a default for global estimation if `co` is not passed at call time).
    min_n : int
        Minimum number of observations required to estimate an entity-level R.
        Entities with fewer observations are returned with NaN R (and diagnostics).
    """

    R_grid: Sequence[float] = (0.5, 1.0, 2.0, 3.0)
    co: float = 1.0
    min_n: int = 30

    def __post_init__(self) -> None:
        grid = np.asarray(list(self.R_grid), dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("R_grid must be a non-empty 1D sequence of floats.")
        if not np.any(grid > 0):
            raise ValueError(
                "R_grid must contain at least one strictly positive value."
            )

        if not np.isfinite(self.co) or float(self.co) <= 0:
            raise ValueError(f"co must be finite and strictly positive. Got {self.co}.")

        if self.min_n < 1:
            raise ValueError(f"min_n must be >= 1. Got {self.min_n}.")


DEFAULT_COST_RATIO_POLICY = CostRatioPolicy()


def apply_cost_ratio_policy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    policy: CostRatioPolicy = DEFAULT_COST_RATIO_POLICY,
    co: float | ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Apply a frozen cost-ratio policy to estimate a global R.

    Parameters
    ----------
    y_true, y_pred
        Realized and forecast demand vectors (non-negative, same shape).
    policy
        Frozen configuration controlling the candidate grid and defaults.
    co
        Overbuild cost coefficient. If None, uses `policy.co`.
        Can be scalar or per-interval array (passed through to tuning).
    sample_weight
        Optional non-negative per-interval weights.

    Returns
    -------
    (R, diagnostics)
        R is a float.
        Diagnostics include the chosen R, co summary, and candidate grid.

    Notes
    -----
    The underlying selection rule is in `tuning.estimate_R_cost_balance`:
    choose R that minimizes |under_cost(R) - over_cost|.
    """
    co_val = policy.co if co is None else co

    R = float(
        estimate_R_cost_balance(
            y_true=y_true,
            y_pred=y_pred,
            R_grid=policy.R_grid,
            co=co_val,
            sample_weight=sample_weight,
        )
    )

    diag: dict[str, Any] = {
        "method": "cost_balance",
        "R_grid": list(map(float, policy.R_grid)),
        "co_is_array": isinstance(co_val, (list, tuple, np.ndarray, pd.Series)),
        "co_default_used": co is None,
        "R": R,
    }
    return (R, diag)


def apply_entity_cost_ratio_policy(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_true_col: str,
    y_pred_col: str,
    policy: CostRatioPolicy = DEFAULT_COST_RATIO_POLICY,
    co: float | None = None,
    sample_weight_col: str | None = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Apply a frozen cost-ratio policy per entity.

    This wraps `tuning.estimate_entity_R_from_balance` but adds policy governance:
    - enforces a minimum sample size per entity (min_n)
    - pins the candidate ratio grid (R_grid)
    - provides stable, auditable defaults for co and configuration

    Parameters
    ----------
    df
        Input data containing entity ids, actuals, forecasts, and optionally weights.
    entity_col
        Column name identifying the entity to calibrate (e.g., restaurant_id).
    y_true_col, y_pred_col
        Column names for realized demand and forecast.
    policy
        Frozen configuration controlling ratios grid and governance.
    co
        Scalar overbuild cost coefficient. If None, uses `policy.co`.
    sample_weight_col
        Optional column name containing non-negative sample weights.
    include_diagnostics
        If True, returns diagnostics columns (under_cost, over_cost, diff) produced
        by tuning. If False, trims to [entity_col, R, cu, co] plus governance columns.

    Returns
    -------
    pandas.DataFrame
        One row per entity with chosen R and supporting diagnostics. Entities
        with fewer than `policy.min_n` observations will be returned with NaN R.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If invalid governance values or negative weights are present.

    Notes
    -----
    The tuning implementation currently treats `co` as a scalar for entity-level
    estimation (consistent with the function signature in tuning).
    """
    # ---- validation: columns ----
    if entity_col not in df.columns:
        raise KeyError(f"entity_col {entity_col!r} not found in df")
    if y_true_col not in df.columns:
        raise KeyError(f"y_true_col {y_true_col!r} not found in df")
    if y_pred_col not in df.columns:
        raise KeyError(f"y_pred_col {y_pred_col!r} not found in df")
    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    co_val = float(policy.co if co is None else co)
    if not np.isfinite(co_val) or co_val <= 0:
        raise ValueError(f"co must be finite and strictly positive. Got {co_val}.")

    # ---- governance: min_n (simple deterministic rule: row count per entity) ----
    counts = df.groupby(entity_col, dropna=False, sort=False).size()
    eligible_entities = counts[counts >= policy.min_n].index
    ineligible_entities = counts[counts < policy.min_n].index

    eligible = df[df[entity_col].isin(eligible_entities)].copy()
    ineligible = df[df[entity_col].isin(ineligible_entities)].copy()

    # ---- tune eligible entities ----
    if not eligible.empty:
        tuned = estimate_entity_R_from_balance(
            df=eligible,
            entity_col=entity_col,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            ratios=policy.R_grid,
            co=co_val,
            sample_weight_col=sample_weight_col,
        ).copy()
        tuned["reason"] = None
        tuned["n"] = tuned[entity_col].map(counts).astype(int)
    else:
        tuned = pd.DataFrame(
            columns=[
                entity_col,
                "R",
                "cu",
                "co",
                "under_cost",
                "over_cost",
                "diff",
                "reason",
                "n",
            ],
        )

    # ---- build rows for ineligible entities (one row per entity) ----
    if not ineligible.empty:
        ineligible_rows = (
            ineligible[[entity_col]]
            .drop_duplicates()
            .assign(
                R=np.nan,
                cu=np.nan,
                co=co_val,
                under_cost=np.nan,
                over_cost=np.nan,
                diff=np.nan,
                reason=f"min_n_not_met(<{policy.min_n})",
            )
            .copy()
        )
        ineligible_rows["n"] = ineligible_rows[entity_col].map(counts).astype(int)
    else:
        ineligible_rows = pd.DataFrame(
            columns=[
                entity_col,
                "R",
                "cu",
                "co",
                "under_cost",
                "over_cost",
                "diff",
                "reason",
                "n",
            ],
        )

    # ---- combine (avoid pandas FutureWarning on concat with empty/all-NA frames) ----
    if ineligible_rows.empty:
        out = tuned
    elif tuned.empty:
        out = ineligible_rows
    else:
        out = pd.concat([tuned, ineligible_rows], ignore_index=True, sort=False)

    # ---- stable column ordering ----
    base_cols = [entity_col, "R", "cu", "co", "n", "reason"]
    diag_cols = ["under_cost", "over_cost", "diff"]
    remaining = [c for c in out.columns if c not in base_cols + diag_cols]

    cols = (
        (base_cols + diag_cols + remaining)
        if include_diagnostics
        else (base_cols + remaining)
    )

    # Ensure all expected columns exist (even if empty)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[cols]


__all__ = [
    "DEFAULT_COST_RATIO_POLICY",
    "CostRatioPolicy",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",
]
