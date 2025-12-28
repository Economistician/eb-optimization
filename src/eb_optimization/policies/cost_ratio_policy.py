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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

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
    co : float
        Default overbuild cost coefficient used for entity-level estimation.
    min_n : int
        Minimum number of observations required to estimate an entity-level R.
    """

    R_grid: Sequence[float] = (0.5, 1.0, 2.0, 3.0)
    co: float = 1.0
    min_n: int = 30

    def __post_init__(self) -> None:
        grid = np.asarray(list(self.R_grid), dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("R_grid must be a non-empty 1D sequence of floats.")
        if not np.any(grid > 0):
            raise ValueError("R_grid must contain at least one strictly positive value.")

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
    """
    co_val = policy.co if co is None else co

    # Use Any as an intermediate to satisfy the 'ConvertibleToFloat' protocol
    R_raw: Any = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=policy.R_grid,
        co=co_val,
        sample_weight=sample_weight,
    )
    R = float(R_raw)

    diag: dict[str, Any] = {
        "method": "cost_balance",
        "R_grid": [float(x) for x in policy.R_grid],
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
    """
    # ---- validation: columns ----
    required_cols = {entity_col, y_true_col, y_pred_col}
    if sample_weight_col is not None:
        required_cols.add(sample_weight_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    co_val = float(policy.co if co is None else co)
    if not np.isfinite(co_val) or co_val <= 0:
        raise ValueError(f"co must be finite and strictly positive. Got {co_val}.")

    # ---- governance: identify eligible entities ----
    # Resolution for Error 133: Ensure the groupby result is strictly typed as a Series
    # so that the .index access is valid.
    counts_ser = cast(pd.Series, df.groupby(entity_col, dropna=False, sort=False).size())
    
    # Filtering creates a slice; we cast the result of that slice to Series to access .index
    eligible_counts = cast(pd.Series, counts_ser[counts_ser >= policy.min_n])
    eligible_list = cast(list[Any], eligible_counts.index.tolist())
    
    mask = df[entity_col].isin(eligible_list)
    
    # Cast slices back to DataFrame to preserve attribute access
    eligible_df = cast(pd.DataFrame, df[mask]).copy()
    ineligible_df = cast(pd.DataFrame, df[~mask]).copy()

    # ---- tune eligible entities ----
    results_list: list[pd.DataFrame] = []

    if not eligible_df.empty:
        tuned_raw: Any = estimate_entity_R_from_balance(
            df=eligible_df,
            entity_col=entity_col,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            ratios=policy.R_grid,
            co=co_val,
            sample_weight_col=sample_weight_col,
        )
        tuned = cast(pd.DataFrame, tuned_raw).copy()
        
        tuned["reason"] = None
        # Explicit mapping: use Any bridge to satisfy Pyright's strict 'map' signature
        mapper: Any = counts_ser
        tuned["n"] = tuned[entity_col].map(mapper).astype(int)
        results_list.append(tuned)

    # ---- build rows for ineligible entities ----
    if not ineligible_df.empty:
        ineligible_rows = (
            cast(pd.DataFrame, ineligible_df[[entity_col]])
            .drop_duplicates()
        )
        ineligible_rows = ineligible_rows.assign(
            R=np.nan,
            cu=np.nan,
            co=co_val,
            under_cost=np.nan,
            over_cost=np.nan,
            diff=np.nan,
            reason=f"min_n_not_met(<{policy.min_n})",
        )
        mapper_ineligible: Any = counts_ser
        ineligible_rows["n"] = ineligible_rows[entity_col].map(mapper_ineligible).astype(int)
        results_list.append(ineligible_rows)

    # ---- combine and organize ----
    if not results_list:
        schema_cols = [entity_col, "R", "cu", "co", "n", "reason", "under_cost", "over_cost", "diff"]
        # Resolution for Error 187: Wrap list in pd.Index to satisfy the 'Axes' requirement
        return pd.DataFrame(columns=pd.Index(schema_cols))

    out = pd.concat(results_list, ignore_index=True, sort=False)

    base_cols = [entity_col, "R", "cu", "co", "n", "reason"]
    diag_cols = ["under_cost", "over_cost", "diff"]
    
    remaining = [str(c) for c in out.columns if c not in base_cols + diag_cols]
    target_cols = (base_cols + diag_cols + remaining) if include_diagnostics else (base_cols + remaining)

    # Resolution for Error 187: Use pd.Index for the column slice to satisfy strict typing
    return cast(pd.DataFrame, out[pd.Index(target_cols)])


__all__ = [
    "DEFAULT_COST_RATIO_POLICY",
    "CostRatioPolicy",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",
]
