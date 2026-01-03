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
- search/   : reusable candidate-space utilities (grids, kernels)
- tuning/   : define candidate grids + objectives + return calibration artifacts
- policies/ : frozen artifacts that apply parameters deterministically at runtime

Selection strategy
------------------
`estimate_R_cost_balance` can select the optimal ratio in two equivalent ways:

1) selection="curve" (default)
   - Materialize the full sensitivity curve over the candidate grid.
   - Select $R^*$ by a direct reduction on the curve (NumPy argmin over `gap`).
   - This is typically faster and emphasizes that the curve is the primary audit artifact.

2) selection="kernel"
   - Materialize the full sensitivity curve over the candidate grid.
   - Select $R^*$ using the generic candidate-search kernel (`argmin_over_candidates`)
     by scoring each candidate using the curve-derived `gap`.
   - This preserves a consistent “kernel-governed” selection pathway that matches other
     search utilities in the library.

Both strategies are deterministic and use the same tie-breaking semantics: the first
candidate (in filtered grid order) achieving the minimum `gap` is chosen.

Returning governance artifacts
------------------------------
If `return_curve=False` (default), `estimate_R_cost_balance` returns the selected ratio
`R_star` as a plain float (backwards compatible).

If `return_curve=True`, it returns a `CostRatioEstimate` object that mirrors the intent
of `TauEstimate`: it includes the selected ratio plus the candidate grid actually
searched, the selection strategy, the tie-break rule, sample count, and a diagnostics
payload alongside the full sensitivity curve.

Entity-level calibration (recommended API)
------------------------------------------
Entity-level calibration produces richer outputs than a flat DataFrame can cleanly
represent (because each entity has its own sensitivity curve).

To keep both:
- analysis convenience (legacy DataFrame), and
- gold-standard “persistable artifact” structure,

`estimate_entity_R_from_balance` supports **two return modes**:

1) Default (backwards compatible):
   - Returns a DataFrame with one row per entity and scalar outputs (no curves).
   - This preserves legacy column names (`R`, `diff`) for compatibility.

2) `return_result=True`:
   - Returns an `EntityCostRatioEstimate` artifact containing:
     - `table`: one row per entity with standardized scalar outputs + per-entity diagnostics
               (no DataFrame-in-cell curves)
     - `curves`: dict mapping entity_id -> sensitivity curve DataFrame
     - shared governance metadata (method, grid, selection, tie_break)

Serialization guidance
---------------------
- The entity-level `table` is designed to be Parquet/CSV friendly.
- The `diagnostics` dict is intended to remain JSON-serializable (floats/bools/ints/str).
- Full per-entity curves are kept in the `curves` mapping to avoid object dtype columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from .._utils import broadcast_param, handle_sample_weight, to_1d_array
from ..search.kernels import argmin_over_candidates

__all__ = [
    "CostRatioEstimate",
    "EntityCostRatioEstimate",
    "estimate_R_cost_balance",
    "estimate_entity_R_from_balance",
]


@dataclass(frozen=True)
class CostRatioEstimate:
    """
    Result container for cost-ratio (R) calibration.

    This is intentionally "TauEstimate-like": downstream code can persist or log a
    single object that fully describes the calibration decision.

    Attributes
    ----------
    R_star
        Selected cost ratio from the candidate grid.
    method
        Method identifier used to produce the estimate (e.g., "cost_balance").
    n
        Number of samples used in the calibration (after validation).
    grid
        The filtered candidate grid actually searched (strictly positive values).
        The order of this grid defines tie-breaking.
    selection
        Selection strategy used once the curve is computed ("curve" or "kernel").
    tie_break
        Tie-breaking rule applied when multiple candidates achieve the same objective.
        For this routine it is always "first".
    diagnostics
        Method-specific diagnostic metadata intended for governance and reporting.
        Keys include:
        - "over_cost_const": float
        - "min_gap": float
        - "degenerate_perfect_forecast": bool
        - "rel_min_gap": float
        - "grid_sensitivity": dict[str, float]
        - "grid_instability_log": float
        - "identifiability_thresholds": dict[str, float]
        - "is_identifiable": bool
    rel_min_gap
        Relative imbalance at the chosen point: min_gap / over_cost_const (or inf if
        over_cost_const==0 and min_gap>0). This is a *cost-separation* diagnostic.
    R_min
        Minimum R_star across built-in grid perturbations (base/exclude/shift).
    R_max
        Maximum R_star across built-in grid perturbations (base/exclude/shift).
    grid_instability_log
        log(R_max / R_min) across built-in perturbations. This is a *grid-sensitivity*
        diagnostic capturing weak identifiability due to discretization.
    is_identifiable
        Boolean summary derived from conservative thresholds on rel_min_gap and
        grid_instability_log. This does not change selection; it only reports stability.
    curve
        Sensitivity curve / diagnostics for each candidate ratio.
        Columns are:
        - R
        - under_cost
        - over_cost
        - gap  (= |under_cost - over_cost|)
    """

    R_star: float
    method: str
    n: int
    grid: np.ndarray
    selection: str
    tie_break: str
    diagnostics: dict[str, Any]
    rel_min_gap: float
    R_min: float
    R_max: float
    grid_instability_log: float
    is_identifiable: bool
    curve: pd.DataFrame


@dataclass(frozen=True)
class EntityCostRatioEstimate:
    """
    Entity-level cost ratio calibration artifact.

    This container is designed to be persistable and ergonomic:

    - `table` can be saved to Parquet/CSV without DataFrame-in-cell object columns.
    - `curves` retains the full per-entity sensitivity curves for audit/plotting.
    - Shared metadata captures the governance context of the run.

    Attributes
    ----------
    entity_col
        Name of the entity identifier column used in `table`.
    method
        Method identifier used to produce the estimate (e.g., "cost_balance").
    grid
        The filtered candidate grid actually searched (strictly positive values).
        The order of this grid defines tie-breaking for each entity.
    selection
        Selection strategy used once each curve is computed ("curve" or "kernel").
    tie_break
        Tie-breaking rule applied when multiple candidates achieve the same objective.
        For this routine it is always "first".
    table
        One row per entity with scalar results and summary diagnostics.
        Columns include:
        - entity_col
        - R_star
        - n
        - under_cost
        - over_cost
        - gap
        - diagnostics  (dict; intended to be JSON-serializable)
    curves
        Mapping of entity_id -> sensitivity curve DataFrame for that entity.
        Each curve has columns: [R, under_cost, over_cost, gap].
    """

    entity_col: str
    method: str
    grid: np.ndarray
    selection: str
    tie_break: str
    table: pd.DataFrame
    curves: dict[Any, pd.DataFrame]


# ---------------------------------------------------------------------
# Global calibration (array-like)
# ---------------------------------------------------------------------
def estimate_R_cost_balance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    R_grid: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float | ArrayLike = 1.0,
    sample_weight: ArrayLike | None = None,
    *,
    return_curve: bool = False,
    selection: Literal["curve", "kernel"] = "curve",
) -> float | CostRatioEstimate:
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
        The filtered grid order is preserved for tie-breaking.
    co
        Overbuild cost coefficient $c_o$. May be scalar or 1D array of shape (n_samples,).
        Underbuild cost is implied as $c_{u,i} = R \cdot c_{o,i}$.
    sample_weight
        Optional non-negative weights. If None, all intervals receive weight 1.0.
    return_curve
        If True, return a CostRatioEstimate containing both the chosen R and the
        full sensitivity curve diagnostics over the filtered grid, plus governance
        metadata (grid used, selection strategy, tie-break rule, sample size).
    selection
        Strategy used to select $R^*$ once the sensitivity curve has been computed:

        - "curve"  : Select via NumPy argmin over the curve's `gap` column.
                     This is typically faster and treats the curve as the primary
                     artifact.
        - "kernel" : Select via `argmin_over_candidates`, scoring each candidate using
                     curve-derived gap values. This maintains consistency with other
                     candidate-search kernels.

        Both strategies are deterministic and share the same tie-breaking behavior.

    Returns
    -------
    float or CostRatioEstimate
        By default returns the selected cost ratio minimizing |under_cost - over_cost|.

        If ``return_curve=True``, returns a CostRatioEstimate with:
        - R_star
        - method, n, grid, selection, tie_break, diagnostics
        - rel_min_gap, R_min, R_max, grid_instability_log, is_identifiable
        - curve : pd.DataFrame with columns [R, under_cost, over_cost, gap]

        Tie-breaking:
        - In the degenerate perfect-forecast case (zero error everywhere), chooses
          the candidate closest to 1.0.
        - Otherwise, if multiple candidates yield the same minimal gap, the first
          encountered candidate (in filtered grid order) is returned.
    """
    if selection not in ("curve", "kernel"):
        raise ValueError("selection must be one of: 'curve', 'kernel'")

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

    co_arr_f = co_arr.astype(float, copy=False)
    w_f = w.astype(float, copy=False)

    # Precompute the over_cost component (independent of R)
    over_cost_const = float(np.sum(w_f * co_arr_f * overbuild))

    def _compute_under_over_gap(R: float) -> tuple[float, float, float]:
        cu_arr = float(R) * co_arr_f
        under_cost = float(np.sum(w_f * cu_arr * shortfall))
        gap = abs(under_cost - over_cost_const)
        return under_cost, over_cost_const, gap

    def _select_R_from_curve(curve_df: pd.DataFrame, grid: np.ndarray) -> float:
        if selection == "curve":
            idx = int(np.argmin(curve_df["gap"].to_numpy()))
            return float(curve_df["R"].to_numpy()[idx])

        r_vals = curve_df["R"].to_numpy()
        gap_vals = curve_df["gap"].to_numpy()
        idx_map = {float(r): i for i, r in enumerate(r_vals.tolist())}

        def _gap_for_R(R: float, *, _gap_vals=gap_vals, _idx_map=idx_map) -> float:
            return float(_gap_vals[_idx_map[float(R)]])

        best_R, _best_gap = argmin_over_candidates(
            candidates=grid,
            score_fn=_gap_for_R,
            tie_break="first",
        )
        return float(best_R)

    def _run_on_grid(grid: np.ndarray) -> float:
        # Build curve for this grid and select R* with consistent tie-breaking.
        under_list: list[float] = []
        gap_list: list[float] = []
        for R in grid:
            u, _o, g = _compute_under_over_gap(float(R))
            under_list.append(u)
            gap_list.append(g)

        curve_df = pd.DataFrame(
            {
                "R": grid.astype(float),
                "under_cost": np.asarray(under_list, dtype=float),
                "over_cost": np.full(shape=grid.shape, fill_value=over_cost_const, dtype=float),
                "gap": np.asarray(gap_list, dtype=float),
            }
        )
        return _select_R_from_curve(curve_df, grid)

    # Conservative identifiability thresholds (reporting only; does not change selection)
    rel_gap_threshold = 0.05
    log_instability_threshold = float(np.log(1.25))

    # Degenerate case: perfect forecast (no error anywhere)
    if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
        idx = int(np.argmin(np.abs(positive_R - 1.0)))
        R_star = float(positive_R[idx])

        if not return_curve:
            return R_star

        curve = pd.DataFrame(
            {
                "R": positive_R.astype(float),
                "under_cost": np.zeros_like(positive_R, dtype=float),
                "over_cost": np.zeros_like(positive_R, dtype=float),
                "gap": np.zeros_like(positive_R, dtype=float),
            }
        )

        min_gap = 0.0
        rel_min_gap = 0.0
        R_min = R_star
        R_max = R_star
        grid_instability_log = 0.0
        is_identifiable = True

        diagnostics: dict[str, Any] = {
            "over_cost_const": 0.0,
            "min_gap": 0.0,
            "degenerate_perfect_forecast": True,
            "rel_min_gap": float(rel_min_gap),
            "grid_sensitivity": {
                "base": float(R_star),
                "exclude_pivot": float(R_star),
                "shifted": float(R_star),
            },
            "grid_instability_log": float(grid_instability_log),
            "identifiability_thresholds": {
                "rel_gap_threshold": float(rel_gap_threshold),
                "log_instability_threshold": float(log_instability_threshold),
            },
            "is_identifiable": bool(is_identifiable),
        }

        return CostRatioEstimate(
            R_star=R_star,
            method="cost_balance",
            n=int(y_true_arr.shape[0]),
            grid=positive_R.astype(float),
            selection=str(selection),
            tie_break="first",
            diagnostics=diagnostics,
            rel_min_gap=float(rel_min_gap),
            R_min=float(R_min),
            R_max=float(R_max),
            grid_instability_log=float(grid_instability_log),
            is_identifiable=bool(is_identifiable),
            curve=curve,
        )

    # Build sensitivity curve (diagnostics) once for the base grid
    under_list_base: list[float] = []
    gap_list_base: list[float] = []

    for R in positive_R:
        u, _o, g = _compute_under_over_gap(float(R))
        under_list_base.append(u)
        gap_list_base.append(g)

    curve = pd.DataFrame(
        {
            "R": positive_R.astype(float),
            "under_cost": np.asarray(under_list_base, dtype=float),
            "over_cost": np.full(shape=positive_R.shape, fill_value=over_cost_const, dtype=float),
            "gap": np.asarray(gap_list_base, dtype=float),
        }
    )

    # Select R* using the requested strategy on the base curve/grid
    R_star = _select_R_from_curve(curve, positive_R)

    if return_curve:
        min_gap = float(curve.loc[curve["R"] == R_star, "gap"].iloc[0])

        if over_cost_const > 0.0:
            rel_min_gap = float(min_gap / over_cost_const)
        else:
            rel_min_gap = 0.0 if min_gap == 0.0 else float("inf")

        # -----------------------------------------------------------------
        # Grid sensitivity diagnostics (built-in perturbations)
        # -----------------------------------------------------------------
        # 1) Exclude pivot: remove the candidate closest to 1.0 (if possible).
        if positive_R.size >= 2:
            pivot_idx = int(np.argmin(np.abs(positive_R - 1.0)))
            grid_exclude = np.delete(positive_R, pivot_idx)
            R_exclude = _run_on_grid(grid_exclude)
        else:
            R_exclude = R_star

        # 2) Shifted grid: apply a half-step shift in log space based on the grid's median spacing.
        if positive_R.size >= 2:
            sorted_R = np.sort(positive_R.astype(float))
            log_sorted = np.log(sorted_R)
            deltas = np.diff(log_sorted)
            median_delta = float(np.median(deltas)) if deltas.size > 0 else 0.0
            shift = 0.5 * median_delta
            factor = float(np.exp(shift))
            grid_shifted = (positive_R.astype(float) * factor).astype(float)
            # Ensure strictly positive candidates (should be, but keep it explicit)
            grid_shifted = grid_shifted[grid_shifted > 0]
            R_shifted = R_star if grid_shifted.size == 0 else _run_on_grid(grid_shifted)
        else:
            R_shifted = R_star

        R_min = float(min(R_star, R_exclude, R_shifted))
        R_max = float(max(R_star, R_exclude, R_shifted))

        grid_instability_log = float(np.log(R_max) - np.log(R_min)) if R_min > 0.0 else 0.0

        is_identifiable = bool(
            (grid_instability_log <= log_instability_threshold)
            and (rel_min_gap <= rel_gap_threshold)
        )

        diagnostics = {
            "over_cost_const": float(over_cost_const),
            "min_gap": float(min_gap),
            "degenerate_perfect_forecast": False,
            "rel_min_gap": float(rel_min_gap),
            "grid_sensitivity": {
                "base": float(R_star),
                "exclude_pivot": float(R_exclude),
                "shifted": float(R_shifted),
            },
            "grid_instability_log": float(grid_instability_log),
            "identifiability_thresholds": {
                "rel_gap_threshold": float(rel_gap_threshold),
                "log_instability_threshold": float(log_instability_threshold),
            },
            "is_identifiable": bool(is_identifiable),
        }

        return CostRatioEstimate(
            R_star=float(R_star),
            method="cost_balance",
            n=int(y_true_arr.shape[0]),
            grid=positive_R.astype(float),
            selection=str(selection),
            tie_break="first",
            diagnostics=diagnostics,
            rel_min_gap=float(rel_min_gap),
            R_min=float(R_min),
            R_max=float(R_max),
            grid_instability_log=float(grid_instability_log),
            is_identifiable=bool(is_identifiable),
            curve=curve,
        )

    return float(R_star)


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
    sample_weight_col: str | None = None,
    *,
    return_result: bool = False,
    selection: Literal["curve", "kernel"] = "curve",
) -> pd.DataFrame | EntityCostRatioEstimate:
    r"""
    Estimate an entity-level cost ratio via a cost-balance grid search.

    This function estimates a per-entity underbuild-to-overbuild cost ratio:

    $$
        R_e = \frac{c_{u,e}}{c_o}
    $$

    by searching over a user-provided grid of candidate ratios.

    Parameters
    ----------
    df
        Input DataFrame containing entity identifiers, true values, and predictions.
    entity_col
        Column identifying the entity (e.g., store, restaurant, item).
    y_true_col
        Column containing realized demand.
    y_pred_col
        Column containing forecast demand.
    ratios
        Candidate ratio grid. For backward compatibility, the default return mode
        requires all candidates be positive. In artifact mode (`return_result=True`),
        the grid is filtered to strictly positive candidates and the filtered order
        governs tie-breaking.
    co
        Overbuild cost coefficient (scalar). Underbuild cost coefficient for a given
        candidate is `cu = R * co`.
    sample_weight_col
        Optional column containing non-negative sample weights.
    return_result
        If False (default), returns a backward-compatible DataFrame with one row per entity
        using legacy column names (`R`, `diff`).

        If True, returns an `EntityCostRatioEstimate` artifact with:
        - `table`: one row per entity with standardized columns (R_star, gap, etc.)
        - `curves`: dict mapping entity_id -> curve DataFrame with columns [R, under_cost, over_cost, gap]
        - shared governance metadata (method, grid, selection, tie_break)
    selection
        Strategy used to select each entity's $R^*$ once its curve has been computed:

        - "curve"  : Select via NumPy argmin over the curve's `gap` column.
        - "kernel" : Select via `argmin_over_candidates`, scoring each candidate using curve-derived `gap`.

        Both strategies are deterministic and share the same tie-breaking behavior ("first").

    Returns
    -------
    pd.DataFrame or EntityCostRatioEstimate
        If `return_result=False`, returns a DataFrame with one row per entity (legacy schema).
        If `return_result=True`, returns a persistable artifact with per-entity curves.
    """
    if selection not in ("curve", "kernel"):
        raise ValueError("selection must be one of: 'curve', 'kernel'")

    required = {entity_col, y_true_col, y_pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    ratios_arr = np.asarray(list(ratios), dtype=float)
    if ratios_arr.ndim != 1 or ratios_arr.size == 0:
        raise ValueError("ratios must be a non-empty 1D sequence of floats.")

    # Mirror global: filter to strictly positive candidates (and preserve order)
    positive_R = ratios_arr[ratios_arr > 0]
    if positive_R.size == 0:
        raise ValueError("ratios must contain at least one positive value.")

    if co <= 0:
        raise ValueError("co must be strictly positive.")

    # Conservative identifiability thresholds (reporting only; does not change selection)
    rel_gap_threshold = 0.05
    log_instability_threshold = float(np.log(1.25))

    grouped = df.groupby(entity_col, sort=False)

    # Backward-compatible output collector
    legacy_rows: list[dict] = []

    # Artifact-mode collectors
    table_rows: list[dict[str, Any]] = []
    curves: dict[Any, pd.DataFrame] = {}

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
            raise ValueError(f"For entity {entity_id!r}, y_true and y_pred must be non-negative.")
        if np.any(w < 0):
            raise ValueError(f"For entity {entity_id!r}, sample weights must be non-negative.")

        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)

        w_f = w.astype(float, copy=False)
        co_f = float(co)

        # Precompute entity-specific over_cost constant (independent of R)
        over_cost_const = float(np.sum(w_f * co_f * overbuild))

        def _build_curve(
            grid: np.ndarray,
            *,
            _co_f: float = co_f,
            _w_f: np.ndarray = w_f,
            _shortfall: np.ndarray = shortfall,
            _over_cost_const: float = over_cost_const,
        ) -> pd.DataFrame:
            under_list: list[float] = []
            gap_list: list[float] = []
            for R in grid:
                cu_val = float(R) * _co_f
                under_cost = float(np.sum(_w_f * cu_val * _shortfall))
                gap = abs(under_cost - _over_cost_const)
                under_list.append(under_cost)
                gap_list.append(gap)

            return pd.DataFrame(
                {
                    "R": grid.astype(float),
                    "under_cost": np.asarray(under_list, dtype=float),
                    "over_cost": np.full(
                        shape=grid.shape, fill_value=_over_cost_const, dtype=float
                    ),
                    "gap": np.asarray(gap_list, dtype=float),
                }
            )

        def _select_R_from_curve(curve_df: pd.DataFrame, grid: np.ndarray) -> float:
            if selection == "curve":
                idx = int(np.argmin(curve_df["gap"].to_numpy()))
                return float(curve_df["R"].to_numpy()[idx])

            r_vals = curve_df["R"].to_numpy()
            gap_vals = curve_df["gap"].to_numpy()
            idx_map = {float(r): i for i, r in enumerate(r_vals.tolist())}

            def _gap_for_R(R: float, *, _gap_vals=gap_vals, _idx_map=idx_map) -> float:
                return float(_gap_vals[_idx_map[float(R)]])

            best_R, _best_gap = argmin_over_candidates(
                candidates=grid,
                score_fn=_gap_for_R,
                tie_break="first",
            )
            return float(best_R)

        # Degenerate case: no error at all for this entity
        if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
            idx = int(np.argmin(np.abs(positive_R - 1.0)))
            R_star = float(positive_R[idx])
            cu_star = R_star * co_f

            # Legacy row (keep legacy names in legacy mode)
            legacy_rows.append(
                {
                    entity_col: entity_id,
                    "R": R_star,
                    "cu": float(cu_star),
                    "co": float(co_f),
                    "under_cost": 0.0,
                    "over_cost": 0.0,
                    "diff": 0.0,
                }
            )

            if return_result:
                curve = pd.DataFrame(
                    {
                        "R": positive_R.astype(float),
                        "under_cost": np.zeros_like(positive_R, dtype=float),
                        "over_cost": np.zeros_like(positive_R, dtype=float),
                        "gap": np.zeros_like(positive_R, dtype=float),
                    }
                )
                curves[entity_id] = curve

                min_gap = 0.0
                rel_min_gap = 0.0
                R_min = R_star
                R_max = R_star
                grid_instability_log = 0.0
                is_identifiable = True

                diagnostics: dict[str, Any] = {
                    "over_cost_const": 0.0,
                    "min_gap": 0.0,
                    "degenerate_perfect_forecast": True,
                    "rel_min_gap": float(rel_min_gap),
                    "grid_sensitivity": {
                        "base": float(R_star),
                        "exclude_pivot": float(R_star),
                        "shifted": float(R_star),
                    },
                    "grid_instability_log": float(grid_instability_log),
                    "identifiability_thresholds": {
                        "rel_gap_threshold": float(rel_gap_threshold),
                        "log_instability_threshold": float(log_instability_threshold),
                    },
                    "is_identifiable": bool(is_identifiable),
                }

                table_rows.append(
                    {
                        entity_col: entity_id,
                        "R_star": float(R_star),
                        "n": int(y_true.shape[0]),
                        "under_cost": 0.0,
                        "over_cost": 0.0,
                        "gap": 0.0,
                        "diagnostics": diagnostics,
                    }
                )
            continue

        # Base curve on the canonical grid (positive_R preserves order/tie-break)
        curve = _build_curve(positive_R)
        R_star = _select_R_from_curve(curve, positive_R)

        min_gap = float(curve.loc[curve["R"] == R_star, "gap"].iloc[0])
        under_star = float(curve.loc[curve["R"] == R_star, "under_cost"].iloc[0])
        over_star = float(over_cost_const)

        # Legacy row (keep existing schema)
        legacy_rows.append(
            {
                entity_col: entity_id,
                "R": float(R_star),
                "cu": float(R_star * co_f),
                "co": float(co_f),
                "under_cost": float(under_star),
                "over_cost": float(over_star),
                "diff": float(min_gap),
            }
        )

        if return_result:
            curves[entity_id] = curve

            if over_cost_const > 0.0:
                rel_min_gap = float(min_gap / over_cost_const)
            else:
                rel_min_gap = 0.0 if min_gap == 0.0 else float("inf")

            # -----------------------------------------------------------------
            # Entity-level grid sensitivity diagnostics (parity with global)
            # -----------------------------------------------------------------
            # 1) Exclude pivot: remove the candidate closest to 1.0 (if possible).
            if positive_R.size >= 2:
                pivot_idx = int(np.argmin(np.abs(positive_R - 1.0)))
                grid_exclude = np.delete(positive_R, pivot_idx)
                curve_exclude = _build_curve(grid_exclude)
                R_exclude = _select_R_from_curve(curve_exclude, grid_exclude)
            else:
                R_exclude = R_star

            # 2) Shifted grid: apply a half-step shift in log space based on the grid's median spacing.
            if positive_R.size >= 2:
                sorted_R = np.sort(positive_R.astype(float))
                log_sorted = np.log(sorted_R)
                deltas = np.diff(log_sorted)
                median_delta = float(np.median(deltas)) if deltas.size > 0 else 0.0
                shift = 0.5 * median_delta
                factor = float(np.exp(shift))
                grid_shifted = (positive_R.astype(float) * factor).astype(float)
                grid_shifted = grid_shifted[grid_shifted > 0]
                if grid_shifted.size == 0:
                    R_shifted = R_star
                else:
                    curve_shifted = _build_curve(grid_shifted)
                    R_shifted = _select_R_from_curve(curve_shifted, grid_shifted)
            else:
                R_shifted = R_star

            R_min = float(min(R_star, R_exclude, R_shifted))
            R_max = float(max(R_star, R_exclude, R_shifted))
            grid_instability_log = float(np.log(R_max) - np.log(R_min)) if R_min > 0.0 else 0.0

            is_identifiable = bool(
                (grid_instability_log <= log_instability_threshold)
                and (rel_min_gap <= rel_gap_threshold)
            )

            # Ensure diagnostics are JSON-serializable simple types
            diagnostics = {
                "over_cost_const": float(over_cost_const),
                "min_gap": float(min_gap),
                "degenerate_perfect_forecast": False,
                "rel_min_gap": float(rel_min_gap),
                "grid_sensitivity": {
                    "base": float(R_star),
                    "exclude_pivot": float(R_exclude),
                    "shifted": float(R_shifted),
                },
                "grid_instability_log": float(grid_instability_log),
                "identifiability_thresholds": {
                    "rel_gap_threshold": float(rel_gap_threshold),
                    "log_instability_threshold": float(log_instability_threshold),
                },
                "is_identifiable": bool(is_identifiable),
            }

            table_rows.append(
                {
                    entity_col: entity_id,
                    "R_star": float(R_star),
                    "n": int(y_true.shape[0]),
                    "under_cost": float(under_star),
                    "over_cost": float(over_star),
                    "gap": float(min_gap),
                    "diagnostics": diagnostics,
                }
            )

    legacy_df = pd.DataFrame(legacy_rows)

    if not return_result:
        return legacy_df

    table = pd.DataFrame(table_rows)
    return EntityCostRatioEstimate(
        entity_col=str(entity_col),
        method="cost_balance",
        grid=positive_R.astype(float),
        selection=str(selection),
        tie_break="first",
        table=table,
        curves=curves,
    )
