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
from typing import Any, Literal, cast
import warnings

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from eb_optimization.tuning.cost_ratio import (
    CostRatioEstimate,
    EntityCostRatioEstimate,
    estimate_entity_R_from_balance,
    estimate_R_cost_balance,
)

GateMode = Literal["off", "warn", "raise"]


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


def _handle_identifiability_gate(
    *,
    gate: GateMode,
    ok: bool,
    message: str,
    override_reason: str | None,
) -> dict[str, Any]:
    """
    Warn-only / gateable hook shared by global + entity policy applications.

    Returns a JSON-serializable dict with the gating decision metadata that can be
    merged into diagnostics payloads.
    """
    gate = cast(GateMode, gate)
    if gate not in ("off", "warn", "raise"):
        raise ValueError("gate must be one of: 'off', 'warn', 'raise'")

    meta: dict[str, Any] = {
        "gate_mode": gate,
        "gate_triggered": bool(not ok),
        "gate_overridden": bool((not ok) and (override_reason is not None)),
        "gate_override_reason": override_reason,
    }

    if ok or gate == "off":
        return meta

    # If not ok:
    if override_reason is not None:
        # Override: do not warn/raise; record reason.
        return meta

    if gate == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return meta

    # gate == "raise"
    raise ValueError(message)


def apply_cost_ratio_policy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    policy: CostRatioPolicy = DEFAULT_COST_RATIO_POLICY,
    co: float | ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    gate: GateMode = "warn",
    identifiability_override_reason: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Apply a frozen cost-ratio policy to estimate a global R.

    Notes
    -----
    This policy boundary surfaces identifiability / stability diagnostics when available.
    It does NOT change the selection behavior; it only enriches the returned diagnostics.

    Gating
    ------
    `gate` controls what happens when tuning reports `is_identifiable=False`:

    - gate="off"  : no action (still reports diagnostics)
    - gate="warn" : emit a RuntimeWarning (default)
    - gate="raise": raise ValueError

    Overrides
    ---------
    If `identifiability_override_reason` is provided, the gate will not warn/raise,
    and the reason is recorded in diagnostics for auditability.
    """
    co_val = policy.co if co is None else co

    # Request the richer artifact so we can surface identifiability at the policy boundary.
    est_any: Any = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=policy.R_grid,
        co=co_val,
        sample_weight=sample_weight,
        return_curve=True,
        selection="curve",
    )

    stability_diag: dict[str, Any]
    gate_meta: dict[str, Any] = {}

    if isinstance(est_any, CostRatioEstimate):
        est = est_any
        R = float(est.R_star)

        ok = bool(est.is_identifiable)
        msg = (
            "Cost ratio calibration is not identifiable/stable under configured diagnostics. "
            f"(rel_min_gap={float(est.rel_min_gap):.6g}, "
            f"grid_instability_log={float(est.grid_instability_log):.6g}, "
            f"R_star={float(est.R_star):.6g}, R_range=[{float(est.R_min):.6g}, {float(est.R_max):.6g}])."
        )

        gate_meta = _handle_identifiability_gate(
            gate=gate,
            ok=ok,
            message=msg,
            override_reason=identifiability_override_reason,
        )

        stability_diag = {
            # Policy-surfaced stability fields (governance/reporting only)
            "rel_min_gap": float(est.rel_min_gap),
            "R_min": float(est.R_min),
            "R_max": float(est.R_max),
            "grid_instability_log": float(est.grid_instability_log),
            "is_identifiable": bool(est.is_identifiable),
            # Full tuning diagnostics
            "calibration_diagnostics": dict(est.diagnostics),
        }
    else:
        R = float(est_any)
        stability_diag = {
            "calibration_diagnostics": {
                "note": "Upstream returned a scalar R; stability diagnostics unavailable.",
            }
        }

    diag: dict[str, Any] = {
        "method": "cost_balance",
        "R_grid": [float(x) for x in policy.R_grid],
        "co_is_array": isinstance(co_val, list | tuple | np.ndarray | pd.Series),
        "co_default_used": co is None,
        "R": float(R),
        **stability_diag,
        **({"identifiability_gate": gate_meta} if gate_meta else {}),
    }
    return (float(R), diag)


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
    gate: GateMode = "warn",
    identifiability_override_reason: str | None = None,
) -> pd.DataFrame:
    """
    Apply a frozen cost-ratio policy per entity.

    Notes
    -----
    This policy boundary surfaces per-entity calibration diagnostics (in the
    `diagnostics` column) for eligible entities when `include_diagnostics=True`.

    Entity-level identifiability
    ----------------------------
    The tuning artifact returns per-entity `diagnostics` dicts. If those dicts contain
    an `is_identifiable` field, this function will:
      - surface a convenience `is_identifiable` column, and
      - optionally warn/raise based on `gate`.

    If no such field exists (older tuning versions), gating is a no-op.
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
    counts_ser = cast(pd.Series, df.groupby(entity_col, dropna=False, sort=False).size())
    eligible_counts = cast(pd.Series, counts_ser[counts_ser >= policy.min_n])
    eligible_list = cast(list[Any], eligible_counts.index.tolist())

    mask = df[entity_col].isin(eligible_list)

    eligible_df = cast(pd.DataFrame, df[mask]).copy()
    ineligible_df = cast(pd.DataFrame, df[~mask]).copy()

    results_list: list[pd.DataFrame] = []

    gate_meta: dict[str, Any] = {}
    failed_entities: list[Any] = []

    # Track whether we can gate at all (only if the diagnostics dicts contain is_identifiable)
    identifiability_available = False

    if not eligible_df.empty:
        tuned_any: Any = estimate_entity_R_from_balance(
            df=eligible_df,
            entity_col=entity_col,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            ratios=policy.R_grid,
            co=co_val,
            sample_weight_col=sample_weight_col,
            return_result=True,
            selection="curve",
        )
        tuned_art = cast(EntityCostRatioEstimate, tuned_any)

        tuned_table = cast(pd.DataFrame, tuned_art.table).copy()

        tuned = pd.DataFrame(
            {
                entity_col: tuned_table[entity_col],
                "R": tuned_table["R_star"].astype(float),
                "cu": (tuned_table["R_star"].astype(float) * float(co_val)).astype(float),
                "co": float(co_val),
                "under_cost": tuned_table["under_cost"].astype(float),
                "over_cost": tuned_table["over_cost"].astype(float),
                "diff": tuned_table["gap"].astype(float),
            }
        )

        # Always attach diagnostics internally (for gating), drop later if include_diagnostics=False
        if "diagnostics" in tuned_table.columns:
            tuned["diagnostics"] = tuned_table["diagnostics"]
        else:
            tuned["diagnostics"] = None

        tuned["reason"] = None

        mapper: Any = counts_ser
        tuned["n"] = tuned[entity_col].map(mapper).astype(int)

        # Surface an `is_identifiable` column if diagnostics provides it.
        # This is a convenience field for users/tests and lets gating be explicit/inspectable.
        def _extract_is_identifiable(v: Any) -> Any:
            if isinstance(v, dict) and ("is_identifiable" in v):
                return bool(v.get("is_identifiable"))
            return None

        tuned["is_identifiable"] = tuned["diagnostics"].map(_extract_is_identifiable)

        # ---- FIX: make the conditional unambiguously bool for Pyright ----
        has_ident = bool(tuned["is_identifiable"].notna().to_numpy().any())
        if has_ident:
            identifiability_available = True
            failed_mask = tuned["is_identifiable"] == False  # noqa: E712
            failed_entities = cast(list[Any], tuned.loc[failed_mask, entity_col].tolist())

        if identifiability_available and failed_entities:
            msg = (
                "One or more entities have non-identifiable cost ratio calibration. "
                f"Failed entities (first 10): {failed_entities[:10]!r}. "
                "You may override with identifiability_override_reason."
            )
            gate_meta = _handle_identifiability_gate(
                gate=gate,
                ok=False,
                message=msg,
                override_reason=identifiability_override_reason,
            )

        results_list.append(tuned)

    if not ineligible_df.empty:
        ineligible_rows = cast(pd.DataFrame, ineligible_df[[entity_col]]).drop_duplicates()
        ineligible_rows = ineligible_rows.assign(
            R=np.nan,
            cu=np.nan,
            co=co_val,
            under_cost=np.nan,
            over_cost=np.nan,
            diff=np.nan,
            reason=f"min_n_not_met(<{policy.min_n})",
            diagnostics=None,
            is_identifiable=None,
        )

        mapper_ineligible: Any = counts_ser
        ineligible_rows["n"] = ineligible_rows[entity_col].map(mapper_ineligible).astype(int)
        results_list.append(ineligible_rows)

    if not results_list:
        schema_cols = [
            entity_col,
            "R",
            "cu",
            "co",
            "n",
            "reason",
            "under_cost",
            "over_cost",
            "diff",
        ]
        if include_diagnostics:
            schema_cols.append("diagnostics")
        # Include is_identifiable if we can ever surface it (safe default: include it anyway)
        schema_cols.append("is_identifiable")
        if gate_meta:
            schema_cols.append("identifiability_gate")

        return pd.DataFrame(columns=pd.Index(schema_cols))

    out = pd.concat(results_list, ignore_index=True, sort=False)

    # Attach gate metadata as a repeated column (JSON-serializable) if it exists
    if gate_meta:
        out["identifiability_gate"] = [gate_meta] * int(out.shape[0])

    base_cols = [entity_col, "R", "cu", "co", "n", "reason", "is_identifiable"]
    diag_cols = ["under_cost", "over_cost", "diff"]

    if include_diagnostics:
        base_cols.append("diagnostics")

    if not include_diagnostics and "diagnostics" in out.columns:
        out = out.drop(columns=["diagnostics"])

    if "identifiability_gate" in out.columns:
        base_cols.append("identifiability_gate")

    remaining = [str(c) for c in out.columns if c not in base_cols + diag_cols]
    target_cols = (
        (base_cols + diag_cols + remaining) if include_diagnostics else (base_cols + remaining)
    )

    return cast(pd.DataFrame, out[pd.Index(target_cols)])


__all__ = [
    "DEFAULT_COST_RATIO_POLICY",
    "CostRatioPolicy",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",
]
