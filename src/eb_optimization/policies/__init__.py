"""Frozen runtime policies (τ, cost ratio, RAL, DQC enforcement).

Immutable configuration objects with no learning logic. Prefer
``eb_evaluation.diagnostics.validate_dqc`` then ``enforce_snapping`` /
``evaluate_with_dqc_hr``; ``compute_dqc`` remains for compatibility.
"""

from __future__ import annotations

from .cost_ratio_policy import (
    DEFAULT_COST_RATIO_POLICY,
    CostRatioPolicy,
    apply_cost_ratio_policy,
    apply_entity_cost_ratio_policy,
)
from .dqc_policy import (
    DEFAULT_DQC_POLICY,
    DQCPolicy,
    DQCResult,
    compute_dqc,
    enforce_snapping,
    hr_at_tau_grid_units,
    snap_to_grid,
)
from .evaluation import DQCEvaluation, evaluate_with_dqc_hr
from .ral_policy import (
    DEFAULT_RAL_POLICY,
    RALBands,
    RALBandThresholds,
    RALPolicy,
    RALThresholdTwoBandPolicy,
    apply_ral_policy,
)
from .tau_policy import (
    DEFAULT_TAU_POLICY,
    TauPolicy,
    apply_entity_tau_policy,
    apply_tau_policy,
    apply_tau_policy_hr,
)

__all__ = [
    "DEFAULT_COST_RATIO_POLICY",
    "DEFAULT_DQC_POLICY",
    "DEFAULT_RAL_POLICY",
    "DEFAULT_TAU_POLICY",
    "CostRatioPolicy",
    "DQCEvaluation",
    "DQCPolicy",
    "DQCResult",
    "RALBandThresholds",
    "RALBands",
    "RALPolicy",
    "RALThresholdTwoBandPolicy",
    "TauPolicy",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",
    "apply_entity_tau_policy",
    "apply_ral_policy",
    "apply_tau_policy",
    "apply_tau_policy_hr",
    "compute_dqc",
    "enforce_snapping",
    "evaluate_with_dqc_hr",
    "hr_at_tau_grid_units",
    "snap_to_grid",
]
