from __future__ import annotations

"""
Frozen policy artifacts for the Electric Barometer optimization layer.

The `eb_optimization.policies` package contains **governance-level, immutable
configuration objects** that define how tuned parameters are selected and applied
at runtime.

Design principles
-----------------
- Policies are **frozen** (dataclass(frozen=True)) and versionable
- Policies contain **no learning or tuning logic**
- Policies wrap tuning utilities with deterministic application semantics
- Policies are safe to ship to production systems

Layering
--------
- tuning/    : derives parameters from data (calibration, grid search)
- policies/  : freezes configuration + applies tuning deterministically
- runtime    : consumes policy outputs only (no re-tuning)

Exported policies
-----------------
- Tau (τ) tolerance governance for HR@τ
- Cost-ratio (R = c_u / c_o) governance for asymmetric loss
- RAL policy governance (readiness adjustment layer)
"""

# ---------------------------------------------------------------------
# Tau (tolerance) policies
# ---------------------------------------------------------------------
from .tau_policy import (
    TauPolicy,
    apply_tau_policy,
    apply_tau_policy_hr,
    apply_entity_tau_policy,
)

# ---------------------------------------------------------------------
# Cost-ratio (R) policies
# ---------------------------------------------------------------------
from .cost_ratio_policy import (
    CostRatioPolicy,
    DEFAULT_COST_RATIO_POLICY,
    apply_cost_ratio_policy,
    apply_entity_cost_ratio_policy,
)

# ---------------------------------------------------------------------
# RAL policies
# ---------------------------------------------------------------------
from .ral_policy import (
    RALPolicy,
    DEFAULT_RAL_POLICY,
    apply_ral_policy,
)

__all__ = [
    # Tau policies
    "TauPolicy",
    "apply_tau_policy",
    "apply_tau_policy_hr",
    "apply_entity_tau_policy",

    # Cost ratio policies
    "CostRatioPolicy",
    "DEFAULT_COST_RATIO_POLICY",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",

    # RAL policies
    "RALPolicy",
    "DEFAULT_RAL_POLICY",
    "apply_ral_policy",
]