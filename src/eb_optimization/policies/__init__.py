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

from __future__ import annotations

from .cost_ratio_policy import (
    DEFAULT_COST_RATIO_POLICY,
    CostRatioPolicy,
    apply_cost_ratio_policy,
    apply_entity_cost_ratio_policy,
)
from .ral_policy import DEFAULT_RAL_POLICY, RALPolicy, apply_ral_policy
from .tau_policy import (
    TauPolicy,
    apply_entity_tau_policy,
    apply_tau_policy,
    apply_tau_policy_hr,
)

__all__ = [
    "DEFAULT_COST_RATIO_POLICY",
    "DEFAULT_RAL_POLICY",
    "CostRatioPolicy",
    "RALPolicy",
    "TauPolicy",
    "apply_cost_ratio_policy",
    "apply_entity_cost_ratio_policy",
    "apply_entity_tau_policy",
    "apply_ral_policy",
    "apply_tau_policy",
    "apply_tau_policy_hr",
]
