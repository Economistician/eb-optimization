"""
Tau (τ) policy artifacts for eb-optimization.

This module defines *frozen governance* for selecting a tolerance τ used by HR@τ.

- tuning/tau.py: calibration logic (estimating τ from residuals)
- policies/tau_policy.py: frozen configuration + deterministic application wrappers

Policies should be stable, auditable, and safe to apply at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eb_optimization.tuning.tau import (
    TauMethod,
    estimate_entity_tau,
    estimate_tau,
    hr_at_tau,
)


@dataclass(frozen=True)
class TauPolicy:
    """
    Frozen τ policy configuration.

    This is the governance object you can persist, version, and ship to downstream
    consumers.

    Notes
    -----
    - `estimate_kwargs` are passed through to `estimate_tau`.
    - If `cap_with_global` is True, entity τ values are capped by a global cap
      derived from the full residual distribution at `global_cap_quantile`.
    """

    method: TauMethod = "target_hit_rate"
    min_n: int = 30

    # Passed to estimate_tau(...)
    estimate_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # Governance
    cap_with_global: bool = False
    global_cap_quantile: float = 0.99

    def __post_init__(self) -> None:
        if self.min_n < 1:
            raise ValueError(f"min_n must be >= 1. Got {self.min_n}.")
        if not (0.0 < self.global_cap_quantile <= 1.0):
            raise ValueError(
                f"global_cap_quantile must be in (0, 1]. Got {self.global_cap_quantile}."
            )


# ----------------------------------------------------------------------
# Default, exported policy
# ----------------------------------------------------------------------
DEFAULT_TAU_POLICY = TauPolicy(
    method="target_hit_rate",
    min_n=30,
    estimate_kwargs={
        "target_hit_rate": 0.90,
        "tau_floor": 0.0,
        "tau_cap": None,
    },
    cap_with_global=False,
    global_cap_quantile=0.99,
)


def apply_tau_policy(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
    policy: TauPolicy = DEFAULT_TAU_POLICY,
) -> tuple[float, dict[str, Any]]:
    """
    Apply a frozen τ policy to produce τ (global).

    Returns
    -------
    (tau, diagnostics)
    """
    est = estimate_tau(
        y=y,
        yhat=yhat,
        method=policy.method,
        **dict(policy.estimate_kwargs),
    )
    return (float(est.tau), dict(est.diagnostics or {}))


def apply_tau_policy_hr(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
    policy: TauPolicy = DEFAULT_TAU_POLICY,
) -> tuple[float, float, dict[str, Any]]:
    """
    Apply τ policy, then compute HR@τ.

    Returns
    -------
    (hr, tau, diagnostics)
    """
    tau, diag = apply_tau_policy(y=y, yhat=yhat, policy=policy)
    if not np.isfinite(tau):
        return (np.nan, np.nan, diag)
    hr = hr_at_tau(y=y, yhat=yhat, tau=tau)
    return (float(hr), float(tau), diag)


def apply_entity_tau_policy(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_col: str,
    yhat_col: str,
    policy: TauPolicy = DEFAULT_TAU_POLICY,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Apply a frozen τ policy per entity (with optional global cap governance).

    This wraps tuning.estimate_entity_tau but pins governance via TauPolicy.
    """
    return estimate_entity_tau(
        df=df,
        entity_col=entity_col,
        y_col=y_col,
        yhat_col=yhat_col,
        method=policy.method,
        min_n=policy.min_n,
        estimate_kwargs=policy.estimate_kwargs,
        cap_with_global=policy.cap_with_global,
        global_cap_quantile=policy.global_cap_quantile,
        include_diagnostics=include_diagnostics,
    )


__all__ = [
    "DEFAULT_TAU_POLICY",
    "TauPolicy",
    "apply_entity_tau_policy",
    "apply_tau_policy",
    "apply_tau_policy_hr",
]
