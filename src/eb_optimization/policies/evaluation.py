"""Policy-composed evaluation entrypoints for Electric Barometer.

This module provides "blessed" evaluation functions that compose multiple policy
artifacts into a single, decision-safe call. The intent is to make correct
evaluation easy to use and hard to misuse.

Primary responsibilities:
- Provide a stable API for evaluation that downstream code can depend on.
- Enforce structural compatibility rules (e.g., DQC snapping for packed demand).
- Interpret tolerance parameters in the correct unit space (e.g., grid units).
- Delegate metric primitives to eb-metrics after applying policy semantics.

This module is intentionally small. It does not implement forecasting models,
data adapters, or tuning routines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from eb_optimization.policies.dqc_policy import (
    DEFAULT_DQC_POLICY,
    DQCPolicy,
    DQCResult,
    EnforcementMode,
    SnapMode,
    compute_dqc,
    enforce_snapping,
)

try:
    from eb_metrics.metrics.service import hr_at_tau as _hr_at_tau
except Exception:  # pragma: no cover
    _hr_at_tau = None


@dataclass(frozen=True, slots=True)
class DQCEvaluation:
    """Result container for evaluations performed under DQC governance."""

    dqc: DQCResult
    tau_units: float
    tau_y_units: float
    hr_at_tau: float


def evaluate_with_dqc_hr(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    *,
    tau_units: float,
    dqc: DQCResult | None = None,
    y_for_dqc: Sequence[float] | np.ndarray | None = None,
    policy: DQCPolicy = DEFAULT_DQC_POLICY,
    enforce: EnforcementMode = "snap",
    snap_mode: SnapMode = "nearest",
    use_positive_only_for_dqc: bool = True,
) -> DQCEvaluation:
    """Evaluate HR@τ under DQC governance with τ expressed in grid units.

    This entrypoint enforces Demand Quantization Compatibility (DQC) rules:
    - If demand is PACKED or QUANTIZED (per DQCResult), forecasts are snapped
      to the governing grid size Δ* (subject to enforcement mode).
    - Tolerance τ is interpreted in grid units. The numeric τ used for the
      metric primitive is τ_y_units = tau_units * Δ*.

    Callers may supply a precomputed DQCResult (recommended in production), or
    allow this function to compute DQC from realized demand via `y_for_dqc` (or
    `y_true` if `y_for_dqc` is not provided).

    Args:
        y_true: Realized demand values.
        y_hat: Forecast demand values.
        tau_units: Tolerance in grid units (multiples of Δ*).
        dqc: Optional precomputed DQCResult. If provided, DQC is not recomputed.
        y_for_dqc: Optional series to compute DQC from (e.g., entity-level
            historical realized demand). Ignored if `dqc` is provided.
        policy: DQCPolicy thresholds and candidate grids used if computing DQC.
        enforce: Snapping enforcement behavior ("snap", "raise", "ignore").
        snap_mode: Snapping mode applied when enforce == "snap".
        use_positive_only_for_dqc: If True, DQC detection uses y>0 values.

    Returns:
        DQCEvaluation including the DQCResult, τ in grid units, τ in y-units,
        and HR@τ.

    Raises:
        ImportError: If eb-metrics is not available.
        ValueError: If enforcement mode is "raise" and forecasts are off-grid.
    """
    if _hr_at_tau is None:  # pragma: no cover
        raise ImportError("eb-metrics is required (missing eb_metrics.metrics.service.hr_at_tau).")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)

    if dqc is None:
        y_src = y_true_arr if y_for_dqc is None else np.asarray(y_for_dqc, dtype=float)
        dqc = compute_dqc(y_src, policy=policy, use_positive_only=use_positive_only_for_dqc)

    # Default: treat tau_units as y-units unless DQC indicates a grid.
    tau_y_units = float(tau_units)

    if dqc.dqc_class in ("PACKED", "QUANTIZED") and dqc.delta_star is not None:
        y_hat_arr = enforce_snapping(y_hat_arr, dqc=dqc, enforce=enforce, mode=snap_mode)
        tau_y_units = float(tau_units) * float(dqc.delta_star)

    hr = float(_hr_at_tau(y_true_arr, y_hat_arr, tau=tau_y_units))

    return DQCEvaluation(
        dqc=dqc,
        tau_units=float(tau_units),
        tau_y_units=tau_y_units,
        hr_at_tau=hr,
    )
