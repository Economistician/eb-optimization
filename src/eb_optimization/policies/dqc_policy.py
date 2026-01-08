"""Demand Quantization Compatibility (DQC) policy and snapping enforcement.

This module defines the governance logic required to correctly evaluate forecasts
when realized demand is quantized or unit-packed. It implements Demand Quantization
Compatibility (DQC) as a diagnostic and enforcement layer that detects whether demand
lies on a discrete grid and, if so, enforces projection ("snapping") of forecasts
onto that grid prior to evaluation.

Key responsibilities:
- Detect demand quantization structure and infer the governing grid size (Δ*).
- Classify demand as CONTINUOUS, QUANTIZED, or PACKED based on alignment evidence.
- Enforce unit compatibility by snapping forecasts to Δ* when required.
- Interpret evaluation tolerances (τ) in grid units rather than raw numeric units.

This module does not define forecasting models or metric primitives. Instead, it
provides policy-level wrappers that ensure evaluation metrics operate in a valid
unit space. When demand is PACKED or QUANTIZED, unsnapped evaluation is considered
invalid and must be corrected or rejected according to policy.

DQC is a structural compatibility gate, not a performance metric. Its purpose is to
prevent mathematically invalid evaluation and model comparison when demand outcomes
are intrinsically discrete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

try:
    # eb-optimization does not define metric primitives; it delegates to eb-metrics.
    from eb_metrics.metrics.service import hr_at_tau as _hr_at_tau
except Exception:  # pragma: no cover
    _hr_at_tau = None


DQCClass = Literal["CONTINUOUS", "QUANTIZED", "PACKED"]
SnapMode = Literal["nearest", "floor", "ceil"]
EnforcementMode = Literal["snap", "raise", "ignore"]


@dataclass(frozen=True, slots=True)
class DQCPolicy:
    """Governance thresholds and candidate grids for DQC."""

    # Candidate grids (Δ) to test for alignment. Keep small and auditable.
    candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)

    # Numerical tolerance for "on-grid" checks in y-units.
    tol: float = 1e-6

    # Classification thresholds.
    rho_packed: float = 0.99
    rho_quantized: float = 0.90

    # Packed-ness evidence thresholds.
    support_packed_max: int = 100
    offgrid_mad_over_delta_max: float = 0.05

    # Minimum positive support required to consider the DQC result meaningful.
    min_n_pos: int = 50

    # Tie-break rule for selecting delta_star among candidates with equal rho.
    # "max_delta" prefers the coarsest grid among ties (more conservative packing).
    tie_break: Literal["max_delta", "min_delta"] = "max_delta"


DEFAULT_DQC_POLICY = DQCPolicy()


@dataclass(frozen=True, slots=True)
class DQCResult:
    """Output of DQC computation."""

    dqc_class: DQCClass
    delta_star: float | None
    rho_star: float | None

    # Evidence / diagnostics
    n_pos: int
    support_size: int
    offgrid_mad_over_delta: float | None


def snap_to_grid(
    x: np.ndarray,
    delta: float,
    *,
    mode: SnapMode = "nearest",
    nonneg: bool = True,
) -> np.ndarray:
    """Project values onto multiples of delta.

    Args:
        x: Array of values to snap (may include NaNs).
        delta: Grid size (Δ). Must be > 0.
        mode: Nearest, floor, or ceil snapping.
        nonneg: If True, clamps to >= 0 after snapping.

    Returns:
        Snapped array (float dtype), preserving NaNs.
    """
    if not (isinstance(delta, int | float) and delta > 0.0):
        raise ValueError(f"delta must be > 0; got {delta!r}")

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    m = np.isfinite(x)
    if not np.any(m):
        return out

    q = x[m] / delta

    if mode == "nearest":
        snapped = np.round(q) * delta
    elif mode == "floor":
        snapped = np.floor(q) * delta
    elif mode == "ceil":
        snapped = np.ceil(q) * delta
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if nonneg:
        snapped = np.maximum(snapped, 0.0)

    out[m] = snapped
    return out


def _mad(a: np.ndarray) -> float:
    """Median absolute deviation around the median (robust)."""
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def compute_dqc(
    y: Sequence[float] | np.ndarray,
    *,
    policy: DQCPolicy = DEFAULT_DQC_POLICY,
    use_positive_only: bool = True,
) -> DQCResult:
    """Compute DQC over a realized demand series.

    Notes:
    - For grid detection, positive demand values carry the most information.
      Zeros can dominate alignment trivially, so default behavior uses positives only.
    - Missing values (NaN) are ignored.

    Args:
        y: Realized demand sequence.
        policy: DQCPolicy thresholds and candidate grids.
        use_positive_only: If True, only y>0 are used for detection.

    Returns:
        DQCResult with class + delta_star + diagnostics.
    """
    y_arr = np.asarray(y, dtype=float)
    y_arr = y_arr[np.isfinite(y_arr)]

    if use_positive_only:
        y_arr = y_arr[y_arr > 0.0]

    n_pos = int(y_arr.size)
    if n_pos < policy.min_n_pos:
        # Not enough signal to conclude; treat as continuous-like by default.
        return DQCResult(
            dqc_class="CONTINUOUS",
            delta_star=None,
            rho_star=None,
            n_pos=n_pos,
            support_size=int(np.unique(np.round(y_arr, 4)).size) if n_pos else 0,
            offgrid_mad_over_delta=None,
        )

    # Support size (rounded to label precision)
    support_size = int(np.unique(np.round(y_arr, 4)).size)

    candidates = np.asarray(policy.candidates, dtype=float)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("policy.candidates must be a non-empty 1D sequence")

    if np.any(candidates <= 0.0):
        raise ValueError("All candidate deltas must be > 0")

    # Residual to nearest multiple: |y - round(y/Δ)*Δ|
    y_col = y_arr.reshape(-1, 1)  # (n,1)
    deltas = candidates.reshape(1, -1)  # (1,k)

    nearest = np.round(y_col / deltas) * deltas
    resid = np.abs(y_col - nearest)  # (n,k)

    rho = (resid <= policy.tol).mean(axis=0)  # (k,)

    # Choose delta_star: maximize rho, break ties by delta direction.
    max_rho = float(np.max(rho))
    tie_idx = np.flatnonzero(rho == max_rho)

    if tie_idx.size == 1:
        best_idx = int(tie_idx[0])
    else:
        if policy.tie_break == "max_delta":
            best_idx = int(tie_idx[np.argmax(candidates[tie_idx])])
        elif policy.tie_break == "min_delta":
            best_idx = int(tie_idx[np.argmin(candidates[tie_idx])])
        else:
            raise ValueError(f"Unsupported tie_break: {policy.tie_break!r}")

    delta_star = float(candidates[best_idx])
    rho_star = float(rho[best_idx])

    resid_star = resid[:, best_idx]
    mad = _mad(resid_star)
    offgrid_mad_over_delta = float(mad / delta_star) if delta_star > 0 else math.nan

    # Classification
    if (
        rho_star >= policy.rho_packed
        and offgrid_mad_over_delta <= policy.offgrid_mad_over_delta_max
        and support_size <= policy.support_packed_max
    ):
        dqc_class: DQCClass = "PACKED"
    elif rho_star >= policy.rho_quantized:
        dqc_class = "QUANTIZED"
    else:
        dqc_class = "CONTINUOUS"

    return DQCResult(
        dqc_class=dqc_class,
        delta_star=delta_star,
        rho_star=rho_star,
        n_pos=n_pos,
        support_size=support_size,
        offgrid_mad_over_delta=offgrid_mad_over_delta,
    )


def enforce_snapping(
    y_hat: Sequence[float] | np.ndarray,
    *,
    dqc: DQCResult,
    enforce: EnforcementMode = "snap",
    mode: SnapMode = "nearest",
) -> np.ndarray:
    """Apply DQC snapping enforcement to forecasts.

    Policy intent:
    - PACKED demand => snapping is required (unit compatibility).
    - QUANTIZED demand => snapping is strongly recommended; default behavior snaps.
    - CONTINUOUS => no snapping.

    Args:
        y_hat: Forecast values.
        dqc: DQCResult for the relevant entity/window.
        enforce: "snap" (default), "raise" (error if off-grid), "ignore".
        mode: Snapping mode (if enforce == "snap").

    Returns:
        Forecast array, snapped or unchanged depending on class and enforcement.
    """
    y_hat_arr = np.asarray(y_hat, dtype=float)

    if dqc.dqc_class == "CONTINUOUS" or dqc.delta_star is None:
        return y_hat_arr

    if enforce == "ignore":
        return y_hat_arr

    delta = float(dqc.delta_star)

    if enforce == "raise":
        snapped = snap_to_grid(y_hat_arr, delta, mode="nearest", nonneg=True)
        offgrid = np.isfinite(y_hat_arr) & (np.abs(y_hat_arr - snapped) > DEFAULT_DQC_POLICY.tol)
        if bool(np.any(offgrid)):
            raise ValueError("Forecast contains off-grid values under PACKED/QUANTIZED DQC policy.")
        return y_hat_arr

    if enforce == "snap":
        return snap_to_grid(y_hat_arr, delta, mode=mode, nonneg=True)

    raise ValueError(f"Unsupported enforce mode: {enforce!r}")


def hr_at_tau_grid_units(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    *,
    dqc: DQCResult,
    tau_units: float,
    enforce: EnforcementMode = "snap",
    snap_mode: SnapMode = "nearest",
) -> float:
    """Compute HR@τ where τ is measured in grid units (Δ*).

    For PACKED / QUANTIZED demand:
    - Forecasts are snapped per enforcement policy (default: snap).
    - Error is evaluated in grid units: |y - yhat| / Δ* <= τ_units.

    Delegates to eb-metrics `hr_at_tau` after converting τ into y-units.

    Args:
        y_true: Realized demand.
        y_hat: Forecast demand.
        dqc: DQCResult for the relevant entity/window.
        tau_units: Tolerance in grid units (Δ* multiples).
        enforce: Snapping enforcement behavior.
        snap_mode: Snapping mode for y_hat when enforce == "snap".

    Returns:
        Hit rate in [0,1].
    """
    if _hr_at_tau is None:  # pragma: no cover
        raise ImportError(
            "eb-metrics is required to compute HR@τ (missing eb_metrics.metrics.service.hr_at_tau)."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)

    if dqc.dqc_class in ("PACKED", "QUANTIZED") and dqc.delta_star is not None:
        y_hat_arr = enforce_snapping(y_hat_arr, dqc=dqc, enforce=enforce, mode=snap_mode)
        tau = float(tau_units) * float(dqc.delta_star)
    else:
        # Continuous-like: interpret tau_units as y-units directly (caller responsibility).
        tau = float(tau_units)

    return float(_hr_at_tau(y_true_arr, y_hat_arr, tau=tau))
