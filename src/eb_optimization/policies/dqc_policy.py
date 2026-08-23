"""DQC policy: snap forecasts to Δ* and interpret τ in grid units.

Consumes a DQC diagnostic result from ``eb-evaluation`` and enforces
snap / raise modes for packed or quantized demand. ``enforce="ignore"``
is hard-deprecated and always raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Any, Literal

import numpy as np

try:
    # eb-optimization does not define metric primitives; it delegates to eb-metrics.
    from eb_metrics.metrics.service import hr_at_tau as _hr_at_tau
except Exception:  # pragma: no cover
    _hr_at_tau = None

# Preferred DQC source: public evaluation root (full-series classify_dqc).
try:  # pragma: no cover - import guard
    from eb_evaluation import classify_dqc as _classify_dqc
except Exception:  # pragma: no cover - import guard
    _classify_dqc = None


DQCClass = Literal["CONTINUOUS", "QUANTIZED", "PACKED", "UNKNOWN"]
SnapMode = Literal["nearest", "floor", "ceil"]
EnforcementMode = Literal["snap", "raise", "ignore"]


@dataclass(frozen=True, slots=True)
class DQCPolicy:
    """Legacy DQCPolicy container.

    This policy previously owned DQC *detection* thresholds. DQC detection now lives
    in `eb-evaluation` (diagnostics). This object remains for backwards compatibility
    with callers that may still pass `policy=` into `compute_dqc`.

    New code should:
    - run DQC in `eb-evaluation` (e.g., `validate_dqc(y=...)`)
    - pass the resulting DQCResult into `enforce_snapping` / `hr_at_tau_grid_units`
    """

    # Minimum positive support required to consider a DQC result meaningful.
    min_n_pos: int = 50


DEFAULT_DQC_POLICY = DQCPolicy()


@dataclass(frozen=True, slots=True)
class DQCResult:
    """Output of DQC computation (policy-facing summary).

    Notes
    -----
    - `delta_star` is the inferred grid (Δ*) in y-units.
    - This is a lightweight summary shape used by eb-optimization policy code.
    - When using eb-evaluation DQC, `delta_star` maps to `signals.granularity`.
    - The public export name is :data:`DQCResultSummary`. That alias avoids colliding
      with ``eb_evaluation.DQCResult``, which is the full diagnostic result.
    """

    dqc_class: DQCClass
    delta_star: float | None
    rho_star: float | None

    # Evidence / diagnostics
    n_pos: int
    support_size: int
    offgrid_mad_over_delta: float | None


# Public name that does not collide with eb_evaluation.DQCResult (diagnostic result).
DQCResultSummary = DQCResult


_EVAL_SNAP_MODE = {"nearest": "round", "floor": "floor", "ceil": "ceil"}


def _snap_eval_math(values: Sequence[float], unit: float, *, mode: str) -> list[float]:
    """Match ``eb_evaluation.diagnostics.governance.snap_to_grid`` arithmetic."""
    snapped: list[float] = []
    inv = 1.0 / unit
    for v in values:
        fv = float(v)
        q = fv * inv
        if mode == "ceil":
            qi = ceil(q)
        elif mode == "floor":
            qi = floor(q)
        else:
            qi = floor(q + 0.5) if q >= 0.0 else ceil(q - 0.5)
        snapped.append(float(qi) * unit)
    return snapped


def snap_to_grid(
    x: np.ndarray,
    delta: float,
    *,
    mode: SnapMode = "ceil",
    nonneg: bool = True,
) -> np.ndarray:
    """Project values onto multiples of delta.

    Snapping arithmetic is the evaluation engine: ``math.ceil`` / ``math.floor``
    and half-away-from-zero for ``nearest`` (mapped to evaluation ``round``).
    Non-finite cells raise. ``nonneg=True`` clamps after snapping.

    Args:
        x: Array of values to snap. Must be finite.
        delta: Grid size (Δ). Must be > 0.
        mode: Ceil (default, matches ``apply_ral``), floor, or nearest snapping.
        nonneg: If True, clamps to >= 0 after snapping.

    Returns:
        Snapped array (float dtype).

    Raises:
        ValueError
            If any cell is NaN or ±inf.
    """
    if not (isinstance(delta, int | float) and delta > 0.0):
        raise ValueError(f"delta must be > 0; got {delta!r}")

    eval_mode = _EVAL_SNAP_MODE.get(mode)
    if eval_mode is None:
        raise ValueError(f"Unsupported mode: {mode!r}")

    x = np.asarray(x, dtype=float)
    if x.size and not bool(np.isfinite(x).all()):
        raise ValueError(
            "snap_to_grid values must be finite; refusing fail-open NaN/inf forecasts."
        )

    values = x.tolist()
    try:
        from eb_evaluation.diagnostics.governance import snap_to_grid as eval_snap

        snapped = np.asarray(eval_snap(values, float(delta), mode=eval_mode), dtype=float)
    except ImportError:  # pragma: no cover - evaluation is optional at import
        snapped = np.asarray(
            _snap_eval_math(values, float(delta), mode=eval_mode),
            dtype=float,
        )

    if nonneg:
        snapped = np.maximum(snapped, 0.0)
    return snapped


def _type_label(obj: Any) -> str:
    try:
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    except Exception:  # pragma: no cover
        return str(type(obj))


def _get_eval_granularity(dqc: Any) -> float | None:
    """Extract Δ* (granularity) from an eb-evaluation DQCResult-like object."""
    try:
        signals = dqc.signals
        g = signals.granularity
        if g is None:
            return None
        fg = float(g)
        return fg if fg > 0 else None
    except Exception:
        return None


def _get_eval_class_value(dqc: Any) -> str:
    """Return a normalized eb-evaluation DQC class value (lowercase) if possible."""
    try:
        cls = dqc.dqc_class
    except Exception:
        return ""

    # Enum-like (preferred)
    if hasattr(cls, "value"):
        try:
            return str(cls.value).lower()
        except Exception:
            return ""

    # Fallback stringification
    try:
        return str(cls).lower()
    except Exception:
        return ""


def _map_eval_dqc_to_policy_dqc(dqc: Any) -> DQCResult:
    """Map eb-evaluation DQCResult -> eb-optimization DQCResult summary."""
    granularity = _get_eval_granularity(dqc)

    # Class mapping (eb-evaluation -> policy). Unrecognized values fail closed.
    cls_val = _get_eval_class_value(dqc)

    if cls_val == "quantized":
        dqc_class: DQCClass = "QUANTIZED"
    elif cls_val == "piecewise_packed":
        dqc_class = "PACKED"
    elif cls_val == "continuous_like":
        dqc_class = "CONTINUOUS"
    elif cls_val in ("unknown", ""):
        dqc_class = "UNKNOWN"
    else:
        raise ValueError(
            f"Unrecognized DQC class {cls_val!r}; refusing fail-open CONTINUOUS mapping."
        )

    rho_star: float | None = None
    support_size: int = 0
    offgrid_mad_over_delta: float | None = None

    try:
        signals = dqc.signals
        rho_star = float(signals.multiple_rate)
        support_size = int(signals.support_size)
        offgrid_mad = float(signals.offgrid_mad)
        if granularity is not None and granularity > 0:
            offgrid_mad_over_delta = offgrid_mad / granularity
    except Exception:
        pass

    # `n_pos` is not carried explicitly by eb-evaluation DQC; callers can compute it
    # from their realized series if needed. We set it to 0 here and optionally fill
    # it in compute_dqc().
    return DQCResult(
        dqc_class=dqc_class,
        delta_star=granularity,
        rho_star=rho_star,
        n_pos=0,
        support_size=support_size,
        offgrid_mad_over_delta=offgrid_mad_over_delta,
    )


def _as_series_list(y: Any) -> list[object]:
    """Preserve the full caller series, including zeros and non-finite cells."""
    if isinstance(y, np.ndarray):
        return y.tolist()
    if isinstance(y, (list, tuple)):
        return list(y)
    try:
        return list(y)
    except TypeError:
        return [y]


def compute_dqc(
    y: Any,
    *,
    policy: DQCPolicy = DEFAULT_DQC_POLICY,
    use_positive_only: bool = True,
) -> DQCResult:
    """Compute DQC by delegating the full series to ``eb_evaluation.classify_dqc``.

    Classification uses the complete input series so scoring and governance
    produce the same DQC class and grid as the evaluation diagnostic. ``policy``
    and ``use_positive_only`` are retained for call-site compatibility; they do
    not subset the series before classification. ``n_pos`` is copied from the
    evaluation ``nonzero_obs`` signal.

    Args:
        y: Realized demand sequence (full series).
        policy: Legacy DQCPolicy (unused for classification).
        use_positive_only: Unused for classification; retained for compatibility.

    Returns:
        DQCResult summary mapped from the evaluation diagnostic.
    """
    del policy, use_positive_only

    if _classify_dqc is None:
        raise ImportError(
            "DQC diagnostics are not available. Install/enable `eb-evaluation` to compute DQC, "
            "or run DQC in the evaluation layer and pass the result into policy enforcement."
        )

    eval_result = _classify_dqc(y=_as_series_list(y), thresholds=None)
    mapped = _map_eval_dqc_to_policy_dqc(eval_result)
    n_pos = int(getattr(eval_result.signals, "nonzero_obs", 0) or 0)
    return DQCResult(
        dqc_class=mapped.dqc_class,
        delta_star=mapped.delta_star,
        rho_star=mapped.rho_star,
        n_pos=n_pos,
        support_size=mapped.support_size,
        offgrid_mad_over_delta=mapped.offgrid_mad_over_delta,
    )


def _require_known_dqc_class(dqc_class: DQCClass) -> DQCClass:
    """Refuse UNKNOWN so enforcement cannot fail open to CONTINUOUS."""
    if dqc_class == "UNKNOWN":
        raise ValueError(
            "DQC class is UNKNOWN (insufficient evidence or unrecognized class); "
            "refusing fail-open CONTINUOUS enforcement."
        )
    return dqc_class


def _require_grid_delta(dqc_class: DQCClass, delta: float | None) -> float:
    """Require a usable Δ* whenever snapping is required."""
    if dqc_class not in ("QUANTIZED", "PACKED"):
        raise ValueError(f"Grid delta is only required for QUANTIZED/PACKED; got {dqc_class!r}.")
    if delta is None:
        raise ValueError(
            f"DQC class is {dqc_class} but delta_star/granularity is missing; "
            "refusing fail-open unsnapped forecasts."
        )
    try:
        value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DQC class is {dqc_class} but delta_star/granularity is invalid ({delta!r}); "
            "refusing fail-open unsnapped forecasts."
        ) from exc
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"DQC class is {dqc_class} but delta_star/granularity must be finite and > 0; "
            f"got {delta!r}."
        )
    return value


def _resolve_policy_class_and_delta(dqc: Any) -> tuple[DQCClass, float | None]:
    """
    Resolve a DQCResult-like input into a policy class and Δ*.

    Supports:
    - this module's DQCResult (dqc_class, delta_star)
    - eb-evaluation DQCResult (dqc_class.value, signals.granularity)
    """
    if isinstance(dqc, DQCResult):
        return _require_known_dqc_class(dqc.dqc_class), dqc.delta_star

    delta = _get_eval_granularity(dqc)
    cls_val = _get_eval_class_value(dqc)

    if cls_val == "quantized":
        return "QUANTIZED", delta
    if cls_val == "piecewise_packed":
        return "PACKED", delta
    if cls_val == "continuous_like":
        return "CONTINUOUS", delta
    if cls_val in ("unknown", ""):
        raise ValueError(
            "DQC class is UNKNOWN (insufficient evidence or unrecognized class); "
            "refusing fail-open CONTINUOUS enforcement."
        )
    raise ValueError(f"Unrecognized DQC class {cls_val!r}; refusing fail-open CONTINUOUS mapping.")


def _require_finite_forecasts(values: np.ndarray, *, context: str) -> None:
    """Refuse NaN/±inf so enforcement cannot fail-open on any DQC class."""
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError(f"{context} values must be finite; refusing fail-open NaN/inf forecasts.")


def enforce_snapping(
    y_hat: Sequence[float] | np.ndarray,
    *,
    dqc: Any,
    enforce: EnforcementMode = "snap",
    mode: SnapMode = "ceil",
    tol: float = 1e-6,
) -> np.ndarray:
    """Apply DQC snapping enforcement to forecasts.

    Policy intent:
    - PACKED / QUANTIZED demand => snapping is required (unit compatibility).
    - CONTINUOUS-like demand => no snapping, but forecasts must still be finite.
    - UNKNOWN or unrecognized class => raise (fail closed).
    - PACKED / QUANTIZED with missing or invalid Δ* => raise (fail closed).
    - ``enforce="ignore"`` is hard-deprecated and always raises.

    Args:
        y_hat: Forecast values. Must be finite.
        dqc: DQCResult-like object (preferred: eb-evaluation DQCResult) OR this module's DQCResult.
        enforce: "snap" (default) or "raise" (error if off-grid).
        mode: Snapping mode (default ``ceil``, matching ``apply_ral``).
        tol: Absolute tolerance for off-grid checks (used when enforce == "raise").

    Returns:
        Forecast array, snapped or unchanged depending on class and enforcement.

    Raises:
        ValueError: If class is UNKNOWN, unrecognized, or QUANTIZED/PACKED without a
            finite positive Δ*; if any forecast cell is non-finite; or if
            ``enforce="ignore"``.
    """
    y_hat_arr = np.asarray(y_hat, dtype=float)
    if enforce == "ignore":
        raise ValueError(
            "enforce='ignore' is hard-deprecated on enforce_snapping. "
            "Use electric_barometer.apply_ral (or eb_evaluation.apply_ral) "
            "with a valid governance decisions dataframe."
        )
    _require_finite_forecasts(y_hat_arr, context="enforce_snapping")

    dqc_class, delta = _resolve_policy_class_and_delta(dqc)

    if dqc_class == "CONTINUOUS":
        return y_hat_arr

    unit = _require_grid_delta(dqc_class, delta)

    if enforce == "raise":
        snapped = snap_to_grid(y_hat_arr, unit, mode=mode, nonneg=True)
        offgrid = np.abs(y_hat_arr - snapped) > tol
        if bool(np.any(offgrid)):
            raise ValueError(
                "Forecast contains off-grid values under PACKED/QUANTIZED DQC policy. "
                "Either snap forecasts before evaluation or use enforce='snap'."
            )
        return y_hat_arr

    if enforce == "snap":
        return snap_to_grid(y_hat_arr, unit, mode=mode, nonneg=True)

    raise ValueError(f"Unsupported enforce mode: {enforce!r}")


def hr_at_tau_grid_units(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    *,
    dqc: Any,
    tau_units: float,
    enforce: EnforcementMode = "snap",
    snap_mode: SnapMode = "ceil",
) -> float:
    """Compute HR@τ where τ is measured in grid units (Δ*).

    For PACKED / QUANTIZED demand:
    - Forecasts are snapped per enforcement policy (default: snap).
    - Error is evaluated in grid units: |y - yhat| / Δ* <= τ_units.
    - We convert tau_units to y-units via tau = tau_units * Δ*.

    For CONTINUOUS-like demand:
    - tau_units is interpreted as y-units directly (caller responsibility).

    Delegates to eb-metrics `hr_at_tau` after converting τ into y-units.
    """
    if _hr_at_tau is None:  # pragma: no cover
        raise ImportError(
            "eb-metrics is required to compute HR@τ (missing eb_metrics.metrics.service.hr_at_tau)."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)

    dqc_class, delta = _resolve_policy_class_and_delta(dqc)

    if dqc_class in ("PACKED", "QUANTIZED"):
        unit = _require_grid_delta(dqc_class, delta)
        y_hat_arr = enforce_snapping(y_hat_arr, dqc=dqc, enforce=enforce, mode=snap_mode)
        tau = float(tau_units) * unit
    elif dqc_class == "CONTINUOUS":
        tau = float(tau_units)
    else:
        raise ValueError(
            f"DQC class is {dqc_class!r}; refusing fail-open CONTINUOUS HR@τ interpretation."
        )

    return float(_hr_at_tau(y_true_arr, y_hat_arr, tau=tau))
