"""Tests for eb_optimization.policies.dqc_policy."""

from __future__ import annotations

import numpy as np
import pytest

from eb_optimization.policies.dqc_policy import (
    DEFAULT_DQC_POLICY,
    DQCPolicy,
    compute_dqc,
    enforce_snapping,
    hr_at_tau_grid_units,
    snap_to_grid,
)


def _require_eb_evaluation() -> None:
    """Skip tests that require eb-evaluation to be installed/available."""
    pytest.importorskip("eb_evaluation", reason="eb-evaluation not installed/available")


def test_snap_to_grid_nearest_clamps_nonneg() -> None:
    x = np.array([-0.2, 0.2, 0.49, 0.51, 1.49, 1.51], dtype=float)
    got = snap_to_grid(x, 0.5, mode="nearest", nonneg=True)

    assert got[0] == 0.0  # clamped
    assert got[1] == 0.0
    assert got[2] == 0.5
    assert got[3] == 0.5
    assert got[4] == 1.5
    assert got[5] == 1.5


def test_snap_to_grid_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        snap_to_grid(np.array([1.0, np.nan], dtype=float), 0.5, mode="nearest")
    with pytest.raises(ValueError, match="finite"):
        snap_to_grid(np.array([1.0, np.inf], dtype=float), 0.5, mode="nearest")


def test_snap_to_grid_floor_and_ceil() -> None:
    x = np.array([0.1, 0.9, 1.1, 1.9], dtype=float)
    floor = snap_to_grid(x, 1.0, mode="floor", nonneg=True)
    ceil = snap_to_grid(x, 1.0, mode="ceil", nonneg=True)

    assert np.allclose(floor, np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.allclose(ceil, np.array([1.0, 1.0, 2.0, 2.0]))


def test_snap_to_grid_nearest_is_half_away_from_zero() -> None:
    got = snap_to_grid(np.array([0.5, -1.5], dtype=float), 1.0, mode="nearest", nonneg=False)
    np.testing.assert_allclose(got, np.array([1.0, -2.0], dtype=float))


def test_compute_dqc_does_not_short_circuit_on_min_n_pos() -> None:
    _require_eb_evaluation()
    from eb_evaluation import classify_dqc

    policy = DQCPolicy(min_n_pos=50)
    y = np.array([0, 0, 1, 2, 3, 4], dtype=float)

    dqc = compute_dqc(y, policy=policy, use_positive_only=True)
    eval_dqc = classify_dqc(y.tolist())

    expected = {
        "continuous_like": "CONTINUOUS",
        "quantized": "QUANTIZED",
        "piecewise_packed": "PACKED",
        "unknown": "UNKNOWN",
    }[eval_dqc.dqc_class.value]
    assert dqc.dqc_class == expected
    assert dqc.n_pos == eval_dqc.signals.nonzero_obs == 4


def test_compute_dqc_matches_classify_dqc_on_full_series() -> None:
    _require_eb_evaluation()
    from eb_evaluation import classify_dqc

    y = [0.0] * 60 + [4.0] * 120 + [8.0] * 120 + [12.0] * 120 + [16.0] * 60
    eval_dqc = classify_dqc(y)
    dqc = compute_dqc(y)

    expected = {
        "continuous_like": "CONTINUOUS",
        "quantized": "QUANTIZED",
        "piecewise_packed": "PACKED",
        "unknown": "UNKNOWN",
    }[eval_dqc.dqc_class.value]
    assert dqc.dqc_class == expected
    assert dqc.delta_star == eval_dqc.signals.granularity
    assert dqc.n_pos == eval_dqc.signals.nonzero_obs
    assert dqc.support_size == eval_dqc.signals.support_size


def test_enforce_snapping_unknown_fails_closed() -> None:
    from eb_optimization.policies.dqc_policy import DQCResult

    dqc = DQCResult(
        dqc_class="UNKNOWN",
        delta_star=None,
        rho_star=None,
        n_pos=4,
        support_size=3,
        offgrid_mad_over_delta=None,
    )
    yhat = np.array([1.0, 2.0], dtype=float)
    with pytest.raises(ValueError, match="UNKNOWN"):
        enforce_snapping(yhat, dqc=dqc, enforce="snap")


def test_enforce_snapping_evaluation_unknown_fails_closed() -> None:
    pytest.importorskip("eb_evaluation", reason="eb-evaluation not installed/available")
    from eb_evaluation.diagnostics.dqc import DQCClass, DQCResult, DQCSignals

    dqc = DQCResult(
        dqc_class=DQCClass.UNKNOWN,
        signals=DQCSignals(
            n_obs=4,
            nonzero_obs=4,
            granularity=None,
            multiple_rate=float("nan"),
            support_size=3,
            zero_mass=0.0,
            small_value_mass=0.0,
            offgrid_mad=float("nan"),
            candidate_units=(),
            unit_scores=(),
        ),
        reasons=("insufficient_nonzero_obs",),
    )
    yhat = np.array([1.0, 2.0], dtype=float)
    with pytest.raises(ValueError, match="UNKNOWN"):
        enforce_snapping(yhat, dqc=dqc, enforce="snap")


def test_enforce_snapping_quantized_without_delta_fails_closed() -> None:
    from eb_optimization.policies.dqc_policy import DQCResult, enforce_snapping

    dqc = DQCResult(
        dqc_class="QUANTIZED",
        delta_star=None,
        rho_star=1.0,
        n_pos=80,
        support_size=4,
        offgrid_mad_over_delta=0.0,
    )
    yhat = np.array([1.1, 2.2], dtype=float)
    with pytest.raises(ValueError, match=r"delta_star|granularity"):
        enforce_snapping(yhat, dqc=dqc, enforce="snap")


def test_enforce_snapping_packed_invalid_delta_fails_closed() -> None:
    from eb_optimization.policies.dqc_policy import DQCResult, enforce_snapping

    dqc = DQCResult(
        dqc_class="PACKED",
        delta_star=0.0,
        rho_star=1.0,
        n_pos=80,
        support_size=4,
        offgrid_mad_over_delta=0.0,
    )
    yhat = np.array([1.1, 2.2], dtype=float)
    with pytest.raises(ValueError, match=r"delta_star|granularity"):
        enforce_snapping(yhat, dqc=dqc, enforce="snap")


def test_evaluate_helpers_default_enforce_is_snap() -> None:
    import inspect

    from eb_optimization.policies.dqc_policy import enforce_snapping
    from eb_optimization.policies.evaluation import evaluate_with_dqc_hr

    assert inspect.signature(enforce_snapping).parameters["enforce"].default == "snap"
    assert inspect.signature(enforce_snapping).parameters["mode"].default == "ceil"
    assert inspect.signature(evaluate_with_dqc_hr).parameters["enforce"].default == "snap"
    assert inspect.signature(evaluate_with_dqc_hr).parameters["snap_mode"].default == "ceil"


def test_compute_dqc_detects_packed_grid() -> None:
    _require_eb_evaluation()

    # Perfectly aligned to Δ=2.0 (all values are multiples of 2)
    y = np.tile(np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float), 100)

    dqc = compute_dqc(y, policy=DEFAULT_DQC_POLICY, use_positive_only=True)

    # With eb-evaluation DQC, this pattern should be detected as "packed" (piecewise packed)
    # and map into policy as PACKED with Δ*=2.0.
    assert dqc.dqc_class == "PACKED"
    assert dqc.delta_star == 2.0
    assert dqc.rho_star == 1.0
    assert dqc.offgrid_mad_over_delta == 0.0
    assert dqc.n_pos == 500


def test_enforce_snapping_snap_for_packed() -> None:
    _require_eb_evaluation()

    # Packed at Δ*=2.0
    y = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    dqc = compute_dqc(y, policy=DEFAULT_DQC_POLICY)

    # Off-grid forecasts should be snapped
    yhat = np.array([1.1, 2.9, 4.2, 5.1], dtype=float)
    got = enforce_snapping(yhat, dqc=dqc, enforce="snap", mode="nearest")

    assert np.allclose(got, np.array([2.0, 2.0, 4.0, 6.0]))


def test_enforce_snapping_ignore_is_hard_deprecated() -> None:
    from eb_optimization.policies.dqc_policy import DQCResult

    dqc = DQCResult(
        dqc_class="CONTINUOUS",
        delta_star=None,
        rho_star=None,
        n_pos=4,
        support_size=4,
        offgrid_mad_over_delta=None,
    )
    with pytest.raises(ValueError, match=r"electric_barometer\.apply_ral"):
        enforce_snapping(np.array([1.0, 2.0], dtype=float), dqc=dqc, enforce="ignore")


def test_enforce_snapping_continuous_rejects_nan() -> None:
    from eb_optimization.policies.dqc_policy import DQCResult

    dqc = DQCResult(
        dqc_class="CONTINUOUS",
        delta_star=None,
        rho_star=None,
        n_pos=4,
        support_size=4,
        offgrid_mad_over_delta=None,
    )
    with pytest.raises(ValueError, match="finite"):
        enforce_snapping(np.array([1.0, np.nan], dtype=float), dqc=dqc, enforce="snap")


def test_enforce_snapping_rejects_nan_when_snap_required() -> None:
    _require_eb_evaluation()
    y = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    dqc = compute_dqc(y, policy=DEFAULT_DQC_POLICY)
    with pytest.raises(ValueError, match="finite"):
        enforce_snapping(
            np.array([1.1, np.nan], dtype=float), dqc=dqc, enforce="snap", mode="nearest"
        )


def test_enforce_snapping_raise_when_offgrid() -> None:
    _require_eb_evaluation()

    # Packed at Δ*=1.0
    y = np.tile(np.array([1.0, 2.0, 3.0, 4.0], dtype=float), 100)
    dqc = compute_dqc(y, policy=DEFAULT_DQC_POLICY)

    yhat = np.array([1.0, 2.0, 2.5], dtype=float)  # 2.5 is off-grid for Δ=1
    with pytest.raises(ValueError, match="off-grid"):
        enforce_snapping(yhat, dqc=dqc, enforce="raise")


def test_enforce_snapping_accepts_eb_evaluation_dqc_result() -> None:
    _require_eb_evaluation()
    from eb_evaluation.diagnostics import validate_dqc

    # Construct a clear grid signal at Δ*=2.0
    y = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    eval_dqc = validate_dqc(y=y.tolist())

    yhat = np.array([1.1, 2.9, 4.2, 5.1], dtype=float)
    got = enforce_snapping(yhat, dqc=eval_dqc, enforce="snap", mode="nearest")

    assert np.allclose(got, np.array([2.0, 2.0, 4.0, 6.0]))


def test_hr_at_tau_grid_units_delegates_and_snaps(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_eb_evaluation()

    # We monkeypatch the eb-metrics primitive to avoid depending on external behavior.
    # This also verifies tau scaling (tau_units * delta_star) and snapping is applied.
    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["y_true"] = np.asarray(y_true, dtype=float)
        called["y_hat"] = np.asarray(y_hat, dtype=float)
        called["tau"] = float(tau)
        # return something deterministic
        return 0.123

    import eb_optimization.policies.dqc_policy as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    # Packed grid Δ*=2.0
    y = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    dqc = compute_dqc(y, policy=DEFAULT_DQC_POLICY)

    y_true = np.array([2.0, 4.0, 6.0], dtype=float)
    y_hat = np.array([2.1, 3.6, 6.9], dtype=float)  # off-grid floats
    tau_units = 1  # 1 grid unit

    out = hr_at_tau_grid_units(
        y_true,
        y_hat,
        dqc=dqc,
        tau_units=tau_units,
        enforce="snap",
        snap_mode="nearest",
    )

    assert out == 0.123
    # Tau should be scaled to y-units: 1 * 2.0
    assert called["tau"] == 2.0
    # Forecast should be snapped to Δ*=2.0 before delegation:
    # 2.1 -> 2, 3.6 -> 4, 6.9 -> 6 (nearest multiples of 2)
    assert np.allclose(np.asarray(called["y_hat"]), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(np.asarray(called["y_true"]), y_true)


def test_hr_at_tau_grid_units_accepts_eb_evaluation_dqc_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_eb_evaluation()
    from eb_evaluation.diagnostics import validate_dqc

    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["y_true"] = np.asarray(y_true, dtype=float)
        called["y_hat"] = np.asarray(y_hat, dtype=float)
        called["tau"] = float(tau)
        return 0.456

    import eb_optimization.policies.dqc_policy as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    y_hist = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    eval_dqc = validate_dqc(y=y_hist.tolist())

    y_true = np.array([2.0, 4.0, 6.0], dtype=float)
    y_hat = np.array([2.1, 3.6, 6.9], dtype=float)
    tau_units = 1.0

    out = hr_at_tau_grid_units(
        y_true,
        y_hat,
        dqc=eval_dqc,
        tau_units=tau_units,
        enforce="snap",
        snap_mode="nearest",
    )

    assert out == 0.456
    # Δ* inferred from eval_dqc should be 2.0 for this pattern, so tau=2.0
    assert called["tau"] == 2.0
    assert np.allclose(np.asarray(called["y_hat"]), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(np.asarray(called["y_true"]), y_true)
