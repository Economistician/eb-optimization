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


def test_snap_to_grid_nearest_preserves_nan_and_nonneg() -> None:
    x = np.array([np.nan, -0.2, 0.2, 0.49, 0.51, 1.49, 1.51], dtype=float)
    got = snap_to_grid(x, 0.5, mode="nearest", nonneg=True)

    assert np.isnan(got[0])
    assert got[1] == 0.0  # clamped
    assert got[2] == 0.0
    assert got[3] == 0.5
    assert got[4] == 0.5
    assert got[5] == 1.5
    assert got[6] == 1.5


def test_snap_to_grid_floor_and_ceil() -> None:
    x = np.array([0.1, 0.9, 1.1, 1.9], dtype=float)
    floor = snap_to_grid(x, 1.0, mode="floor", nonneg=True)
    ceil = snap_to_grid(x, 1.0, mode="ceil", nonneg=True)

    assert np.allclose(floor, np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.allclose(ceil, np.array([1.0, 1.0, 2.0, 2.0]))


def test_compute_dqc_returns_continuous_when_insufficient_signal() -> None:
    # This path should not require eb-evaluation because we return early.
    policy = DQCPolicy(min_n_pos=50)
    y = np.array([0, 0, 1, 2, 3, 4], dtype=float)  # only 4 positives

    dqc = compute_dqc(y, policy=policy, use_positive_only=True)

    assert dqc.dqc_class == "CONTINUOUS"
    assert dqc.delta_star is None
    assert dqc.rho_star is None
    assert dqc.n_pos == 4


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
    yhat = np.array([1.1, 2.9, 4.2, np.nan], dtype=float)
    got = enforce_snapping(yhat, dqc=dqc, enforce="snap", mode="nearest")

    assert np.allclose(got[:3], np.array([2.0, 2.0, 4.0]))
    assert np.isnan(got[3])


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

    yhat = np.array([1.1, 2.9, 4.2, np.nan], dtype=float)
    got = enforce_snapping(yhat, dqc=eval_dqc, enforce="snap", mode="nearest")

    assert np.allclose(got[:3], np.array([2.0, 2.0, 4.0]))
    assert np.isnan(got[3])


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
