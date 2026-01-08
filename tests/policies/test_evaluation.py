"""Tests for eb_optimization.policies.evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from eb_optimization.policies.dqc_policy import compute_dqc
from eb_optimization.policies.evaluation import evaluate_with_dqc_hr


def test_evaluate_with_dqc_hr_uses_provided_dqc_and_scales_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["y_true"] = np.asarray(y_true, dtype=float)
        called["y_hat"] = np.asarray(y_hat, dtype=float)
        called["tau"] = float(tau)
        return 0.777

    import eb_optimization.policies.evaluation as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    # Construct a PACKED DQCResult with Δ*=2.0
    y_hist = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    dqc = compute_dqc(y_hist)

    y_true = np.array([2.0, 4.0, 6.0], dtype=float)
    y_hat = np.array([2.1, 3.6, 6.9], dtype=float)  # off-grid floats
    tau_units = 1.0

    out = evaluate_with_dqc_hr(
        y_true,
        y_hat,
        tau_units=tau_units,
        dqc=dqc,
        enforce="snap",
        snap_mode="nearest",
    )

    assert out.hr_at_tau == 0.777
    assert out.dqc == dqc
    assert out.tau_units == tau_units
    assert out.tau_y_units == 2.0  # 1 * Δ*

    # Forecast should be snapped to Δ*=2.0 (nearest multiples)
    assert np.allclose(called["y_hat"], np.array([2.0, 4.0, 6.0]))
    assert np.allclose(called["y_true"], y_true)
    assert called["tau"] == 2.0


def test_evaluate_with_dqc_hr_computes_dqc_from_y_for_dqc(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["tau"] = float(tau)
        return 0.123

    import eb_optimization.policies.evaluation as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    # Provide a packed historical series for DQC computation
    y_for_dqc = np.tile(np.array([1.0, 2.0, 3.0, 4.0], dtype=float), 100)  # Δ*=1.0 expected

    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_hat = np.array([1.2, 2.0, 2.8], dtype=float)

    out = evaluate_with_dqc_hr(
        y_true,
        y_hat,
        tau_units=2.0,
        dqc=None,
        y_for_dqc=y_for_dqc,
        enforce="snap",
    )

    assert out.dqc.delta_star == 1.0
    assert out.tau_y_units == 2.0  # 2 * 1.0
    assert out.hr_at_tau == 0.123
    assert called["tau"] == 2.0


def test_evaluate_with_dqc_hr_continuous_pass_through_tau_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["tau"] = float(tau)
        return 0.999

    import eb_optimization.policies.evaluation as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    # Force CONTINUOUS by using a policy requiring more positive samples than provided
    # (This avoids relying on random noise structure in a test.)
    y_for_dqc = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=float)

    y_true = np.array([1.0, 2.0], dtype=float)
    y_hat = np.array([1.1, 2.2], dtype=float)

    out = evaluate_with_dqc_hr(
        y_true,
        y_hat,
        tau_units=0.5,
        dqc=None,
        y_for_dqc=y_for_dqc,
        # Compute DQC from too-little data -> returns CONTINUOUS by default
    )

    assert out.dqc.dqc_class == "CONTINUOUS"
    assert out.tau_y_units == 0.5
    assert out.hr_at_tau == 0.999
    assert called["tau"] == 0.5


def test_evaluate_with_dqc_hr_raise_when_offgrid(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        return 0.0

    import eb_optimization.policies.evaluation as mod

    monkeypatch.setattr(mod, "_hr_at_tau", fake_hr_at_tau)

    # Packed Δ*=1.0
    y_hist = np.tile(np.array([1.0, 2.0, 3.0, 4.0], dtype=float), 100)
    dqc = compute_dqc(y_hist)

    y_true = np.array([1.0, 2.0], dtype=float)
    y_hat = np.array([1.0, 2.5], dtype=float)  # 2.5 is off-grid for Δ=1.0

    with pytest.raises(ValueError, match="off-grid"):
        evaluate_with_dqc_hr(
            y_true,
            y_hat,
            tau_units=1.0,
            dqc=dqc,
            enforce="raise",
        )
