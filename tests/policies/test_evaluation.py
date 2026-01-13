"""Tests for eb_optimization.policies.evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from eb_optimization.policies.dqc_policy import DQCPolicy, compute_dqc
from eb_optimization.policies.evaluation import evaluate_with_dqc_hr


def _require_eb_evaluation() -> None:
    """Skip tests that require eb-evaluation to be installed/available."""
    pytest.importorskip("eb_evaluation", reason="eb-evaluation not installed/available")


def _patch_eval_hr_at_tau(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[..., float],
) -> None:
    """Patch the HR@τ primitive used by policy evaluation.

    The evaluation layer typically delegates HR@τ computation through
    `eb_optimization.policies.dqc_policy.hr_at_tau_grid_units`, which in turn
    calls the module-global `_hr_at_tau` inside `dqc_policy`.

    So: patch `eb_optimization.policies.dqc_policy._hr_at_tau` (primary).
    We also patch `eb_optimization.policies.evaluation._hr_at_tau` if present
    (harmless, but covers alternate implementations).
    """
    import eb_optimization.policies.dqc_policy as dqc_mod

    monkeypatch.setattr(dqc_mod, "_hr_at_tau", fn)

    # Optional: if evaluation.py defines its own _hr_at_tau, patch it too.
    import eb_optimization.policies.evaluation as eval_mod

    if hasattr(eval_mod, "_hr_at_tau"):
        monkeypatch.setattr(eval_mod, "_hr_at_tau", fn)


def test_evaluate_with_dqc_hr_uses_provided_dqc_and_scales_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_eb_evaluation()

    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["y_true"] = np.asarray(y_true, dtype=float)
        called["y_hat"] = np.asarray(y_hat, dtype=float)
        called["tau"] = float(tau)
        return 0.777

    _patch_eval_hr_at_tau(monkeypatch, fake_hr_at_tau)

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
    assert np.allclose(np.asarray(called["y_hat"]), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(np.asarray(called["y_true"]), y_true)
    assert called["tau"] == 2.0


def test_evaluate_with_dqc_hr_accepts_eb_evaluation_dqc_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_eb_evaluation()
    from eb_evaluation.diagnostics import validate_dqc

    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["y_true"] = np.asarray(y_true, dtype=float)
        called["y_hat"] = np.asarray(y_hat, dtype=float)
        called["tau"] = float(tau)
        return 0.555

    _patch_eval_hr_at_tau(monkeypatch, fake_hr_at_tau)

    # Create eb-evaluation DQCResult (Δ*=2.0 for this pattern)
    y_hist = np.tile(np.array([2.0, 4.0, 6.0, 8.0], dtype=float), 100)
    eval_dqc = validate_dqc(y=y_hist.tolist())

    y_true = np.array([2.0, 4.0, 6.0], dtype=float)
    y_hat = np.array([2.1, 3.6, 6.9], dtype=float)
    tau_units = 1.0

    out = evaluate_with_dqc_hr(
        y_true,
        y_hat,
        tau_units=tau_units,
        dqc=eval_dqc,  # NOTE: eb-evaluation DQCResult
        enforce="snap",
        snap_mode="nearest",
    )

    assert out.hr_at_tau == 0.555
    assert out.tau_units == tau_units
    assert out.tau_y_units == 2.0

    assert np.allclose(np.asarray(called["y_hat"]), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(np.asarray(called["y_true"]), y_true)
    assert called["tau"] == 2.0


def test_evaluate_with_dqc_hr_computes_dqc_from_y_for_dqc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_eb_evaluation()

    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["tau"] = float(tau)
        return 0.123

    _patch_eval_hr_at_tau(monkeypatch, fake_hr_at_tau)

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
    called: dict[str, object] = {}

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        called["tau"] = float(tau)
        return 0.999

    _patch_eval_hr_at_tau(monkeypatch, fake_hr_at_tau)

    # Force CONTINUOUS by requiring more positive samples than we provide.
    # This ensures the compute_dqc early-return path (no eb-evaluation dependency).
    y_for_dqc = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
    strict_policy = DQCPolicy(min_n_pos=10)

    y_true = np.array([1.0, 2.0], dtype=float)
    y_hat = np.array([1.1, 2.2], dtype=float)

    out = evaluate_with_dqc_hr(
        y_true,
        y_hat,
        tau_units=0.5,
        dqc=None,
        y_for_dqc=y_for_dqc,
        policy=strict_policy,
    )

    assert out.dqc.dqc_class == "CONTINUOUS"
    assert out.tau_y_units == 0.5
    assert out.hr_at_tau == 0.999
    assert called["tau"] == 0.5


def test_evaluate_with_dqc_hr_raise_when_offgrid(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_eb_evaluation()

    def fake_hr_at_tau(y_true, y_hat, *, tau):  # type: ignore[no-untyped-def]
        return 0.0

    _patch_eval_hr_at_tau(monkeypatch, fake_hr_at_tau)

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
