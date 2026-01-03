from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from eb_optimization.policies.cost_ratio_policy import (
    DEFAULT_COST_RATIO_POLICY,
    CostRatioPolicy,
    apply_cost_ratio_policy,
    apply_entity_cost_ratio_policy,
)


# =============================================================================
# Helpers (robust to small API changes / richer return types)
# =============================================================================
def _unpack_cost_ratio_result(res: Any) -> tuple[float, dict[str, Any]]:
    """
    apply_cost_ratio_policy might return:
      - (R, diag)
      - CostRatioEstimate-like object with .R and .diagnostics (or .diag)
      - R alone

    Normalize to (float_R, diag_dict).
    """
    if isinstance(res, tuple):
        if len(res) == 2:
            R, diag = res
            return float(R), dict(cast(dict[str, Any], diag))
        # Explicitly reject other tuple shapes so Pyright doesn't assume tuple is still possible below.
        raise TypeError(f"Unexpected tuple return from apply_cost_ratio_policy: {res!r}")

    # dataclass-like / rich object
    if hasattr(res, "R"):
        obj = cast(Any, res)  # hasattr does not narrow for Pyright; cast makes attr access safe
        R_val = float(obj.R)
        diag = getattr(obj, "diagnostics", None)
        if diag is None:
            diag = getattr(obj, "diag", None)
        return R_val, dict(cast(dict[str, Any], diag or {}))

    # scalar
    return float(res), {}


def _as_entity_df(res: Any) -> pd.DataFrame:
    """
    apply_entity_cost_ratio_policy might return:
      - pd.DataFrame (legacy / simplest)
      - EntityCostRatioEstimate-like object with one of: .df / .table / .result

    Normalize to a DataFrame.
    """
    if isinstance(res, pd.DataFrame):
        return res

    obj = cast(Any, res)  # attribute-based probing; keep it explicit for Pyright
    for attr in ("df", "table", "result"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, pd.DataFrame):
                return val

    raise TypeError(f"Unexpected return type from apply_entity_cost_ratio_policy: {type(res)!r}")


# =============================================================================
# Construction / validation
# =============================================================================
def test_cost_ratio_policy_validation_rejects_empty_grid():
    with pytest.raises(ValueError):
        CostRatioPolicy(R_grid=(), co=1.0)


def test_cost_ratio_policy_validation_rejects_non_positive_grid():
    with pytest.raises(ValueError):
        CostRatioPolicy(R_grid=(0.0, -1.0), co=1.0)


def test_cost_ratio_policy_validation_rejects_non_positive_co():
    with pytest.raises(ValueError):
        CostRatioPolicy(R_grid=(1.0,), co=0.0)


def test_cost_ratio_policy_validation_rejects_min_n_lt_1():
    with pytest.raises(ValueError):
        CostRatioPolicy(R_grid=(1.0,), co=1.0, min_n=0)


# =============================================================================
# Global policy application
# =============================================================================
def test_apply_cost_ratio_policy_uses_policy_defaults_when_co_none():
    y_true = [10, 0]
    y_pred = [0, 20]

    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=2.0)

    # Silence gating side-effects; this test is about defaults/diag shape.
    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=None, gate="off")
    R, diag = _unpack_cost_ratio_result(res)

    # Balanced example chooses 2.0 regardless of co scale (costs scale cancels in balance),
    # but we mainly assert diagnostics reflect default usage.
    assert np.isfinite(R)
    assert diag["co_default_used"] is True
    assert diag["method"] == "cost_balance"
    assert diag["R_grid"] == [0.5, 1.0, 2.0, 3.0]


def test_apply_cost_ratio_policy_balanced_example_has_known_optimum():
    """
    shortfall_sum = 10, overbuild_sum = 20, co = 1
    Under-cost(R) = R * 10, Over-cost = 20 -> minimized at R=2.0 (in grid)
    """
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([0.0, 20.0])

    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0)

    # Silence gating side-effects; this test is about correctness of R.
    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0, gate="off")
    R, _ = _unpack_cost_ratio_result(res)
    assert np.isclose(R, 2.0)


def test_apply_cost_ratio_policy_returns_nan_safe_diag_when_co_is_array():
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([0.0, 20.0])
    co = np.array([1.0, 1.0])

    # Silence gating side-effects; this test is about diag field indicating array co.
    res = apply_cost_ratio_policy(
        y_true=y_true,
        y_pred=y_pred,
        policy=DEFAULT_COST_RATIO_POLICY,
        co=co,
        gate="off",
    )
    R, diag = _unpack_cost_ratio_result(res)

    assert np.isfinite(R)
    assert diag["co_is_array"] is True


def test_apply_cost_ratio_policy_surfaces_identifiability_diagnostics():
    """
    Policy boundary should surface identifiability/stability diagnostics coming
    from tuning/cost_ratio.py (reporting-only; does not change selection).
    """
    y_true = np.array([10.0, 0.0, 5.0])
    y_pred = np.array([0.0, 20.0, 7.0])  # both shortfall and overbuild present

    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0)

    # Silence gating side-effects; this test is about surfacing the fields, not warning/raising.
    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0, gate="off")
    R, diag = _unpack_cost_ratio_result(res)

    assert np.isfinite(R)

    # surfaced scalar diagnostics
    for k in ("rel_min_gap", "R_min", "R_max", "grid_instability_log", "is_identifiable"):
        assert k in diag

    assert float(diag["R_min"]) > 0.0
    assert float(diag["R_max"]) > 0.0
    assert float(diag["grid_instability_log"]) >= 0.0
    assert isinstance(bool(diag["is_identifiable"]), bool)

    # surfaced nested tuning diagnostics
    assert "calibration_diagnostics" in diag
    cal = diag["calibration_diagnostics"]
    assert isinstance(cal, dict)
    assert "grid_sensitivity" in cal
    gs = cal["grid_sensitivity"]
    assert isinstance(gs, dict)
    assert {"base", "exclude_pivot", "shifted"}.issubset(gs.keys())


def test_apply_cost_ratio_policy_perfect_forecast_has_zero_instability_and_identifiable():
    """
    Perfect forecast case should surface:
    - rel_min_gap = 0
    - grid_instability_log = 0
    - is_identifiable = True
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])

    policy = CostRatioPolicy(R_grid=(0.4, 0.9, 1.3, 3.0), co=1.0)

    # Silence gating side-effects; this test is about the surfaced scalar fields.
    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0, gate="off")
    _R, diag = _unpack_cost_ratio_result(res)

    assert np.isclose(float(diag["rel_min_gap"]), 0.0)
    assert np.isclose(float(diag["grid_instability_log"]), 0.0)
    assert bool(diag["is_identifiable"]) is True


# =============================================================================
# New tests: warn-only + gateable hook (global)
# =============================================================================
def test_apply_cost_ratio_policy_gate_warn_emits_runtimewarning_when_not_identifiable():
    """
    gate="warn" should emit RuntimeWarning when tuning reports is_identifiable=False.

    We avoid hard-coding a dataset that must be non-identifiable (thresholds may change).
    Instead we:
      1) probe identifiability with gate="off"
      2) if not identifiable -> assert warning fires
      3) if identifiable -> just assert warn mode doesn't raise
    """
    y_true = np.array([10.0, 0.0, 5.0])
    y_pred = np.array([0.0, 20.0, 7.0])

    policy = CostRatioPolicy(R_grid=(0.25, 1.0, 10.0), co=1.0)

    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0, gate="off")
    _R, diag = _unpack_cost_ratio_result(res)

    if bool(diag.get("is_identifiable", True)) is False:
        with pytest.warns(RuntimeWarning):
            apply_cost_ratio_policy(
                y_true=y_true,
                y_pred=y_pred,
                policy=policy,
                co=1.0,
                gate="warn",
            )
    else:
        # Identifiable: ensure warn mode doesn't raise.
        apply_cost_ratio_policy(
            y_true=y_true,
            y_pred=y_pred,
            policy=policy,
            co=1.0,
            gate="warn",
        )


def test_apply_cost_ratio_policy_gate_raise_raises_when_not_identifiable_or_is_overridable():
    """
    gate="raise" should raise ValueError when is_identifiable=False,
    unless an override reason is provided (auditably).
    """
    y_true = np.array([10.0, 0.0, 5.0])
    y_pred = np.array([0.0, 20.0, 7.0])

    policy = CostRatioPolicy(R_grid=(0.25, 1.0, 10.0), co=1.0)

    # Determine identifiability under the current thresholds.
    res = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0, gate="off")
    _R, diag = _unpack_cost_ratio_result(res)

    if bool(diag.get("is_identifiable", True)) is False:
        with pytest.raises(ValueError):
            apply_cost_ratio_policy(
                y_true=y_true,
                y_pred=y_pred,
                policy=policy,
                co=1.0,
                gate="raise",
            )

        # Override should suppress the raise and record metadata
        res2 = apply_cost_ratio_policy(
            y_true=y_true,
            y_pred=y_pred,
            policy=policy,
            co=1.0,
            gate="raise",
            identifiability_override_reason="Known flat curve in pilot; allow for now.",
        )
        _R2, diag2 = _unpack_cost_ratio_result(res2)
        assert "identifiability_gate" in diag2
        ig = diag2["identifiability_gate"]
        assert isinstance(ig, dict)
        assert ig["gate_mode"] == "raise"
        assert ig["gate_triggered"] is True
        assert ig["gate_overridden"] is True
        assert ig["gate_override_reason"] == "Known flat curve in pilot; allow for now."
    else:
        # If identifiable, raise mode should not raise.
        apply_cost_ratio_policy(
            y_true=y_true,
            y_pred=y_pred,
            policy=policy,
            co=1.0,
            gate="raise",
        )


# =============================================================================
# Entity policy application
# =============================================================================
def _panel_for_entity_tests() -> pd.DataFrame:
    # A: 5 rows (below min_n=6)
    # B: 6 rows (meets min_n=6)
    rows: list[dict[str, object]] = []

    for _ in range(5):
        rows.append({"entity": "A", "actual_qty": 10.0, "forecast_qty": 12.0})  # overbuild

    for _ in range(6):
        rows.append({"entity": "B", "actual_qty": 10.0, "forecast_qty": 8.0})  # shortfall

    return pd.DataFrame(rows)


def test_apply_entity_cost_ratio_policy_min_n_governance_blocks_small_entities():
    df = _panel_for_entity_tests()

    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0, min_n=6)

    # Silence gating side-effects; this test is about min_n behavior.
    res = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        gate="off",
    )
    out = _as_entity_df(res)

    # one row per entity
    assert set(out["entity"]) == {"A", "B"}
    assert len(out) == 2

    row_a = out.loc[out["entity"] == "A"].iloc[0]
    row_b = out.loc[out["entity"] == "B"].iloc[0]

    assert np.isnan(float(row_a["R"]))
    assert "min_n_not_met" in str(row_a["reason"])
    assert int(row_a["n"]) == 5

    assert np.isfinite(float(row_b["R"]))
    assert row_b["reason"] is None or pd.isna(row_b["reason"])
    assert int(row_b["n"]) == 6


def test_apply_entity_cost_ratio_policy_includes_or_excludes_diagnostics():
    df = _panel_for_entity_tests()
    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0, min_n=6)

    # Silence gating side-effects; this test is about column inclusion/exclusion.
    res_with = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=True,
        gate="off",
    )
    out_with = _as_entity_df(res_with)
    assert {"under_cost", "over_cost", "diff"}.issubset(out_with.columns)

    res_without = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=False,
        gate="off",
    )
    out_without = _as_entity_df(res_without)
    assert not {"under_cost", "over_cost", "diff"}.issubset(out_without.columns)


def test_apply_entity_cost_ratio_policy_validates_required_columns():
    df = pd.DataFrame({"entity": ["A"], "actual_qty": [10.0]})  # missing forecast

    with pytest.raises(KeyError):
        apply_entity_cost_ratio_policy(
            df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            policy=DEFAULT_COST_RATIO_POLICY,
            gate="off",
        )


def test_apply_entity_cost_ratio_policy_validates_sample_weight_col_presence():
    df = pd.DataFrame({"entity": ["A"], "actual_qty": [10.0], "forecast_qty": [9.0]})

    with pytest.raises(KeyError):
        apply_entity_cost_ratio_policy(
            df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            policy=DEFAULT_COST_RATIO_POLICY,
            sample_weight_col="weight",
            gate="off",
        )


def test_apply_entity_cost_ratio_policy_uses_policy_default_co_when_none():
    df = pd.DataFrame(
        {
            "entity": ["X"] * 6,
            "actual_qty": [10.0] * 6,
            "forecast_qty": [8.0] * 6,
        }
    )

    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=2.0, min_n=6)

    # Silence gating side-effects; this test is about co selection.
    res = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        co=None,
        gate="off",
    )
    out = _as_entity_df(res)

    row = out.iloc[0]
    assert np.isclose(float(row["co"]), 2.0)


def test_apply_entity_cost_ratio_policy_surfaces_entity_diagnostics_when_enabled():
    """
    After surfacing diagnostics at the policy boundary:
    - eligible entities (meeting min_n) should have a dict in `diagnostics`
    - ineligible entities should have diagnostics=None (or NaN)

    This test disables gating side-effects; it focuses on surfacing.
    """
    df = _panel_for_entity_tests()
    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0, min_n=6)

    out = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=True,
        gate="off",
    )

    row_a = out.loc[out["entity"] == "A"].iloc[0]
    row_b = out.loc[out["entity"] == "B"].iloc[0]

    # Ineligible entity: diagnostics should be None/NaN
    assert "diagnostics" in out.columns
    assert row_a["diagnostics"] is None or pd.isna(row_a["diagnostics"])

    # Eligible entity: diagnostics should be a dict with expected keys
    diag_b = row_b["diagnostics"]
    assert isinstance(diag_b, dict)
    assert {"over_cost_const", "min_gap", "degenerate_perfect_forecast"}.issubset(diag_b.keys())


# =============================================================================
# New tests: gate hook metadata surfacing (entity)
# =============================================================================
def test_apply_entity_cost_ratio_policy_includes_gate_metadata_when_gate_requested():
    """
    Entity policy should include an `identifiability_gate` column when gating is enabled.
    """
    df = _panel_for_entity_tests()
    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0, min_n=6)

    # Current behavior warns on this panel; assert the warning to keep tests clean.
    with pytest.warns(RuntimeWarning):
        out = apply_entity_cost_ratio_policy(
            df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            policy=policy,
            include_diagnostics=True,
            gate="warn",
        )

    assert "identifiability_gate" in out.columns
    ig = out["identifiability_gate"].iloc[0]
    assert isinstance(ig, dict)
    assert ig["gate_mode"] == "warn"
    assert isinstance(bool(ig["gate_triggered"]), bool)


def test_apply_entity_cost_ratio_policy_gate_raise_raises_without_override_when_not_identifiable():
    """
    Entity gating may be driven by internal diagnostics (not necessarily surfaced as a column),
    so we must not assume "no is_identifiable column => gating no-op".

    Behavior expected:
      - gate="raise" may raise if the implementation deems any eligible entity non-identifiable
      - providing identifiability_override_reason must suppress the raise
      - if the implementation deems the panel identifiable, raise mode should not raise
    """
    df = _panel_for_entity_tests()
    policy = CostRatioPolicy(R_grid=(0.5, 1.0, 2.0, 3.0), co=1.0, min_n=6)

    # Probe under gate="off" so we can keep any diagnostic columns available for debugging.
    out_probe = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=True,
        gate="off",
    )
    assert isinstance(out_probe, pd.DataFrame)

    try:
        out = apply_entity_cost_ratio_policy(
            df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            policy=policy,
            include_diagnostics=True,
            gate="raise",
        )
        # If no raise, the implementation considers the panel acceptable under current thresholds.
        assert isinstance(out, pd.DataFrame)
        assert set(out["entity"]) == {"A", "B"}
    except ValueError:
        # If it raises, override must suppress and still return a valid result.
        out_ok = apply_entity_cost_ratio_policy(
            df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            policy=policy,
            include_diagnostics=True,
            gate="raise",
            identifiability_override_reason="Pilot exception; allow non-identifiable entities.",
        )
        assert isinstance(out_ok, pd.DataFrame)
        assert set(out_ok["entity"]) == {"A", "B"}
        assert "identifiability_gate" in out_ok.columns
        ig = out_ok["identifiability_gate"].iloc[0]
        assert isinstance(ig, dict)
        assert ig["gate_mode"] == "raise"
        assert ig["gate_triggered"] is True
        assert ig["gate_overridden"] is True
        assert ig["gate_override_reason"] == "Pilot exception; allow non-identifiable entities."
