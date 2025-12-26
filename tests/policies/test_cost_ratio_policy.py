from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_optimization.policies.cost_ratio_policy import (
    CostRatioPolicy,
    DEFAULT_COST_RATIO_POLICY,
    apply_cost_ratio_policy,
    apply_entity_cost_ratio_policy,
)


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

    R, diag = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=None)

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

    R, _ = apply_cost_ratio_policy(y_true=y_true, y_pred=y_pred, policy=policy, co=1.0)
    assert np.isclose(R, 2.0)


def test_apply_cost_ratio_policy_returns_nan_safe_diag_when_co_is_array():
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([0.0, 20.0])
    co = np.array([1.0, 1.0])

    R, diag = apply_cost_ratio_policy(
        y_true=y_true,
        y_pred=y_pred,
        policy=DEFAULT_COST_RATIO_POLICY,
        co=co,
    )

    assert np.isfinite(R)
    assert diag["co_is_array"] is True


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

    out = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
    )

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

    out_with = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=True,
    )
    assert {"under_cost", "over_cost", "diff"}.issubset(out_with.columns)

    out_without = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        include_diagnostics=False,
    )
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

    out = apply_entity_cost_ratio_policy(
        df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        policy=policy,
        co=None,
    )

    row = out.iloc[0]
    assert np.isclose(float(row["co"]), 2.0)