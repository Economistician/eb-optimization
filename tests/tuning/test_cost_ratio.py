from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_optimization.tuning.cost_ratio import (
    estimate_R_cost_balance,
    estimate_entity_R_from_balance,
)


# =============================================================================
# Global R calibration tests (lifted from eb-metrics)
# =============================================================================
def test_estimate_R_cost_balance_perfect_forecast_prefers_R_near_1():
    """
    If there is no error at all, the estimator should return the grid value
    closest to 1.0.
    """
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]

    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert R == 1.0


def test_estimate_R_cost_balance_balanced_example_has_known_optimum():
    """
    Construct a toy example where the cost-balance solution is known.

    shortfall_sum = 10
    overbuild_sum = 20

    Under-cost(R) = R * 10
    Over-cost     = 20

    |Under - Over| is minimized at R* = 2.0, which is in the grid.
    """
    # Interval 1: purely shortfall of 10
    # Interval 2: purely overbuild of 20
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([0.0, 20.0])

    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert np.isclose(R, 2.0)


def test_estimate_R_cost_balance_respects_sample_weights():
    """
    Check that sample_weight is accepted and doesn't crash.
    We don't assert a specific R here, just that it runs and returns
    a valid scalar from the grid.
    """
    y_true = [10, 0, 5]
    y_pred = [0, 20, 5]
    w = [1.0, 2.0, 0.5]

    R_grid = (0.5, 1.0, 2.0, 3.0)
    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=R_grid,
        co=1.0,
        sample_weight=w,
    )

    assert R in R_grid


def test_estimate_R_cost_balance_raises_on_invalid_R_grid():
    """
    R_grid with no positive entries should raise ValueError.
    """
    y_true = [10, 20]
    y_pred = [8, 22]

    with pytest.raises(ValueError):
        estimate_R_cost_balance(
            y_true=y_true,
            y_pred=y_pred,
            R_grid=(0.0, -1.0),
            co=1.0,
        )


# =============================================================================
# Entity-level R calibration tests (existing eb-optimization tests)
# =============================================================================
def _build_simple_panel() -> pd.DataFrame:
    """
    Construct a small panel with two entities:

    - Entity A: mostly overbuild (y_pred > y_true)
    - Entity B: mostly shortfall (y_pred < y_true)

    Used to sanity-check that a shortfall-heavy entity prefers an equal or larger R.
    """
    rows: list[dict[str, object]] = []

    # Entity A: overbuild-dominant
    for y_true, y_pred in [(10, 12), (12, 15), (15, 18)]:
        rows.append({"entity": "A", "actual_qty": y_true, "forecast_qty": y_pred})

    # Entity B: shortfall-dominant
    for y_true, y_pred in [(10, 8), (12, 9), (15, 13)]:
        rows.append({"entity": "B", "actual_qty": y_true, "forecast_qty": y_pred})

    return pd.DataFrame(rows)


def test_estimate_entity_R_basic_structure():
    """
    Structural test:
    - one row per entity
    - expected columns present
    - R, cu, co are finite and positive
    """
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert set(result["entity"]) == {"A", "B"}
    assert len(result) == 2

    expected_cols = {"entity", "R", "cu", "co", "under_cost", "over_cost", "diff"}
    assert expected_cols.issubset(result.columns)

    assert np.all(np.isfinite(result["R"]))
    assert np.all(result["R"] > 0)
    assert np.all(result["cu"] > 0)
    assert np.all(result["co"] > 0)


def test_estimate_entity_R_behavior_shortfall_vs_overbuild():
    """
    Behavioral test:
    On a symmetric-ish grid of R values, the shortfall-heavy entity
    should prefer an equal or higher R than the overbuild-heavy entity.
    """
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 4.0),
        co=1.0,
    )

    R_A = float(result.loc[result["entity"] == "A", "R"].iloc[0])
    R_B = float(result.loc[result["entity"] == "B", "R"].iloc[0])

    assert R_B >= R_A


def test_estimate_entity_R_zero_error_picks_R_near_one():
    """
    Degenerate case: entity with no error (y_true == y_pred) everywhere.

    We expect:
    - chosen R to be the grid value closest to 1.0
    - under_cost == over_cost == 0
    - diff == 0
    """
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "actual_qty": [10.0, 12.0, 5.0, 7.0],
            "forecast_qty": [10.0, 12.0, 5.0, 7.0],
        }
    )

    ratios = (0.4, 0.9, 1.3, 3.0)
    expected_R = 0.9  # closest to 1.0 in this grid

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=2.0,
    )

    for _, row in result.iterrows():
        assert np.isclose(row["R"], expected_R)
        assert np.isclose(row["cu"], expected_R * 2.0)
        assert np.isclose(row["under_cost"], 0.0)
        assert np.isclose(row["over_cost"], 0.0)
        assert np.isclose(row["diff"], 0.0)


def test_estimate_entity_R_respects_sample_weights():
    """
    Check that sample weights are accepted and do not raise, and that
    changing weights can change the chosen R.

    We construct one entity with one shortfall and one overbuild
    of equal magnitude. Under the cost-balance rule, overweighting
    the shortfall interval pushes the chosen R downward on a simple grid,
    because we reduce the per-unit shortfall cost needed to balance totals.
    """
    df = pd.DataFrame(
        {
            "entity": ["X", "X"],
            "actual_qty": [10.0, 10.0],
            "forecast_qty": [8.0, 12.0],  # shortfall=2, overbuild=2
            "w_balanced": [1.0, 1.0],
            "w_shortfall_heavy": [3.0, 1.0],
        }
    )

    ratios = (0.5, 1.0, 2.0, 4.0)

    res_balanced = estimate_entity_R_from_balance(
        df=df.rename(columns={"w_balanced": "weight"}),
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        sample_weight_col="weight",
    )

    res_shortfall_heavy = estimate_entity_R_from_balance(
        df=df.rename(columns={"w_shortfall_heavy": "weight"}),
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        sample_weight_col="weight",
    )

    R_balanced = float(res_balanced.loc[res_balanced["entity"] == "X", "R"].iloc[0])
    R_shortfall_heavy = float(
        res_shortfall_heavy.loc[res_shortfall_heavy["entity"] == "X", "R"].iloc[0]
    )

    assert R_shortfall_heavy <= R_balanced


def test_estimate_entity_R_invalid_inputs():
    """
    Basic validation tests:
    - missing required columns raises KeyError
    - non-positive ratios raise ValueError
    - non-positive co raises ValueError
    """
    df = pd.DataFrame(
        {
            "entity": ["A"],
            "actual_qty": [10.0],
            "forecast_qty": [9.0],
        }
    )

    with pytest.raises(KeyError):
        estimate_entity_R_from_balance(
            df=df.drop(columns=["entity"]),
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
        )

    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.0, -1.0),
        )

    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.5, 1.0),
            co=0.0,
        )