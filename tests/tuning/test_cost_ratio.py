from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_optimization.tuning.cost_ratio import (
    EntityCostRatioEstimate,
    estimate_R_cost_balance,
    estimate_entity_R_from_balance,
)


# =============================================================================
# Global R calibration tests
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
# Entity-level R calibration tests
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
    - ratios with no positive candidates raise ValueError
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


# =============================================================================
# New tests for entity-level artifact mode (return_result=True)
# =============================================================================
def test_estimate_entity_R_return_result_artifact_structure_and_curves():
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 4.0),
        co=1.0,
        return_result=True,
        selection="curve",
    )

    assert isinstance(result, EntityCostRatioEstimate)
    assert result.entity_col == "entity"
    assert result.method == "cost_balance"
    assert result.tie_break == "first"
    assert result.selection == "curve"

    # Table: one row per entity, no curve DataFrames embedded in cells
    assert isinstance(result.table, pd.DataFrame)
    assert set(result.table["entity"]) == {"A", "B"}
    expected_cols = {"entity", "R_star", "n", "under_cost", "over_cost", "gap", "diagnostics"}
    assert expected_cols.issubset(result.table.columns)

    # Curves: dict mapping entity -> DataFrame with required columns
    assert set(result.curves.keys()) == {"A", "B"}
    for ent, curve in result.curves.items():
        assert isinstance(curve, pd.DataFrame)
        assert {"R", "under_cost", "over_cost", "gap"}.issubset(curve.columns)
        # grid order preserved
        assert np.allclose(curve["R"].to_numpy(dtype=float), result.grid.astype(float))


def test_estimate_entity_R_return_result_matches_legacy_R():
    """
    Ensure artifact mode selects the same R as legacy mode for the same inputs
    (with selection='curve').
    """
    df = _build_simple_panel()
    ratios = (0.5, 1.0, 2.0, 4.0)

    legacy = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
    ).set_index("entity")

    art = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        return_result=True,
        selection="curve",
    )

    art_table = art.table.set_index("entity")

    for ent in ["A", "B"]:
        assert np.isclose(float(legacy.loc[ent, "R"]), float(art_table.loc[ent, "R_star"]))
        assert np.isclose(float(legacy.loc[ent, "diff"]), float(art_table.loc[ent, "gap"]))


def test_estimate_entity_R_artifact_degenerate_entity_has_zero_curve_and_min_gap():
    """
    In artifact mode, degenerate entities (perfect forecasts) should:
    - choose R closest to 1.0
    - have a curve with all zeros (under_cost, over_cost, gap)
    - have diagnostics indicating degenerate_perfect_forecast=True
    """
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "actual_qty": [10.0, 12.0, 5.0, 7.0],
            "forecast_qty": [10.0, 12.0, 5.0, 7.0],
        }
    )
    ratios = (0.4, 0.9, 1.3, 3.0)
    expected_R = 0.9

    res = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=2.0,
        return_result=True,
        selection="curve",
    )

    table = res.table.set_index("entity")
    for ent in ["A", "B"]:
        assert np.isclose(float(table.loc[ent, "R_star"]), expected_R)
        assert np.isclose(float(table.loc[ent, "gap"]), 0.0)
        diag = table.loc[ent, "diagnostics"]
        assert isinstance(diag, dict)
        assert diag["degenerate_perfect_forecast"] is True

        curve = res.curves[ent]
        assert np.allclose(curve["under_cost"].to_numpy(dtype=float), 0.0)
        assert np.allclose(curve["over_cost"].to_numpy(dtype=float), 0.0)
        assert np.allclose(curve["gap"].to_numpy(dtype=float), 0.0)


def test_estimate_entity_R_filters_non_positive_ratios_in_artifact_mode_grid():
    """
    New logic: ratios are filtered to strictly positive candidates (order preserved).
    This test ensures the returned artifact grid excludes non-positive values and
    that each curve uses the same filtered grid.
    """
    df = _build_simple_panel()
    ratios = (-1.0, 0.0, 0.5, 1.0, 2.0)

    res = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        return_result=True,
        selection="curve",
    )

    assert np.allclose(res.grid, np.asarray([0.5, 1.0, 2.0], dtype=float))
    for curve in res.curves.values():
        assert np.allclose(curve["R"].to_numpy(dtype=float), res.grid.astype(float))


def test_estimate_entity_R_selection_kernel_matches_curve_selection():
    """
    New logic: entity-level selection supports 'kernel' and should match 'curve'
    for deterministic tie-break='first' scoring based on curve gaps.
    """
    df = _build_simple_panel()
    ratios = (0.5, 1.0, 2.0, 4.0)

    res_curve = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        return_result=True,
        selection="curve",
    )
    res_kernel = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        return_result=True,
        selection="kernel",
    )

    t_curve = res_curve.table.set_index("entity")
    t_kernel = res_kernel.table.set_index("entity")

    for ent in ["A", "B"]:
        assert np.isclose(float(t_curve.loc[ent, "R_star"]), float(t_kernel.loc[ent, "R_star"]))
        assert np.isclose(float(t_curve.loc[ent, "gap"]), float(t_kernel.loc[ent, "gap"]))