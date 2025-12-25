from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_metrics.metrics import cwsl_sensitivity
from eb_optimization.tuning.sensitivity import compute_cwsl_sensitivity_df


def test_compute_cwsl_sensitivity_df_matches_core_function_scalar_co():
    """
    Basic correctness:

    Ensure compute_cwsl_sensitivity_df returns the same CWSL values
    as eb_metrics.metrics.cwsl_sensitivity when using scalar co.
    """
    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0, 8.0],
            "forecast": [9.0, 15.0, 7.0],
        }
    )

    y_true = df["actual"].to_numpy()
    y_pred = df["forecast"].to_numpy()
    R_list = [0.5, 1.0, 2.0, 3.0]
    co = 1.0

    core = cwsl_sensitivity(
        y_true=y_true,
        y_pred=y_pred,
        R_list=R_list,
        co=co,
        sample_weight=None,
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        actual_col="actual",
        forecast_col="forecast",
        R_list=R_list,
        co=co,
        sample_weight_col=None,
    )

    # Same R keys (order should match core mapping iteration)
    assert list(out["R"].astype(float)) == [float(r) for r in core.keys()]

    for r, cwsl_val in core.items():
        row_val = float(out.loc[out["R"] == float(r), "CWSL"].iloc[0])
        assert np.isclose(row_val, float(cwsl_val))


def test_compute_cwsl_sensitivity_df_supports_per_row_co_and_weights():
    """
    Structural + behavioral:

    - Supports co as a column name.
    - Supports sample_weight via column.
    - Matches cwsl_sensitivity when passed the same co array and weights.
    """
    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0, 8.0],
            "forecast": [9.0, 15.0, 7.0],
            "co_col": [1.0, 2.0, 1.5],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    y_true = df["actual"].to_numpy()
    y_pred = df["forecast"].to_numpy()
    co_arr = df["co_col"].to_numpy()
    w = df["weight"].to_numpy()

    R_list = [0.5, 1.0, 2.0]

    core = cwsl_sensitivity(
        y_true=y_true,
        y_pred=y_pred,
        R_list=R_list,
        co=co_arr,
        sample_weight=w,
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        actual_col="actual",
        forecast_col="forecast",
        R_list=R_list,
        co="co_col",
        sample_weight_col="weight",
    )

    assert set(out["R"].astype(float)) == {float(r) for r in core.keys()}

    for r, cwsl_val in core.items():
        row_val = float(out.loc[out["R"] == float(r), "CWSL"].iloc[0])
        assert np.isclose(row_val, float(cwsl_val))


def test_compute_cwsl_sensitivity_df_handles_group_cols():
    """
    Ensure grouping produces one row per (group, R) pair and matches core per group.
    """
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "actual": [10.0, 12.0, 10.0, 12.0],
            "forecast": [9.0, 15.0, 11.0, 10.0],
        }
    )

    R_list = [1.0, 2.0]
    out = compute_cwsl_sensitivity_df(
        df=df,
        actual_col="actual",
        forecast_col="forecast",
        R_list=R_list,
        co=1.0,
        group_cols=["grp"],
    )

    # 2 groups x 2 R values = 4 rows
    assert len(out) == 4
    assert set(out["grp"]) == {"A", "B"}
    assert set(out["R"].astype(float)) == {1.0, 2.0}

    # Match core per group
    for grp, g in df.groupby("grp", dropna=False, sort=False):
        core = cwsl_sensitivity(
            y_true=g["actual"].to_numpy(),
            y_pred=g["forecast"].to_numpy(),
            R_list=R_list,
            co=1.0,
            sample_weight=None,
        )
        for r, cwsl_val in core.items():
            got = float(out.loc[(out["grp"] == grp) & (out["R"] == float(r)), "CWSL"].iloc[0])
            assert np.isclose(got, float(cwsl_val))


def test_compute_cwsl_sensitivity_df_filters_non_positive_R_and_raises_if_none_valid():
    """
    Backward-compatible behavior:
    - non-positive R values are ignored
    - if none remain, raise ValueError
    """
    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0],
            "forecast": [9.0, 11.0],
        }
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        actual_col="actual",
        forecast_col="forecast",
        R_list=[-1.0, 0.0, 1.0, 2.0],
        co=1.0,
    )
    assert set(out["R"].astype(float)) == {1.0, 2.0}

    with pytest.raises(ValueError):
        compute_cwsl_sensitivity_df(
            df=df,
            actual_col="actual",
            forecast_col="forecast",
            R_list=[-2.0, 0.0],
            co=1.0,
        )