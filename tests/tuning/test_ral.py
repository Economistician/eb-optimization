import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl
from eb_optimization.tuning.ral import tune_ral_policy


def _make_global_df() -> pd.DataFrame:
    """Create a simple dataset with systematic underforecast bias."""
    n = 20
    rng = np.random.default_rng(0)
    actual = rng.integers(80, 120, size=n)
    forecast = (actual * 0.8).astype(float)  # biased low
    return pd.DataFrame({"actual": actual, "forecast": forecast})


def test_tune_ral_policy_global_uplift():
    df = _make_global_df()
    cu, co = 2.0, 1.0

    uplift_min = 1.0
    uplift_max = 1.15
    grid_step = 0.01

    policy, diagnostics = tune_ral_policy(
        df,
        forecast_col="forecast",
        actual_col="actual",
        cu=cu,
        co=co,
        uplift_min=uplift_min,
        uplift_max=uplift_max,
        grid_step=grid_step,
    )

    # Basic contract checks
    assert policy is not None
    assert diagnostics is not None
    assert isinstance(diagnostics, pd.DataFrame)
    assert not diagnostics.empty
    assert "uplift" in diagnostics.columns

    # Policy shape checks (global-only)
    assert policy.global_uplift >= uplift_min
    assert policy.global_uplift <= uplift_max
    assert policy.segment_cols == []  # global-only in current impl

    # Behavioral check: applying uplift should not worsen CWSL
    y_true = df["actual"].to_numpy(dtype=float)
    y_pred = df["forecast"].to_numpy(dtype=float)

    original_cwsl = cwsl(y_true, y_pred, cu=cu, co=co)
    y_pred_adj = policy.adjust_forecast(df, forecast_col="forecast").to_numpy(
        dtype=float
    )
    adjusted_cwsl = cwsl(y_true, y_pred_adj, cu=cu, co=co)

    assert adjusted_cwsl <= original_cwsl


def test_tune_ral_policy_with_diagnostics():
    df = _make_global_df()
    cu, co = 2.0, 1.0

    policy, diagnostics = tune_ral_policy(
        df, forecast_col="forecast", actual_col="actual", cu=cu, co=co
    )

    assert policy is not None
    assert diagnostics is not None
    assert isinstance(diagnostics, pd.DataFrame)
    assert len(diagnostics) == 1  # one row for global-only

    # Expected audit columns
    for col in ("uplift", "diff", "under_cost", "over_cost"):
        assert col in diagnostics.columns
