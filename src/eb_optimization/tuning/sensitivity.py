r"""
CWSL cost-ratio sensitivity utilities.

This module provides helpers for computing a *sensitivity curve* of
Cost-Weighted Service Loss (CWSL) across a grid of cost ratios:

$$
R = \frac{c_u}{c_o}
$$

Given an overbuild cost coefficient $c_o$ and ratio $R$, the implied underbuild cost is:

$$
c_u = R \cdot c_o
$$

Why this lives in eb-optimization
--------------------------------
Computing a metric across a candidate grid of hyperparameters (like a cost ratio R)
is an *analysis / calibration workflow* rather than a metric primitive.

- **eb-metrics** remains the source of truth for *metric math* (e.g., ``cwsl``).
- **eb-optimization** owns grid-based evaluation, diagnostics, and tuning utilities.

This module therefore contains:
- ``cwsl_sensitivity``: array-level sweep (grid evaluation)
- ``compute_cwsl_sensitivity_df``: DataFrame-oriented wrapper (tidy long-form output)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from eb_metrics.metrics.loss import cwsl

__all__ = ["compute_cwsl_sensitivity_df", "cwsl_sensitivity"]


def _as_1d_float_array(x: Sequence[float] | np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    return np.asarray(list(x), dtype=float).reshape(-1)


def _normalize_R_list(
    R_list: Sequence[float] | np.ndarray | Iterable[float],
) -> np.ndarray:
    """
    Normalize and validate a candidate R grid.

    Backward-compatible behavior:
    - Non-finite values are dropped.
    - Non-positive values (R <= 0) are dropped.
    - If no valid values remain, raises ValueError.
    - De-duplicates and sorts for stable outputs.

    Parameters
    ----------
    R_list
        Candidate ratios to evaluate.

    Returns
    -------
    numpy.ndarray
        1D array of finite, strictly positive ratios.

    Raises
    ------
    ValueError
        If the candidate list is empty or contains no valid ratios after filtering.
    """
    R_arr = _as_1d_float_array(R_list)
    if R_arr.ndim != 1 or R_arr.size == 0:
        raise ValueError("R_list must be a non-empty 1D sequence of floats.")

    R_arr = R_arr[np.isfinite(R_arr)]
    R_arr = R_arr[R_arr > 0]

    if R_arr.size == 0:
        raise ValueError(
            "R_list contains no valid ratios after filtering. Provide at least one R > 0."
        )

    return np.unique(R_arr)


def cwsl_sensitivity(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    *,
    R_list: Sequence[float] | np.ndarray | Iterable[float] = (0.5, 1.0, 2.0, 3.0),
    co: float | np.ndarray = 1.0,
    sample_weight: np.ndarray | Sequence[float] | None = None,
) -> dict[float, float]:
    r"""
    Evaluate CWSL across a grid of cost ratios (cost sensitivity analysis).

    For each candidate ratio:

    $$ R = \frac{c_u}{c_o} $$

    holding ``co`` fixed and setting:

    $$ c_u = R \cdot c_o $$

    Parameters
    ----------
    y_true
        Realized demand values (non-negative).
    y_pred
        Forecast values (non-negative).
    R_list
        Candidate cost ratios to evaluate. Non-finite and non-positive values are ignored.
    co
        Overbuild cost coefficient. Can be scalar or per-interval array.
    sample_weight
        Optional non-negative weights per interval.

    Returns
    -------
    dict[float, float]
        Mapping ``{R: cwsl_value}`` for each valid ``R``.

    Raises
    ------
    ValueError
        If no valid ratios remain after filtering, if inputs are invalid, or if
        sample_weight contains negatives.
    """
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )
    if np.any(y_true_arr < 0) or np.any(y_pred_arr < 0):
        raise ValueError("y_true and y_pred must be non-negative.")

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w.shape != y_true_arr.shape:
            raise ValueError(
                f"sample_weight must have shape {y_true_arr.shape}; got {w.shape}"
            )
        if np.any(w < 0):
            raise ValueError("sample_weight must be non-negative.")
    else:
        w = None

    # normalize candidate grid
    R_arr = _normalize_R_list(R_list)

    # validate co
    if isinstance(co, np.ndarray):
        co_arr = np.asarray(co, dtype=float).reshape(-1)
        if co_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"co must have shape {y_true_arr.shape}; got {co_arr.shape}"
            )
        if np.any(co_arr <= 0):
            raise ValueError("co must be strictly positive.")
        co_val: float | np.ndarray = co_arr
    else:
        co_float = float(co)
        if co_float <= 0:
            raise ValueError("co must be strictly positive.")
        co_val = co_float

    results: dict[float, float] = {}

    for R in R_arr:
        # cu = R * co (supports scalar or per-interval array)
        cu_val = float(R) * co_val  # type: ignore[operator]

        value = cwsl(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            cu=cu_val,
            co=co_val,
            sample_weight=w,
        )
        results[float(R)] = float(value)

    # R_arr is guaranteed non-empty; results should be non-empty
    return results


def compute_cwsl_sensitivity_df(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float | str = 1.0,
    group_cols: Sequence[str] | None = None,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    r"""
    Compute CWSL sensitivity curves from a DataFrame.

    Evaluates CWSL over a grid of cost ratios:

    $$ R = \frac{c_u}{c_o} $$

    For each ratio value $R$ in ``R_list``, the implied underbuild cost is:

    $$ c_u = R \cdot c_o $$

    where ``co`` may be a scalar (global) or a per-row column name.

    Parameters
    ----------
    df
        Input data containing actuals, forecasts, optional groups, and optional weights.
    actual_col
        Column containing realized demand values.
    forecast_col
        Column containing forecast values.
    R_list
        Candidate ratios to evaluate. Non-finite and non-positive values are ignored.
    co
        Overbuild cost specification:
        - If ``float``: constant $c_o$ applied to all rows and groups.
        - If ``str``: name of a column in ``df`` containing per-row $c_o(i)$ values.
    group_cols
        Optional grouping columns. If ``None`` or empty, the entire DataFrame is treated
        as a single group.
    sample_weight_col
        Optional column name containing non-negative sample weights per row.

    Returns
    -------
    pandas.DataFrame
        Long-form table of sensitivity results with columns:
        - if not grouped: ``["R", "CWSL"]``
        - if grouped: ``group_cols + ["R", "CWSL"]``

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If no valid ratios remain after filtering, or if sample weights are negative.
    """
    gcols = [] if group_cols is None else list(group_cols)

    # ---- validation: columns ----
    required_cols: list[str] = [actual_col, forecast_col]
    if isinstance(co, str):
        required_cols.append(co)
    if sample_weight_col is not None:
        required_cols.append(sample_weight_col)
    if gcols:
        required_cols.extend(gcols)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    # ---- validation: R grid (for early failure and stable ordering) ----
    R_arr = _normalize_R_list(R_list)

    # ---- validation: weights ----
    if sample_weight_col is not None:
        w_all = df[sample_weight_col].to_numpy(dtype=float)
        if np.any(w_all < 0):
            raise ValueError("sample weights must be non-negative.")

    # ---- compute ----
    results: list[dict[str, Any]] = []

    if len(gcols) == 0:
        iter_groups = [((None,), df)]
    else:
        iter_groups = df.groupby(gcols, dropna=False, sort=False)

    for keys, g in iter_groups:
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        co_value: float | np.ndarray
        co_value = g[co].to_numpy(dtype=float) if isinstance(co, str) else float(co)

        sample_weight = (
            g[sample_weight_col].to_numpy(dtype=float)
            if sample_weight_col is not None
            else None
        )

        sensitivity_map = cwsl_sensitivity(
            y_true=y_true,
            y_pred=y_pred,
            R_list=R_arr,
            co=co_value,
            sample_weight=sample_weight,
        )

        for R_val, cwsl_val in sensitivity_map.items():
            row: dict[str, Any] = {"R": float(R_val), "CWSL": float(cwsl_val)}
            for col, value in zip(gcols, keys, strict=False):
                row[col] = value
            results.append(row)

    result_df = pd.DataFrame(results)
    result_df = (
        result_df[[*gcols, "R", "CWSL"]] if len(gcols) > 0 else result_df[["R", "CWSL"]]
    )
    return result_df
