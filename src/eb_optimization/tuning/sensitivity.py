from __future__ import annotations

r"""
CWSL cost-ratio sensitivity utilities.

This module provides DataFrame-oriented helpers for computing a *sensitivity curve*
of Cost-Weighted Service Loss (CWSL) across a grid of cost ratios:

$$
R = \frac{c_u}{c_o}
$$

Given an overbuild cost coefficient $c_o$ and ratio $R$, the implied underbuild cost is:

$$
c_u = R \cdot c_o
$$

The primary helper in this module evaluates :func:`eb_metrics.metrics.cwsl_sensitivity`
over a ratio grid, optionally per group, returning a tidy long-form DataFrame suitable for:

- diagnostic plots,
- governance checks,
- hyperparameter selection workflows.

Design intent
-------------
- This code lives in **eb-optimization** because it performs grid-based evaluation over a
  hyperparameter (R) and is frequently used as part of tuning / model selection.
- Metric math remains a single source of truth in **eb-metrics**.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl_sensitivity

__all__ = ["compute_cwsl_sensitivity_df"]


def _as_1d_float_array(x: Sequence[float] | np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    return np.asarray(list(x), dtype=float).reshape(-1)


def _normalize_R_list(R_list: Sequence[float]) -> np.ndarray:
    """
    Normalize and validate a candidate R grid.

    Behavior is intentionally backward-compatible:

    - Non-finite values are dropped.
    - Non-positive values (R <= 0) are dropped.
    - If no valid values remain, raises ValueError.

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
        If `R_list` is empty / not 1D, or if no valid ratios remain after filtering.
    """
    R_arr = _as_1d_float_array(R_list)
    if R_arr.ndim != 1 or R_arr.size == 0:
        raise ValueError("R_list must be a non-empty 1D sequence of floats.")

    # Filter non-finite and non-positive values (backward-compatible behavior)
    R_arr = R_arr[np.isfinite(R_arr)]
    R_arr = R_arr[R_arr > 0]

    if R_arr.size == 0:
        raise ValueError(
            "R_list contains no valid ratios after filtering. Provide at least one R > 0."
        )

    # De-dup and sort for stable outputs
    R_arr = np.unique(R_arr)
    return R_arr


def compute_cwsl_sensitivity_df(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, str] = 1.0,
    group_cols: Optional[Sequence[str]] = None,
    sample_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    Compute CWSL sensitivity curves from a DataFrame.

    This is a DataFrame-level wrapper around :func:`eb_metrics.metrics.cwsl_sensitivity`.
    It evaluates CWSL over a grid of cost ratios:

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
        Candidate ratios to evaluate.

        Backward-compatible behavior:
        - non-finite values are ignored
        - non-positive values (R <= 0) are ignored
        - if no valid values remain, a ValueError is raised
    co
        Overbuild cost specification.

        - If ``float``: constant $c_o$ applied to all rows and groups.
        - If ``str``: name of a column in ``df`` containing per-row $c_o(i)$ values.
    group_cols
        Optional grouping columns. If ``None`` or empty, the entire DataFrame is treated
        as a single group.
    sample_weight_col
        Optional column name containing non-negative sample weights per row. If provided,
        weights are passed as ``sample_weight`` to the underlying sensitivity computation.

    Returns
    -------
    pandas.DataFrame
        Long-form table of sensitivity results with columns:

        - if not grouped: ``["R", "CWSL"]``
        - if grouped: ``group_cols + ["R", "CWSL"]``

        Each row corresponds to one (group, R) pair.

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If no valid ratios remain after filtering, or if sample weights are negative.

    Notes
    -----
    - This function delegates metric math to `eb-metrics` and only handles DataFrame plumbing.
    - `cwsl_sensitivity` determines exact behavior for non-finite values in inputs.
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

    # ---- validation: R grid ----
    R_arr = _normalize_R_list(R_list)

    # ---- validation: weights ----
    if sample_weight_col is not None:
        w_all = df[sample_weight_col].to_numpy(dtype=float)
        if np.any(w_all < 0):
            raise ValueError("sample weights must be non-negative.")

    # ---- compute ----
    results: List[Dict[str, Any]] = []

    if len(gcols) == 0:
        iter_groups = [((None,), df)]
    else:
        iter_groups = df.groupby(gcols, dropna=False, sort=False)

    for keys, g in iter_groups:
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        co_value: Union[float, np.ndarray]
        if isinstance(co, str):
            co_value = g[co].to_numpy(dtype=float)
        else:
            co_value = float(co)

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
            row: Dict[str, Any] = {"R": float(R_val), "CWSL": float(cwsl_val)}
            for col, value in zip(gcols, keys):
                row[col] = value
            results.append(row)

    result_df = pd.DataFrame(results)

    if len(gcols) > 0:
        result_df = result_df[gcols + ["R", "CWSL"]]
    else:
        result_df = result_df[["R", "CWSL"]]

    return result_df