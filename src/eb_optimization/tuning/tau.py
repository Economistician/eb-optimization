r"""
Data-driven tolerance (τ) selection utilities for HR@τ.

This module provides deterministic, residual-only methods for selecting the tolerance
parameter τ used by the hit-rate metric HR@τ (hit rate within an absolute-error band).

The hit-rate metric is:

$$
\mathrm{HR}@\tau = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left(|y_i-\hat{y}_i|\le \tau\right)
$$

Here, τ defines an *acceptability band*: the maximum absolute error considered operationally
acceptable.

Design notes
------------
- τ is estimated from historical residuals only (no exogenous data, no model assumptions).
- The module supports global τ estimation and entity-level τ estimation.
- Optional governance controls allow capping entity τ values by a global cap to prevent
  tolerance inflation.

Methods
-------
The global estimator supports three selection modes:

1. ``"target_hit_rate"``:
   choose τ such that a target fraction of residual magnitudes is covered:

   $$
   \tau = Q_h\left(|e|\right), \quad e_i = y_i-\hat{y}_i
   $$

   where $Q_h(\cdot)$ is the quantile function at level $h$.

2. ``"knee"``:
   select τ at a diminishing-returns point on the monotone curve $\mathrm{HR}@\tau$.

3. ``"utility"``:
   maximize a simple tradeoff between coverage and tolerance width:

   $$
   \tau^\* = \arg\max_{\tau \in \mathcal{T}}
   \left[\mathrm{HR}@\tau - \lambda\left(\frac{\tau}{\tau_{\max}}\right)\right]
   $$

The entity-level estimator runs the same procedure per entity, optionally capping each
entity τ by a global cap derived from the full residual distribution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from eb_metrics.metrics.service import hr_at_tau as _hr_at_tau_core

TauMethod = Literal["target_hit_rate", "knee", "utility"]


def _to_1d_float_array(x: pd.Series | np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    return np.asarray(x, dtype=float).reshape(-1)


def _nan_safe_abs_errors(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
) -> np.ndarray:
    r"""
    Compute absolute errors with NaN/inf filtering.

    Non-finite (y, yhat) pairs are dropped. The returned array contains:

    $$
    |e_i| = |y_i - \hat{y}_i|
    $$

    for finite pairs only.
    """
    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(
            f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}."
        )
    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    return np.abs(y_arr[mask] - yhat_arr[mask])


def _validate_tau(tau: float) -> float:
    """Validate τ as finite and non-negative."""
    if not np.isfinite(tau):
        raise ValueError(f"tau must be finite. Got {tau}.")
    if tau < 0:
        raise ValueError(f"tau must be >= 0. Got {tau}.")
    return float(tau)


def _quantile(x: np.ndarray, q: float) -> float:
    """Compute a quantile with basic guards; returns NaN for empty input."""
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile q must be in [0, 1]. Got {q}.")
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def _make_tau_grid(
    abs_errors: np.ndarray,
    grid: np.ndarray | Iterable[float] | None = None,
    grid_size: int = 101,
    grid_quantiles: tuple[float, float] = (0.0, 0.99),
) -> np.ndarray:
    """
    Construct a non-negative τ grid.

    If ``grid`` is provided, it is filtered to finite, unique, non-negative values.
    Otherwise, a linear grid is constructed between quantiles of the absolute error
    distribution.
    """
    if abs_errors.size == 0:
        return np.array([], dtype=float)

    if grid is not None:
        g = _to_1d_float_array(grid)
        g = g[np.isfinite(g)]
        g = np.unique(g)
        g = g[g >= 0]
        return g

    q_lo, q_hi = grid_quantiles
    if not (0 <= q_lo <= q_hi <= 1):
        raise ValueError(
            f"grid_quantiles must satisfy 0 <= q_lo <= q_hi <= 1. Got {grid_quantiles}."
        )

    lo = _quantile(abs_errors, q_lo)
    hi = _quantile(abs_errors, q_hi)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([], dtype=float)

    if hi < lo:
        hi = lo

    if grid_size < 2:
        return np.array([lo], dtype=float)

    return np.linspace(lo, hi, grid_size, dtype=float)


def hr_at_tau(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
    tau: float,
) -> float:
    r"""
    Compute HR@τ: fraction of observations whose absolute error is within τ.

    HR@τ is defined as:

    $$
    \mathrm{HR}@\tau = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left(|y_i-\hat{y}_i|\le \tau\right)
    $$

    This is an evaluation-friendly wrapper around the core implementation in
    ``eb_metrics.metrics.service.hr_at_tau``. Non-finite (y, yhat) pairs are dropped
    prior to delegating. If no finite pairs remain, returns ``np.nan``.
    """
    tau = _validate_tau(tau)

    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(
            f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}."
        )

    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    if not np.any(mask):
        return np.nan

    return float(_hr_at_tau_core(y_true=y_arr[mask], y_pred=yhat_arr[mask], tau=tau))


@dataclass(frozen=True)
class TauEstimate:
    """
    Result container for τ estimation.

    Attributes
    ----------
    tau : float
        Estimated tolerance τ (may be NaN if estimation failed).
    method : str
        Method identifier used to produce the estimate.
    n : int
        Number of finite (y, yhat) pairs used.
    diagnostics : dict[str, Any]
        Method-specific diagnostics intended for reporting and governance.
    """

    tau: float
    method: str
    n: int
    diagnostics: dict[str, Any]


def estimate_tau(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
    method: TauMethod = "target_hit_rate",
    *,
    target_hit_rate: float = 0.90,
    grid: np.ndarray | Iterable[float] | None = None,
    grid_size: int = 101,
    grid_quantiles: tuple[float, float] = (0.0, 0.99),
    knee_rule: Literal["slope_threshold", "max_distance"] = "slope_threshold",
    slope_threshold: float = 0.0025,
    lambda_: float = 0.10,
    tau_max: float | None = None,
    tau_floor: float = 0.0,
    tau_cap: float | None = None,
) -> TauEstimate:
    r"""Estimate a global tolerance τ from residuals.

    See module docstring for method definitions and design notes.
    """
    abs_errors = _nan_safe_abs_errors(y, yhat)
    n = int(abs_errors.size)

    if n == 0:
        return TauEstimate(
            tau=np.nan,
            method=str(method),
            n=0,
            diagnostics={"reason": "no_finite_pairs"},
        )

    tau_floor = _validate_tau(tau_floor)
    if tau_cap is not None:
        tau_cap = _validate_tau(tau_cap)

    if method == "target_hit_rate":
        if not (0.0 < target_hit_rate <= 1.0):
            raise ValueError(
                f"target_hit_rate must be in (0, 1]. Got {target_hit_rate}."
            )

        tau = _quantile(abs_errors, target_hit_rate)

        if np.isfinite(tau):
            tau = max(tau, tau_floor)
            if tau_cap is not None:
                tau = min(tau, tau_cap)

        diag = {
            "target_hit_rate": float(target_hit_rate),
            "achieved_hr_calibration": float(np.mean(abs_errors <= tau))
            if np.isfinite(tau)
            else np.nan,
            "abs_error_quantile_used": float(target_hit_rate),
            "tau_floor": float(tau_floor),
            "tau_cap": float(tau_cap) if tau_cap is not None else None,
        }
        return TauEstimate(
            tau=float(tau), method="target_hit_rate", n=n, diagnostics=diag
        )

    tau_grid = _make_tau_grid(
        abs_errors,
        grid=grid,
        grid_size=grid_size,
        grid_quantiles=grid_quantiles,
    )
    if tau_grid.size == 0:
        return TauEstimate(
            tau=np.nan,
            method=str(method),
            n=n,
            diagnostics={"reason": "empty_tau_grid"},
        )

    # HR curve on grid (monotone non-decreasing)
    e_sorted = np.sort(abs_errors)
    idx = np.searchsorted(e_sorted, tau_grid, side="right")
    hr_curve = idx / float(n)

    if method == "knee":
        if knee_rule == "slope_threshold":
            d_tau = np.diff(tau_grid)
            d_hr = np.diff(hr_curve)
            slope = np.where(d_tau > 0, d_hr / d_tau, np.inf)

            candidates = np.where(slope < slope_threshold)[0]
            pick_i = (
                int(candidates[0] + 1)
                if candidates.size > 0
                else int(len(tau_grid) - 1)
            )

            tau = float(tau_grid[pick_i])
            hr_pick = float(hr_curve[pick_i])

            diag = {
                "knee_rule": knee_rule,
                "slope_threshold": float(slope_threshold),
                "picked_index": pick_i,
                "picked_hr_calibration": hr_pick,
                "grid_size": int(tau_grid.size),
                "tau_grid_min": float(tau_grid.min()),
                "tau_grid_max": float(tau_grid.max()),
                "tau_floor": float(tau_floor),
                "tau_cap": float(tau_cap) if tau_cap is not None else None,
            }

        elif knee_rule == "max_distance":
            t0, t1 = float(tau_grid[0]), float(tau_grid[-1])
            t_norm = (tau_grid - t0) / (t1 - t0) if t1 > t0 else np.zeros_like(tau_grid)

            x = t_norm
            yv = hr_curve
            x0, y0 = 0.0, float(hr_curve[0])
            x1, y1 = 1.0, float(hr_curve[-1])

            denom = np.hypot(x1 - x0, y1 - y0)
            if denom == 0:
                pick_i = int(len(tau_grid) // 2)
            else:
                dist = (
                    np.abs((y1 - y0) * x - (x1 - x0) * yv + x1 * y0 - y1 * x0) / denom
                )
                pick_i = int(np.argmax(dist))

            tau = float(tau_grid[pick_i])
            hr_pick = float(hr_curve[pick_i])

            diag = {
                "knee_rule": knee_rule,
                "picked_index": pick_i,
                "picked_hr_calibration": hr_pick,
                "grid_size": int(tau_grid.size),
                "tau_grid_min": float(tau_grid.min()),
                "tau_grid_max": float(tau_grid.max()),
                "tau_floor": float(tau_floor),
                "tau_cap": float(tau_cap) if tau_cap is not None else None,
            }
        else:
            raise ValueError(f"Unknown knee_rule: {knee_rule}")

        tau = max(tau, tau_floor)
        if tau_cap is not None:
            tau = min(tau, tau_cap)

        return TauEstimate(tau=float(tau), method="knee", n=n, diagnostics=diag)

    if method == "utility":
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0. Got {lambda_}.")

        tau_max_val = _quantile(abs_errors, 0.99) if tau_max is None else float(tau_max)

        if not np.isfinite(tau_max_val) or tau_max_val <= 0:
            tau_max_val = (
                float(tau_grid[-1])
                if np.isfinite(tau_grid[-1]) and tau_grid[-1] > 0
                else 1.0
            )

        utility = hr_curve - float(lambda_) * (tau_grid / tau_max_val)
        pick_i = int(np.argmax(utility))

        tau = float(tau_grid[pick_i])
        hr_pick = float(hr_curve[pick_i])
        u_pick = float(utility[pick_i])

        tau = max(tau, tau_floor)
        if tau_cap is not None:
            tau = min(tau, tau_cap)

        diag = {
            "lambda_": float(lambda_),
            "tau_max": float(tau_max_val),
            "picked_index": pick_i,
            "picked_hr_calibration": hr_pick,
            "picked_utility": u_pick,
            "grid_size": int(tau_grid.size),
            "tau_grid_min": float(tau_grid.min()),
            "tau_grid_max": float(tau_grid.max()),
            "tau_floor": float(tau_floor),
            "tau_cap": float(tau_cap) if tau_cap is not None else None,
        }
        return TauEstimate(tau=float(tau), method="utility", n=n, diagnostics=diag)

    raise ValueError(f"Unknown method: {method}")


def estimate_entity_tau(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_col: str,
    yhat_col: str,
    method: TauMethod = "target_hit_rate",
    min_n: int = 30,
    estimate_kwargs: Mapping[str, Any] | None = None,
    cap_with_global: bool = False,
    global_cap_quantile: float = 0.99,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    r"""Estimate τ per entity from residuals. See the original EB docs for details."""
    if estimate_kwargs is None:
        estimate_kwargs = {}

    required = {entity_col, y_col, yhat_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if min_n < 1:
        raise ValueError(f"min_n must be >= 1. Got {min_n}.")

    global_cap = None
    if cap_with_global:
        abs_errors_all = _nan_safe_abs_errors(df[y_col], df[yhat_col])
        global_cap = _quantile(abs_errors_all, global_cap_quantile)
        if not np.isfinite(global_cap):
            global_cap = None

    rows: list[dict[str, Any]] = []

    for ent, g in df.groupby(entity_col, dropna=False):
        abs_errors = _nan_safe_abs_errors(g[y_col], g[yhat_col])
        n = int(abs_errors.size)

        if n < min_n:
            rows.append(
                {
                    entity_col: ent,
                    "tau": np.nan,
                    "n": n,
                    "method": method,
                    "reason": f"min_n_not_met(<{min_n})",
                }
            )
            continue

        est = estimate_tau(
            y=g[y_col],
            yhat=g[yhat_col],
            method=method,
            **dict(estimate_kwargs),
        )

        tau_val = est.tau
        if global_cap is not None and np.isfinite(tau_val):
            tau_val = float(min(tau_val, global_cap))

        row: dict[str, Any] = {
            entity_col: ent,
            "tau": tau_val,
            "n": est.n,
            "method": est.method,
        }

        if include_diagnostics:
            diag = dict(est.diagnostics or {})
            row["diagnostics"] = diag
            row["achieved_hr_calibration"] = diag.get(
                "achieved_hr_calibration", diag.get("picked_hr_calibration")
            )
            row["tau_floor"] = diag.get("tau_floor")
            row["tau_cap"] = diag.get("tau_cap")
            if method == "utility":
                row["lambda_"] = diag.get("lambda_")
                row["tau_max"] = diag.get("tau_max")
                row["picked_utility"] = diag.get("picked_utility")
            if method == "knee":
                row["knee_rule"] = diag.get("knee_rule")

        if global_cap is not None:
            row["global_cap_tau"] = global_cap
            row["global_cap_quantile"] = float(global_cap_quantile)

        rows.append(row)

    out = pd.DataFrame(rows)

    base_cols = [entity_col, "tau", "n", "method"]
    extra_cols = [c for c in out.columns if c not in base_cols]
    return out[base_cols + extra_cols]


def hr_auto_tau(
    y: pd.Series | np.ndarray | Iterable[float],
    yhat: pd.Series | np.ndarray | Iterable[float],
    method: TauMethod = "target_hit_rate",
    **estimate_kwargs: Any,
) -> tuple[float, float, dict[str, Any]]:
    r"""Estimate τ from residuals, then compute HR@τ. Returns (hr, tau, diagnostics)."""
    est = estimate_tau(y=y, yhat=yhat, method=method, **estimate_kwargs)
    if not np.isfinite(est.tau):
        return (np.nan, np.nan, dict(est.diagnostics or {}))

    hr = hr_at_tau(y, yhat, est.tau)
    return (float(hr), float(est.tau), dict(est.diagnostics or {}))
