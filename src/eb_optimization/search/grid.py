from __future__ import annotations

"""
Grid construction utilities for optimization search spaces.

This module provides small, deterministic helpers for constructing bounded,
interpretable parameter grids used by offline optimization routines in
`eb-optimization`.

Responsibilities:
- Create numerically stable, reproducible grids for scalar parameters
- Enforce positivity and boundary constraints
- Standardize grid behavior across tuners

Non-responsibilities:
- Evaluating objectives
- Selecting optimal parameters
- Performing any optimization logic

Design philosophy:
Electric Barometer favors bounded, discrete search spaces to preserve
interpretability, auditability, and deployability of learned policies.
"""

import numpy as np


def make_float_grid(x_min: float, x_max: float, step: float) -> np.ndarray:
    r"""Create a numerically robust 1D grid over a closed interval.

    This utility is used throughout `eb-optimization` to create bounded, interpretable
    candidate sets for discrete parameter search (e.g., uplift multipliers, thresholds).

    The returned grid:

    - starts at `x_min`
    - increments by `step`
    - includes `x_max` (to the extent permitted by floating-point arithmetic)
    - is clipped and de-duplicated for numerical stability

    Parameters
    ----------
    x_min
        Lower bound for the grid (inclusive). Must be strictly positive.
    x_max
        Upper bound for the grid (inclusive). Must be greater than or equal to `x_min`.
    step
        Step size between candidates. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        A 1D array of unique grid values in ascending order.

    Raises
    ------
    ValueError
        If `step` is not strictly positive, if `x_min` is not strictly positive,
        or if `x_max < x_min`.

    Notes
    -----
    Electric Barometer intentionally prefers *bounded, discrete* search spaces for
    interpretability and deployability. This function standardizes grid creation
    across tuners so results are reproducible and comparable.
    """
    if step <= 0.0:
        raise ValueError("step must be strictly positive.")
    if x_min <= 0.0:
        raise ValueError("x_min must be strictly positive.")
    if x_max < x_min:
        raise ValueError("x_max must be >= x_min.")

    span = x_max - x_min
    n_steps = int(round(span / step))

    grid = x_min + step * np.arange(n_steps + 1)

    # Clip and de-duplicate for numerical robustness
    grid = np.clip(grid, x_min, x_max)
    grid = np.unique(np.round(grid, 10))

    return grid