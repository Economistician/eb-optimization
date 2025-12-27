"""
Grid construction utilities for optimization search spaces.

This module provides small, deterministic helpers for constructing bounded,
interpretable parameter grids used by offline optimization routines.

Responsibilities:
- Create numerically stable, reproducible grids for scalar parameters
- Enforce positivity and boundary constraints
- Standardize grid behavior across tuners

Non-responsibilities:
- Evaluating objectives
- Selecting optimal parameters
- Performing any optimization logic

Design philosophy:
This utility favors bounded, discrete search spaces for interpretability, auditability,
and deployability of learned policies.
"""

from __future__ import annotations

import math

import numpy as np


def make_float_grid(
    x_min: float, x_max: float, step: float, decimals: int = 10
) -> np.ndarray:
    r"""Create a numerically robust 1D grid over a closed interval.

    This utility is used throughout optimization to create bounded, interpretable
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
    decimals
        Rounding precision used to stabilize floats and de-duplicate.

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
    This utility ensures reproducible and stable grid construction for parameter tuning
    and optimization purposes, while favoring discrete, bounded search spaces for
    interpretability and deployability.
    """
    if step <= 0.0:
        raise ValueError("step must be strictly positive.")
    if x_min <= 0.0:
        raise ValueError("x_min must be strictly positive.")
    if x_max < x_min:
        raise ValueError("x_max must be >= x_min.")

    # Step-aligned grid starts at the first multiple of `step` that is >= x_min.
    start = math.ceil(x_min / step) * step

    # Generate core grid points [start, start + step, ..., x_max]
    # Add a tiny epsilon to ensure inclusion when we're right on the boundary.
    eps = 10 ** (-(decimals + 2))
    core = np.arange(start, x_max + eps, step, dtype=float)

    # Always include x_min and x_max explicitly
    vals = np.concatenate(([float(x_min)], core, [float(x_max)]))

    # Stabilize: round then unique then sort
    vals = np.round(vals, decimals=decimals)
    vals = np.unique(vals)
    vals.sort()

    return vals
