from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "broadcast_param",
    "handle_sample_weight",
    "to_1d_array",
]


def to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input to a 1D numpy float array.

    Parameters
    ----------
    x
        Array-like input.
    name
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        1D float array.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}")

    return arr


def broadcast_param(x: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """
    Broadcast a scalar or 1D array parameter to a target shape.

    Rules
    -----
    - Scalars are expanded to the given shape
    - 1D arrays must exactly match the target shape

    Parameters
    ----------
    x
        Scalar or 1D array parameter.
    shape
        Target shape.
    name
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``shape``.

    Raises
    ------
    ValueError
        If ``x`` is neither scalar nor matches the target shape.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)

    if arr.shape != shape:
        raise ValueError(
            f"{name} must be scalar or have shape {shape}; got shape {arr.shape}"
        )

    return arr


def handle_sample_weight(sample_weight: ArrayLike | None, n: int) -> np.ndarray:
    """
    Normalize sample weights to a non-negative 1D float array of length n.

    If ``sample_weight`` is None, returns an array of ones.

    Parameters
    ----------
    sample_weight
        None or a 1D array of non-negative weights.
    n
        Expected length.

    Returns
    -------
    numpy.ndarray
        1D float array of length n.

    Raises
    ------
    ValueError
        If weights are not length-n, not 1D, or contain negative values.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer; got {n}")

    if sample_weight is None:
        return np.ones(n, dtype=float)

    w = np.asarray(sample_weight, dtype=float)

    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(
            f"sample_weight must be a 1D array of length {n}; got shape {w.shape}"
        )

    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")

    return w
