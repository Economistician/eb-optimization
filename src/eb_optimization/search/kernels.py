"""
Generic discrete-search kernels for eb-optimization.

This module defines *mechanical* optimization primitives for selecting an
argmin or argmax over a finite candidate set.

Responsibilities
----------------
- Iterate over a discrete candidate set
- Evaluate a scalar objective function
- Apply deterministic tie-breaking rules
- Return the selected candidate and its score

Non-responsibilities
--------------------
- Defining candidate grids (handled by ``search.grid``)
- Computing domain-specific objectives (e.g., cost, HR@τ, utility)
- Inspecting data distributions or residuals
- Returning diagnostics, plots, or policies

Design philosophy
-----------------
These kernels are intentionally simple, deterministic, and domain-agnostic.
They serve as reusable building blocks for higher-level tuning logic
(``tuning`` modules), enabling consistent and auditable optimization behavior
across the Electric Barometer ecosystem.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, TypeVar

import numpy as np

T = TypeVar("T")

__all__ = ["argmax_over_candidates", "argmin_over_candidates"]


def argmin_over_candidates(
    candidates: Iterable[T],
    score_fn: Callable[[T], float],
    *,
    tie_break: Literal["first", "last", "closest_to_zero"] = "first",
) -> tuple[T, float]:
    """
    Select the candidate that minimizes a scalar score.

    Parameters
    ----------
    candidates
        Iterable of candidate values (e.g., floats, ints, tuples).
    score_fn
        Function mapping a candidate to a scalar score.
    tie_break
        Deterministic tie-breaking rule:
        - ``"first"``: first candidate with minimal score
        - ``"last"``: last candidate with minimal score
        - ``"closest_to_zero"``: among ties, choose candidate with smallest
          absolute value

    Returns
    -------
    (best_candidate, best_score)

    Raises
    ------
    ValueError
        If ``candidates`` is empty or if ``score_fn`` returns a non-finite value.
    """
    best_candidate: T | None = None
    best_score: float | None = None

    for cand in candidates:
        score = float(score_fn(cand))

        if not np.isfinite(score):
            raise ValueError(f"Non-finite score returned for candidate {cand!r}")

        if best_score is None or score < best_score:
            best_candidate = cand
            best_score = score
            continue

        if score == best_score and (
            tie_break == "last"
            or (
                tie_break == "closest_to_zero"
                and abs(float(cand)) < abs(float(best_candidate))  # type: ignore[arg-type]
            )
        ):
            best_candidate = cand

    if best_candidate is None or best_score is None:
        raise ValueError("candidates must be a non-empty iterable")

    return best_candidate, best_score


def argmax_over_candidates(
    candidates: Iterable[T],
    score_fn: Callable[[T], float],
    *,
    tie_break: Literal["first", "last", "closest_to_zero"] = "first",
) -> tuple[T, float]:
    """
    Select the candidate that maximizes a scalar score.

    This is the argmax analogue of ``argmin_over_candidates``.
    """
    return argmin_over_candidates(
        candidates=candidates,
        score_fn=lambda c: -float(score_fn(c)),
        tie_break=tie_break,
    )
