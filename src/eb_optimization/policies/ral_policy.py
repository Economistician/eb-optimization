from __future__ import annotations

"""
Policy artifacts for the Readiness Adjustment Layer (RAL).

This module defines portable, immutable policy objects produced by offline
optimization and consumed by deterministic evaluation and production workflows.

Responsibilities:
- Represent learned RAL parameters (global and optional segment-level uplifts)
- Provide a stable, serializable contract between optimization and evaluation
- Support audit and governance workflows

Non-responsibilities:
- Learning or tuning parameters
- Applying policies to data
- Defining metric or loss functions

Design philosophy:
Policies are artifacts, not algorithms. They encode *decisions* derived from
optimization, not the optimization process itself.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class RALPolicy:
    r"""Portable policy artifact for the Readiness Adjustment Layer (RAL).

    A :class:`~eb_optimization.policies.ral_policy.RALPolicy` is the *output* of an
    offline tuning process (e.g., grid search or evolutionary optimization) and the
    *input* to deterministic evaluation / production application.

    Conceptually, RAL applies a multiplicative uplift to a baseline forecast:

    $$
    \hat{y}^{(r)} = u \cdot \hat{y}
    $$

    where `u` can be either:

    - a **global uplift** (`global_uplift`), applied to all rows, and/or
    - **segment-level** uplifts stored in `uplift_table`, keyed by `segment_cols`

    Segment-level uplifts must fall back to the global uplift for unseen segment
    combinations at application time.

    Attributes
    ----------
    global_uplift
        The global multiplicative uplift used as a fallback and baseline readiness adjustment.
    segment_cols
        The segmentation columns used to key `uplift_table`. Empty means "global-only".
    uplift_table
        Optional DataFrame with columns `[*segment_cols, "uplift"]` containing
        segment-level uplifts. If `None` or empty, the policy is global-only.

    Notes
    -----
    This dataclass is intentionally simple and serializable. It is meant to be:

    - produced offline in `eb-optimization`
    - applied deterministically in `eb-evaluation`
    - loggable/auditable as part of operational governance

    The policy does *not* encode metric definitions or optimization state—only the
    artifacts needed to execute the adjustment.
    """

    global_uplift: float
    segment_cols: Sequence[str] = ()
    uplift_table: Optional[pd.DataFrame] = None

    def is_segmented(self) -> bool:
        """Return True if the policy contains segment-level uplifts."""
        return bool(self.segment_cols) and self.uplift_table is not None and not self.uplift_table.empty