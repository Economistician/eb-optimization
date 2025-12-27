"""
Tuning utilities for the Electric Barometer ecosystem.

The `eb_optimization.tuning` package contains grid-search and calibration helpers
used to *select* hyperparameters and operating points for evaluation workflows.

Design intent
-------------
- **eb-metrics**: metric math (single-source-of-truth implementations)
- **eb-evaluation**: deterministic evaluation plumbing / orchestration
- **eb-optimization**: tuning, search, calibration, and sensitivity sweeps

Public API philosophy
---------------------
Keep the package import surface stable by exporting *modules* instead of
re-exporting function symbols. This avoids import-time breakage when internals
are renamed during refactors.
"""

from __future__ import annotations

from . import cost_ratio, ral, sensitivity, tau

__all__ = [
    "cost_ratio",
    "ral",
    "sensitivity",
    "tau",
]
