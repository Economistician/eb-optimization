from __future__ import annotations

"""
Search primitives for the Electric Barometer optimization layer.

The `eb_optimization.search` package contains **generic, reusable search kernels**
that implement *how* to search over a discrete candidate space, independent of
any specific metric, policy, or business objective.

Design intent
-------------
- **search/**: mechanics of search (argmin/argmax, tie-breaking, grid iteration)
- **tuning/**: what to search + objective definition + returned artifacts
- **policies/**: frozen, declarative outputs of tuning (no search at runtime)

Key rules
---------
- No domain-specific policy logic
- No metric semantics
- No pandas-heavy workflows
- Pure, deterministic search utilities
"""

__all__ = [
    "grid",
    "kernels",
]