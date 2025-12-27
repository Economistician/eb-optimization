"""
Search primitives for the Electric Barometer optimization layer.

The `eb_optimization.search` package contains generic, reusable search utilities
for iterating over candidate spaces (e.g., grid generation, argmin/argmax kernels).
"""

from __future__ import annotations

from . import grid, kernels

__all__ = ["grid", "kernels"]
