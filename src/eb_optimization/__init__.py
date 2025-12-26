from __future__ import annotations

"""
eb-optimization
===============

Optimization, tuning, and policy governance components for the
Electric Barometer ecosystem.

This package owns:
- discrete search mechanics
- parameter tuning and calibration logic
- frozen policy artifacts for downstream execution

It does NOT define metric primitives or evaluation math.
"""

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        return version("eb-optimization")
    except PackageNotFoundError:
        # Editable installs / source checkouts
        return "0.0.0"


__version__ = _resolve_version()

__all__ = ["__version__"]