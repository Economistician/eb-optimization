"""
`eb_optimization` — optimization and tuning layer for the Electric Barometer ecosystem.

This package contains the **optimization layer** of Electric Barometer:

- **search**: generic, reusable search mechanics (grids, tie-breaking kernels)
- **tuning**: calibration and selection utilities (e.g., cost-ratio tuning, sensitivity sweeps)
- **policies**: frozen, declarative policy artifacts for downstream execution

It intentionally does **not** define metric primitives or evaluation math.
Those live in `eb-metrics` (and orchestration lives in `eb-evaluation`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .policies.ral_policy import (
    DEFAULT_RAL_POLICY,
    RALDeltas,
    RALTwoBandPolicy,
    apply_ral_policy,
)
from .tuning.cost_ratio import EntityCostRatioEstimate


def _resolve_version() -> str:
    """
    Resolve the installed distribution version.

    Returns
    -------
    str
        Installed version string. If the distribution is not installed (e.g., running
        from source), returns ``"0.0.0"``.
    """
    try:
        # Must match the distribution name in pyproject.toml ([project].name)
        return version("eb-optimization")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()

__all__ = [
    "DEFAULT_RAL_POLICY",
    "EntityCostRatioEstimate",
    "RALDeltas",
    "RALTwoBandPolicy",
    "__version__",
    "apply_ral_policy",
]
