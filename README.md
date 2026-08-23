# Electric Barometer · Optimization (`eb-optimization`)

[![CI](https://github.com/Economistician/eb-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-optimization/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-optimization)
![PyPI](https://img.shields.io/pypi/v/eb-optimization)

Decision and policy layer for the Electric Barometer ecosystem, responsible for tuning, calibration, and governed parameter selection.

---

## Overview

`eb-optimization` calibrates operational parameters (cost ratios, τ, readiness controls) and freezes them into auditable runtime policies. It does not compute metrics or run evaluation panels; those belong to `eb-metrics` and `eb-evaluation`.

---

## Installation

`eb-optimization` is distributed as a standard Python package.

```bash
pip install eb-optimization
```

---

## Core Concepts

- **Parameter governance** — Operational parameters (e.g., cost ratios, tolerances) should be selected through explicit, reproducible rules rather than ad-hoc tuning or implicit defaults.
- **Search over candidate spaces** — Optimization is framed as deterministic search over bounded, interpretable candidate sets, enabling transparent tradeoffs and stable outcomes.
- **Cost balance calibration** — Asymmetric operational costs can be balanced by selecting parameters that equalize or appropriately trade off opposing risk exposures.
- **Tolerance selection from residuals** — Acceptable error bands can be learned directly from historical performance, reflecting empirical system behavior rather than arbitrary thresholds.
- **Policy separation** — Calibration logic is separated from frozen policy artifacts so that parameter selection is auditable, versioned, and safely applied in downstream systems.
- **Decision-aligned optimization** — Optimization is evaluated by operational interpretability and governance fitness, not by abstract numerical optimality alone.

---

## Minimal Example

The example below illustrates a typical optimization workflow using `eb-optimization`: calibrating an operational parameter from historical data and applying it via a frozen policy.

```python
import numpy as np
from eb_optimization.policies import (
    CostRatioPolicy,
    apply_cost_ratio_policy,
)

# Historical actuals and forecasts
y_true = np.array([10, 12, 15, 20])
y_pred = np.array([9, 14, 18, 17])

# Define a frozen cost-ratio policy
policy = CostRatioPolicy(
    R_grid=(0.5, 1.0, 2.0, 3.0),
    co=1.0,
)

# Estimate a global cost ratio R
R, diagnostics = apply_cost_ratio_policy(
    y_true=y_true,
    y_pred=y_pred,
    policy=policy,
)

print(R)
```

---

## License

BSD 3-Clause License.
© 2026 Kyle Corrie.
