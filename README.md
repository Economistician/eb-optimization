# Electric Barometer · Optimization (`eb-optimization`)

[![CI](https://github.com/Economistician/eb-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-optimization/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-optimization)
![PyPI](https://img.shields.io/pypi/v/eb-optimization)

Decision and policy layer for the Electric Barometer ecosystem, responsible for tuning, calibration, and governed parameter selection.

---

## Overview

This repository contains the optimization, tuning, and policy governance layer of the Electric Barometer ecosystem. It defines how key evaluation parameters—such as cost ratios, tolerances, and readiness controls—are selected from data, validated under governance rules, and formalized into deterministic policies that can be reused across systems and environments.

Rather than computing metrics or running evaluations, this repository focuses on decision logic: how parameters are calibrated, how tradeoffs are resolved, and how those decisions are frozen into auditable artifacts. It provides the bridge between metric theory and operational deployment, ensuring that forecast evaluation behavior is consistent, explainable, and governed by explicit intent rather than ad-hoc configuration.

---

## Role in the Electric Barometer Ecosystem

`eb-optimization` defines the parameter selection, calibration, and governance logic used throughout the Electric Barometer ecosystem. It is responsible for determining how key operational parameters—such as cost ratios, tolerance bands, and readiness controls—are selected from data in a disciplined, reproducible, and decision-aware manner.

This repository focuses exclusively on optimization mechanics and policy formation. It does not define metric primitives, perform evaluation orchestration, manage model interfaces, or execute runtime decision logic. Those responsibilities are handled by adjacent layers in the ecosystem that compute metrics, evaluate forecasts, or apply frozen policies in production workflows.

By separating parameter selection and governance from metric semantics and execution concerns, eb-optimization provides a stable optimization layer that enables consistent calibration, transparent decision rules, and auditable policy artifacts across heterogeneous forecasting and operational contexts.

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
© 2025 Kyle Corrie.