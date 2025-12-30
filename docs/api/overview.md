# API Overview

This document describes the **public API surface** of eb-optimization and how to
navigate the reference documentation.

eb-optimization is the decision-optimization layer of the Electric Barometer ecosystem.
Its API is intentionally small, explicit, and policy-centered.

---

## API Design Goals

The eb-optimization API is designed to prioritize:

- **Explicitness**: decision parameters are first-class objects
- **Determinism**: identical inputs produce identical outputs
- **Governance**: decisions can be frozen, reviewed, and versioned
- **Composability**: works standalone or as part of the EB stack
- **Stability**: core public symbols change slowly

---

## What Counts as “Public”

Public APIs in eb-optimization are those that are:

- Imported from stable module paths
- Documented in this API reference
- Covered by tests as part of contract behavior

Anything not documented here should be treated as **internal** and may change without notice.

---

## Module Map

The package is organized into three primary areas:

- **policies/**: policy artifacts and application helpers
- **tuning/**: estimation and calibration routines that produce policy parameters
- **search/**: deterministic candidate selection utilities used by tuning

The API reference mirrors this structure.

---

## Common Import Patterns

### Policy Objects

Policy objects are created explicitly and passed to application helpers.

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy
from eb_optimization.policies.tau_policy import TauPolicy
from eb_optimization.policies.ral_policy import RALPolicy
```

### Tuning (Estimation)

Tuning routines return structured estimates which are then frozen into policies.

```python
from eb_optimization.tuning.cost_ratio import estimate_R_cost_balance
from eb_optimization.tuning.tau import estimate_tau
from eb_optimization.tuning.ral import tune_ral_policy
```

### Search Utilities

Search utilities are typically used indirectly through tuning routines, but are public when
explicitly documented.

```python
from eb_optimization.search.kernels import argmin_over_candidates
```

---

## Stability and Versioning

eb-optimization follows semantic versioning:

- **Patch** releases: bug fixes, no API changes
- **Minor** releases: additive APIs, new tuning strategies, new docs
- **Major** releases: breaking changes

Policy objects and top-level tuning functions are treated as the most stable components.

---

## Naming Conventions

The API uses consistent naming patterns:

- `estimate_*`: produces an estimate artifact from data
- `tune_*`: calibrates a policy or decision rule set
- `apply_*`: applies a policy deterministically to inputs
- `*_policy`: modules defining policy dataclasses and defaults

These conventions support discoverability and reduce ambiguity.

---

## Entity-Level vs Global APIs

Many APIs come in two variants:

- **Global**: one parameter or policy for the entire dataset
- **Entity-level**: one parameter or policy per entity (e.g., store)

Entity-level functions typically accept a DataFrame and an `entity_col`.

---

## Diagnostics and Artifacts

Tuning functions return structured artifacts rather than raw scalars.

These artifacts are intended to:

- Preserve estimation intent and method
- Carry diagnostics for review
- Feed into policy construction
- Support audit and traceability

Treat artifacts as outputs worth saving.

---

## Recommended Reading Order

If you are new to the package:

1. **Concepts → Layering and Scope**
2. **Concepts → Policies**
3. **How-To → Quickstart**
4. **How-To → Estimate R / Estimate τ / Tune RAL**
5. **API Reference** pages for specific modules

---

## Next API Pages

- **API → Policies**: policy objects and application helpers
- **API → Tuning**: estimation routines and artifacts
- **API → Search**: grids and deterministic tie-breaking kernels

---

*The eb-optimization API is small by design — because decisions should be explicit, not sprawling.*
