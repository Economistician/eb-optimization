# Service Threshold Tuning (τ)

This document provides the **API reference** for service threshold tuning functionality in
`eb_optimization.tuning.tau`.

Service threshold tuning selects **τ (tau)**, the parameter that defines service success for
threshold- and hit-rate–based EB metrics. The API is designed to be deterministic, artifact-driven,
and operationally interpretable.

---

## Conceptual Role

τ defines what it means for a forecast to be considered “good enough” from a service perspective.

Within the Electric Barometer ecosystem, τ tuning:

- Calibrates service definitions from data
- Supports global and entity-level thresholds
- Produces structured artifacts for governance

τ tuning does not apply policies and does not evaluate forecasts end-to-end.

---

## Public Functions

### `estimate_tau`

Estimate a global service threshold τ.

**Signature**

```python
estimate_tau(
    y_true,
    y_pred,
    *,
    method=None,
    candidate_tau=None,
    **kwargs,
)
```

**Description**

Selects τ according to the chosen method (e.g., hit-rate alignment or metric alignment),
using deterministic candidate evaluation and explicit tie-breaking.

**Parameters**

- `y_true`
  Actual observed values.

- `y_pred`
  Forecast values corresponding to `y_true`.

- `method` *(optional)*
  Strategy used to select τ. If not provided, a documented default method is used.

- `candidate_tau` *(optional)*
  Iterable of candidate τ values. If not provided, a documented default grid is used.

**Returns**

- `TauEstimate`
  Structured estimate containing:
  - Selected τ value
  - Method metadata
  - Diagnostics suitable for review

---

### `estimate_entity_tau`

Estimate τ per entity from panel data.

**Signature**

```python
estimate_entity_tau(
    df,
    *,
    entity_col,
    actual_col,
    forecast_col,
    method=None,
    candidate_tau=None,
    **kwargs,
)
```

**Description**

Computes one `TauEstimate` per entity using the specified selection method and
candidate grid.

**Parameters**

- `df`
  Panel DataFrame containing actuals and forecasts.

- `entity_col`
  Column identifying entities (e.g., store, region).

- `actual_col`
  Column containing actual values.

- `forecast_col`
  Column containing forecast values.

- `method` *(optional)*
  Selection strategy.

- `candidate_tau` *(optional)*
  Iterable of candidate τ values.

**Returns**

- Mapping or DataFrame of `TauEstimate` objects keyed by entity.

---

### `hr_auto_tau`

Automatically select τ using a hit-rate–driven strategy.

**Signature**

```python
hr_auto_tau(
    y_true,
    y_pred,
    *,
    candidate_tau=None,
    **kwargs,
)
```

**Description**

A convenience wrapper that selects τ using a stable hit-rate–driven approach with
explicit candidate evaluation and deterministic tie-breaking.

**Returns**

- `TauEstimate`

---

### `hr_at_tau`

Compute hit-rate at a specified τ.

**Signature**

```python
hr_at_tau(
    y_true,
    y_pred,
    *,
    tau,
    **kwargs,
)
```

**Description**

Evaluates HR@τ for a given τ, typically used by tuning routines and diagnostics.

**Returns**

- Scalar or structured result representing HR@τ.

---

## Estimation Artifacts

τ tuning returns `TauEstimate` artifacts rather than raw values.

Artifacts typically include:

- Selected τ
- Method identifier
- Diagnostics (e.g., achieved hit rate, candidate scores)
- Sample size and coverage metadata

Treat these artifacts as governance outputs worth saving.

---

## Determinism Guarantees

τ tuning guarantees:

- Deterministic outputs for identical inputs
- Explicit tie-breaking for equivalent candidates
- No stochastic components

These guarantees are critical for reproducible evaluation.

---

## Interaction with Policies

Estimated τ values are converted into a `TauPolicy`:

```python
from eb_optimization.policies.tau_policy import TauPolicy

policy = TauPolicy(tau=estimate.tau)
```

τ policies are applied after RAL and before cost asymmetry weighting during evaluation.

---

## Defaults and Candidate Grids

If no candidate grid is provided:

- A documented default grid is used
- Behavior remains stable across minor versions
- Users may override grids explicitly for domain-specific needs

---

## Common Usage Patterns

- Global τ for a system-wide service definition
- Entity-level τ for heterogeneous environments
- Periodic re-estimation aligned to service expectation changes

---

## Anti-Patterns

The following patterns are discouraged:

- Treating τ as a model hyperparameter
- Re-tuning τ inside evaluation pipelines
- Using inconsistent τ definitions across teams
- Overfitting τ to short or unrepresentative windows

---

## Stability Notes

- The functions listed above are public APIs
- Output artifact structure is stable within major versions
- New methods may be added additively via `method`

---

## Related Documentation

- **How-To → Estimate τ**
- **Concepts → Tuning**
- **Concepts → Policies**
- **API → Policies → TauPolicy**

---

*τ tuning formalizes service definitions — it does not discover them magically.*
