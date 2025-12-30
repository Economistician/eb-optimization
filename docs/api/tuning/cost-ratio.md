# Cost Ratio Tuning (R)

This document provides the **API reference** for cost ratio tuning functionality in
`eb_optimization.tuning.cost_ratio`.

Cost ratio tuning selects the asymmetric cost parameter **R** used by cost-weighted
metrics such as CWSL. The API is intentionally narrow, deterministic, and artifact-driven.

---

## Conceptual Role

Cost ratio tuning determines **how much more costly one class of error is relative to another**.
It does not compute metrics and does not apply policies.

Instead, it:

- Evaluates candidate R values against balance criteria
- Produces structured estimation artifacts
- Supports both global and entity-level calibration

---

## Public Functions

### `estimate_R_cost_balance`

Estimate a global cost ratio using balance-based criteria.

**Signature**

```python
estimate_R_cost_balance(
    y_true,
    y_pred,
    *,
    candidate_R=None,
    **kwargs,
)
```

**Description**

Selects an R value such that opposing cost-weighted service losses are balanced according
to a deterministic criterion.

**Parameters**

- `y_true`
  Actual observed values.

- `y_pred`
  Forecast values corresponding to `y_true`.

- `candidate_R` *(optional)*
  Iterable of candidate R values. If not provided, a documented default grid is used.

**Returns**

- `CostRatioEstimate`
  Structured estimate containing:
  - Selected R value
  - Balance diagnostics
  - Method metadata

---

### `estimate_entity_R_from_balance`

Estimate cost ratios per entity.

**Signature**

```python
estimate_entity_R_from_balance(
    df,
    *,
    entity_col,
    actual_col,
    forecast_col,
    candidate_R=None,
    **kwargs,
)
```

**Description**

Computes one cost ratio estimate per entity using balance-based selection.

**Parameters**

- `df`
  Panel DataFrame containing actuals and forecasts.

- `entity_col`
  Column identifying entities (e.g., store, region).

- `actual_col`
  Column containing actual values.

- `forecast_col`
  Column containing forecast values.

- `candidate_R` *(optional)*
  Iterable of candidate R values.

**Returns**

- Mapping or DataFrame of `CostRatioEstimate` objects keyed by entity.

---

## Estimation Artifacts

Both functions return structured estimation artifacts rather than raw scalars.

Artifacts typically include:

- Selected R value
- Diagnostic statistics
- Sample size and coverage
- Method identifiers

These artifacts are intended for review and governance.

---

## Determinism Guarantees

Cost ratio tuning guarantees:

- Deterministic outputs for identical inputs
- Explicit tie-breaking rules
- No stochastic behavior

These guarantees are critical for reproducibility.

---

## Interaction with Policies

Estimated R values are not applied directly.

Instead, they are used to construct a **CostRatioPolicy**:

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy

policy = CostRatioPolicy(R=estimate.R)
```

This separation enforces clean layering.

---

## Defaults and Candidate Grids

If no candidate grid is provided:

- A documented default grid is used
- Grid selection is explicit and stable
- Behavior does not change silently between versions

Users may override grids when appropriate.

---

## Common Usage Patterns

- Global calibration for system-wide cost assumptions
- Entity-level calibration for heterogeneous environments
- Periodic re-estimation under regime change

---

## Anti-Patterns

The following patterns are discouraged:

- Treating R as a model hyperparameter
- Selecting R directly from sensitivity minima
- Re-estimating R inside evaluation workflows
- Using floating-point equality without tolerance

---

## Stability Notes

- Function signatures are considered **public API**
- Behavior is stable within major versions
- New estimation strategies may be added additively

---

## Related Documentation

- **How-To → Estimate Cost Ratio (R)**
- **Concepts → Tuning**
- **Concepts → Policies**
- **API → Policies → CostRatioPolicy**

---

*Cost ratio tuning formalizes asymmetry — it does not invent it.*
