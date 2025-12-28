# Cost Ratio Policy

This document provides the **API reference** for the cost ratio policy defined in
`eb_optimization.policies.cost_ratio_policy`.

A **CostRatioPolicy** encapsulates the asymmetric cost parameter **R** as a frozen,
governed decision artifact used during evaluation.

---

## Conceptual Role

Cost ratio policies formalize **cost asymmetry** between over-forecasting and
under-forecasting.

Within the Electric Barometer ecosystem, a CostRatioPolicy:

- Encodes a reviewed and approved value of R
- Separates estimation from application
- Ensures deterministic, auditable evaluation behavior

The policy does not compute metrics; it parameterizes them.

---

## Policy Object

### `CostRatioPolicy`

Immutable policy object representing asymmetric cost weighting.

**Signature**

```python
CostRatioPolicy(
    R,
    *,
    metadata=None,
)
```

**Parameters**

- `R`  
  Cost ratio parameter controlling asymmetry.

- `metadata` *(optional)*  
  Additional contextual information (e.g., estimation window, method, version).

**Behavioral Guarantees**

- Immutable after creation
- Serializable
- Safe to reuse across evaluations

---

## Defaults

eb-optimization provides a documented default cost ratio policy.

Defaults are:

- Explicit
- Reviewable
- Intended as safe fallbacks

Implicit defaults are intentionally avoided.

---

## Applying the Policy

### `apply_cost_ratio_policy`

Apply a cost ratio policy during evaluation.

**Signature**

```python
apply_cost_ratio_policy(
    y_true,
    y_pred,
    *,
    policy,
    **kwargs,
)
```

**Description**

Applies asymmetric cost weighting to evaluation metrics using the R value
encoded in the provided policy.

**Parameters**

- `y_true`  
  Actual observed values.

- `y_pred`  
  Forecast values.

- `policy`  
  Instance of `CostRatioPolicy`.

**Returns**

- Metric outputs parameterized by the policy.

---

## Entity-Level Application

Cost ratio policies may be applied:

- Globally (single R for all entities)
- Per entity (heterogeneous cost assumptions)

Entity-level application preserves determinism while enabling flexibility.

---

## Determinism Guarantees

Cost ratio policy application guarantees:

- Deterministic behavior
- No mutation of policy state
- Stable results across runs

---

## Governance and Lifecycle

Cost ratio policies should follow a governed lifecycle:

1. R estimated via tuning routines
2. Diagnostics reviewed
3. Policy instantiated and versioned
4. Applied consistently in evaluation
5. Periodically re-estimated as needed

---

## Common Usage Patterns

- System-wide cost asymmetry calibration
- Segment-level or entity-level cost modeling
- Stable evaluation across time periods

---

## Anti-Patterns

The following patterns are discouraged:

- Modifying R inside evaluation code
- Treating R as a model hyperparameter
- Re-estimating R during evaluation
- Using raw scalar R values without a policy

---

## Stability Notes

- `CostRatioPolicy` is a public, stable API
- Constructor and application semantics are stable within major versions
- Additional metadata fields may be added additively

---

## Related Documentation

- **API → Tuning → Cost Ratio**
- **How-To → Estimate Cost Ratio (R)**
- **Concepts → Policies**
- **Concepts → Artifacts and Governance**

---

*Cost ratio policies make asymmetry explicit — and explicit decisions are governable.*
