# Readiness Adjustment Layer (RAL) Tuning

This document provides the **API reference** for Readiness Adjustment Layer (RAL) tuning
functionality in `eb_optimization.tuning.ral`.

RAL tuning calibrates how forecasts are adjusted to reflect execution readiness *prior*
to evaluation. It produces governed policy artifacts rather than numeric parameters.

---

## Conceptual Role

RAL tuning addresses the gap between **theoretical demand forecasts** and **achievable service**.

Within the Electric Barometer ecosystem, RAL:

- Operates upstream of metric evaluation
- Encodes readiness assumptions explicitly
- Protects evaluation integrity from over-crediting forecasts

RAL tuning is conservative by design and prioritizes realism over score optimization.

---

## Public Functions

### `tune_ral_policy`

Tune and return a readiness adjustment policy based on historical data.

**Signature**

```python
tune_ral_policy(
    df,
    *,
    entity_col=None,
    actual_col,
    forecast_col,
    **kwargs,
)
```

**Description**

Analyzes the relationship between forecasts and realized outcomes to determine
an appropriate readiness adjustment rule. The result is a fully formed `RALPolicy`
artifact suitable for deterministic application.

**Parameters**

- `df`  
  Panel DataFrame containing forecasts and actual outcomes.

- `entity_col` *(optional)*  
  Column identifying entities. When provided, tuning may account for entity-level behavior.

- `actual_col`  
  Column containing realized outcomes.

- `forecast_col`  
  Column containing unadjusted forecast values.

**Returns**

- `RALPolicy`  
  Immutable policy artifact encoding readiness adjustment behavior and metadata.

---

## Output Artifacts

Unlike other tuning routines, RAL tuning returns a **policy object directly** rather than
a scalar estimate.

The returned `RALPolicy` typically includes:

- Adjustment parameters or rules
- Metadata describing the tuning context
- Information necessary for deterministic application

This reflects RAL’s role as a behavioral layer rather than a numeric parameter.

---

## Determinism Guarantees

RAL tuning guarantees:

- Deterministic outputs for identical inputs
- Explicit selection logic
- No stochastic or adaptive behavior

These guarantees ensure reproducibility across environments and time.

---

## Interaction with Policies

RAL tuning produces a policy artifact that is applied via:

```python
from eb_optimization.policies.ral_policy import apply_ral_policy
```

RAL policies are applied *before* τ and R policies during evaluation.

---

## Defaults and Conservatism

When multiple adjustment strategies are viable:

- Conservative rules are preferred
- Stability is favored over marginal gains
- Ambiguity is resolved explicitly

This prevents RAL from becoming a performance enhancement mechanism.

---

## Common Usage Patterns

- Calibrating evaluation realism for operational forecasting
- Aligning metrics with execution capacity
- Freezing readiness assumptions for governance

---

## Anti-Patterns

The following patterns are discouraged:

- Treating RAL as a scoring optimizer
- Applying RAL adjustments during model training
- Re-tuning RAL automatically without review
- Using RAL to mask forecast quality issues

---

## Stability Notes

- `tune_ral_policy` is a public, stable API
- Policy semantics are stable within major versions
- New adjustment strategies may be added additively

---

## Related Documentation

- **How-To → Tune RAL**
- **Concepts → Policies**
- **Concepts → Tuning**
- **API → Policies → RALPolicy**

---

*RAL tuning exists to calibrate realism — not to improve metrics.*
