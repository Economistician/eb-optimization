# RAL Policy

This document provides the **API reference** for the Readiness Adjustment Layer (RAL)
policy defined in `eb_optimization.policies.ral_policy`.

A **RALPolicy** encapsulates how forecasts are adjusted to reflect execution readiness
*prior* to evaluation.

---

## Conceptual Role

RAL policies exist to ensure evaluation reflects **achievable service**, not merely
theoretical demand forecasts.

Within the Electric Barometer ecosystem, a RALPolicy:

- Encodes readiness assumptions explicitly
- Adjusts forecasts deterministically prior to metric computation
- Protects evaluation integrity from over-crediting forecasts
- Functions as a governed behavioral artifact (not a numeric parameter)

---

## Policy Object

### `RALPolicy`

Immutable policy object representing readiness adjustment behavior.

**Signature**

```python
RALPolicy(
    *,
    metadata=None,
    **params,
)
```

**Parameters**

Because readiness adjustment behavior may evolve over time, RALPolicy is treated as a
policy artifact whose internal parameters are defined by the selected adjustment strategy.

Common parameter categories include:

- Adjustment magnitude parameters
- Threshold or regime parameters
- Strategy identifiers
- Calibration diagnostics

`metadata` should capture provenance such as tuning window, method, and version.

**Behavioral Guarantees**

- Immutable after creation
- Deterministic application
- Safe to reuse across evaluations

---

## Applying the Policy

### `apply_ral_policy`

Apply a readiness adjustment policy to forecasts.

**Signature**

```python
apply_ral_policy(
    forecasts,
    *,
    policy,
    **kwargs,
)
```

**Description**

Transforms the input forecast series according to the rules encoded in `policy`.
The result is an adjusted forecast signal suitable for downstream evaluation.

**Parameters**

- `forecasts`  
  Forecast values prior to readiness adjustment.

- `policy`  
  Instance of `RALPolicy`.

**Returns**

- Adjusted forecast values with the same shape as the input.

---

## Ordering With Other Policies

Policy application order is semantically important:

1. **RALPolicy** – adjusts forecasts (pre-evaluation)
2. **TauPolicy** – defines service success
3. **CostRatioPolicy** – applies cost asymmetry weighting

RAL must be applied first to preserve meaning of downstream metrics.

---

## Defaults

A neutral or default RAL policy may be provided to represent **no adjustment**.
Defaults must be explicit, documented, and stable.

---

## Determinism Guarantees

RAL policy application guarantees:

- Deterministic outputs for identical inputs
- No mutation of the policy object
- No hidden randomness or adaptive behavior

These guarantees are necessary for governance.

---

## Governance and Lifecycle

RAL policies should follow a governed lifecycle:

1. Tune readiness behavior offline (see tuning API)
2. Review assumptions and diagnostics
3. Freeze into a RALPolicy artifact
4. Apply consistently in evaluation
5. Re-tune only under material execution change

RAL changes should be treated as significant decision revisions.

---

## Common Usage Patterns

- Aligning evaluation with execution capacity
- Freezing readiness assumptions for cross-team consistency
- Comparing performance under different readiness regimes

---

## Anti-Patterns

The following patterns are discouraged:

- Using RAL to “improve” evaluation scores
- Applying RAL during model training
- Re-tuning RAL automatically without review
- Mutating readiness logic inside evaluation code

---

## Stability Notes

- `RALPolicy` and `apply_ral_policy` are public APIs
- Policy semantics are stable within major versions
- New adjustment strategies may be added additively

---

## Related Documentation

- **API → Tuning → RAL**
- **How-To → Tune RAL**
- **Concepts → Policies**
- **Concepts → Artifacts and Governance**
- **Concepts → Layering and Scope**

---

*RAL policies exist to calibrate realism — and realism must be governed.*
