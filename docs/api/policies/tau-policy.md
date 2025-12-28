# Tau Policy

This document provides the **API reference** for the service threshold policy defined in
`eb_optimization.policies.tau_policy`.

A **TauPolicy** encapsulates the service threshold **τ (tau)** as a frozen, governed decision
artifact used to define service success during evaluation.

---

## Conceptual Role

τ defines what “acceptable service” means for threshold- and hit-rate–based EB metrics.

Within the Electric Barometer ecosystem, a TauPolicy:

- Formalizes service definitions explicitly
- Separates estimation from application
- Ensures deterministic and auditable evaluation behavior
- Enables consistent interpretation across teams and time

Tau policies do not compute metrics; they parameterize service-oriented evaluation.

---

## Policy Object

### `TauPolicy`

Immutable policy object representing a service threshold τ.

**Signature**

```python
TauPolicy(
    tau,
    *,
    metadata=None,
)
```

**Parameters**

- `tau`  
  Service threshold parameter used to compute HR@τ and other threshold-sensitive metrics.

- `metadata` *(optional)*  
  Additional contextual information (e.g., estimation method, window, version, rationale).

**Behavioral Guarantees**

- Immutable after creation
- Serializable
- Safe to reuse across evaluations

---

## Defaults

eb-optimization provides a documented default τ policy intended as a safe fallback.

Defaults are:

- Explicit and reviewable
- Stable across minor releases
- Never applied silently without documentation

---

## Applying the Policy

### `apply_tau_policy`

Apply a τ policy during evaluation.

**Signature**

```python
apply_tau_policy(
    y_true,
    y_pred,
    *,
    policy,
    **kwargs,
)
```

**Description**

Evaluates threshold-sensitive and service-oriented metrics using the τ value encoded
in the provided policy.

**Parameters**

- `y_true`  
  Actual observed values.

- `y_pred`  
  Forecast values.

- `policy`  
  Instance of `TauPolicy`.

**Returns**

- Metric outputs parameterized by the policy (e.g., HR@τ and related diagnostics).

---

## Entity-Level Application

τ policies may be applied:

- Globally (single τ for all entities)
- Per entity (heterogeneous service expectations)

Entity-level policies support localized service definitions while preserving a unified evaluation framework.

---

## Ordering With Other Policies

Policy application order matters:

1. **RALPolicy** – adjusts forecasts
2. **TauPolicy** – defines service success
3. **CostRatioPolicy** – applies cost asymmetry weighting

τ must be applied after readiness adjustments to preserve meaning of service success.

---

## Determinism Guarantees

τ policy application guarantees:

- Deterministic behavior for identical inputs
- No mutation of policy state
- Stable results across runs

These guarantees support reproducible service evaluation.

---

## Governance and Lifecycle

τ policies should follow a governed lifecycle:

1. Estimate τ offline using tuning routines
2. Review achieved service behavior and diagnostics
3. Freeze into a TauPolicy artifact
4. Apply consistently in evaluation
5. Re-estimate only when service expectations change

Service definitions are decisions and should be treated accordingly.

---

## Common Usage Patterns

- Standardizing service evaluation across a forecasting system
- Segment-level τ for heterogeneous environments
- Periodic refresh aligned to operational expectation shifts

---

## Anti-Patterns

The following patterns are discouraged:

- Treating τ as a model hyperparameter
- Re-tuning τ during evaluation
- Using inconsistent τ definitions across teams
- Applying raw scalar τ values without a policy

---

## Stability Notes

- `TauPolicy` and `apply_tau_policy` are public, stable APIs
- Constructor and application semantics are stable within major versions
- Additional metadata fields may be added additively

---

## Related Documentation

- **API → Tuning → Tau**
- **How-To → Estimate τ**
- **Concepts → Policies**
- **Concepts → Artifacts and Governance**
- **How-To → Use Policies in eb-evaluation**

---

*Tau policies make service definitions explicit — and explicit definitions are governable.*
