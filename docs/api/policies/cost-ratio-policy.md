# Cost Ratio Policy

This document provides the **API reference** for the cost ratio policy defined in
`eb_optimization.policies.cost_ratio_policy`.

A **CostRatioPolicy** encapsulates the asymmetric cost *governance* around the
underbuild-to-overbuild cost ratio **R = c_u / c_o**. It does **not** estimate R;
it applies a *frozen, reviewed configuration* produced by tuning routines.

---

## Conceptual Role

Cost ratio policies formalize **cost asymmetry** between over-forecasting and
under-forecasting.

Within the Electric Barometer ecosystem, a CostRatioPolicy:

- Encodes a reviewed and approved *candidate grid* and defaults
- Separates **estimation (tuning)** from **application (policy)**
- Ensures deterministic, auditable evaluation behavior
- Surfaces *identifiability diagnostics* at the policy boundary

The policy does not compute metrics; it governs how asymmetric costs are applied.

---

## Policy Object

### `CostRatioPolicy`

Immutable policy object representing governed cost-ratio configuration.

### Signature

```python
CostRatioPolicy(
    R_grid=(0.5, 1.0, 2.0, 3.0),
    co=1.0,
    min_n=30,
)
```

### Parameters

- `R_grid : Sequence[float]`
  Candidate ratios to search during calibration. Only strictly positive values
  are permitted. Grid order defines tie-breaking behavior.

- `co : float`
  Default overbuild cost coefficient. Underbuild cost is derived as
  `c_u = R * c_o`.

- `min_n : int`
  Minimum number of observations required for entity-level calibration.
  Entities failing this threshold are deterministically excluded.

### Behavioral Guarantees

- Immutable after creation
- Serializable and versionable
- Safe to reuse across evaluations
- No implicit defaults beyond explicit constructor values

---

## Defaults

`eb-optimization` provides a documented default policy:

```python
DEFAULT_COST_RATIO_POLICY = CostRatioPolicy()
```

Defaults are:

- Explicit
- Reviewable
- Intended as safe fallbacks

Implicit or hidden defaults are intentionally avoided.

---

## Applying the Policy (Global)

### `apply_cost_ratio_policy`

Apply a frozen cost-ratio policy to produce a **single global R**.

### Signature

```python
apply_cost_ratio_policy(
    y_true,
    y_pred,
    *,
    policy=DEFAULT_COST_RATIO_POLICY,
    co=None,
    sample_weight=None,
)
```

### Description

Applies the policy-governed cost-ratio calibration procedure and returns:

- The selected cost ratio `R`
- A diagnostics dictionary describing how the decision was made

### Returns

```python
(R: float, diagnostics: dict[str, Any])
```

Diagnostics include:

- `method`
- `R_grid`
- Whether default `co` was used
- Whether `co` was provided as an array
- Identifiability and stability signals (if available)

---

## Entity-Level Application

### `apply_entity_cost_ratio_policy`

Apply the policy per entity, subject to governance constraints.

### Key Behaviors

- Entities with fewer than `min_n` observations are excluded deterministically
- Eligible entities receive an independently calibrated `R`
- Reasons for exclusion are surfaced explicitly
- Diagnostics columns may be included or omitted

### Determinism

- Entity inclusion is deterministic
- Grid order governs tie-breaking
- No mutation of policy state occurs

---

## Identifiability & Stability Diagnostics

Cost ratio calibration may be **weakly identifiable** when:

- Cost curves are flat
- Multiple grid points achieve similar balance
- Results are sensitive to grid perturbation

To surface this, calibration diagnostics include:

- `rel_min_gap`
  Relative imbalance at the chosen point

- `grid_instability_log`
  Log spread of selected R under grid perturbations

- `is_identifiable`
  Boolean summary derived from conservative thresholds

These diagnostics:

- **Do not change selection**
- Exist purely for governance, audit, and reporting

---

## Governance and Lifecycle

Recommended lifecycle:

1. Estimate R using tuning routines
2. Review cost curves and diagnostics
3. Assess identifiability
4. Instantiate and version a policy
5. Apply consistently in evaluation
6. Re-estimate periodically as conditions change

---

## Common Usage Patterns

- System-wide cost asymmetry calibration
- Entity-level asymmetric cost modeling
- Stable, repeatable evaluation across time windows

---

## Anti-Patterns

The following are discouraged:

- Re-estimating R inside evaluation loops
- Treating R as a model hyperparameter
- Modifying R dynamically at runtime
- Using raw scalar R values without a policy

---

## Stability Notes

- `CostRatioPolicy` is a public, stable API
- Semantics are stable within major versions
- Additive diagnostics may appear over time

---

## Related Documentation

- **API → Tuning → Cost Ratio**
- **How-To → Estimate Cost Ratio (R)**
- **Concepts → Policies**
- **Concepts → Artifacts and Governance**

---

*Cost ratio policies make asymmetry explicit — and explicit decisions are governable.*
