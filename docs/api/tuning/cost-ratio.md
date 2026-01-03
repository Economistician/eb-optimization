# Cost Ratio Tuning (R)

This document provides the **API reference** and conceptual framing for cost ratio tuning
functionality in `eb_optimization.tuning.cost_ratio`.

Cost ratio tuning determines the asymmetric cost parameter **R = c_u / c_o** used by
cost-weighted metrics such as CWSL. In Electric Barometer, tuning is **explicit,
deterministic, and artifact-driven**.

---

## Conceptual Role

Cost ratio tuning answers a single question:

> *Given observed forecast errors, what asymmetric cost ratio best reflects the
> operational balance between under-forecasting and over-forecasting?*

It does **not**:
- Apply costs
- Modify forecasts
- Enforce governance
- Decide how R is used downstream

Those responsibilities belong to policies.

---

## Public APIs

### `estimate_R_cost_balance`

Estimate a **global** cost ratio using balance-based criteria.

#### Signature

```python
estimate_R_cost_balance(
    y_true,
    y_pred,
    *,
    R_grid=None,
    co=1.0,
    sample_weight=None,
    return_curve=False,
    selection="curve",
)
```

#### Description

Evaluates a finite grid of candidate R values and selects the one that best balances
under-build and over-build cost contributions.

Selection is deterministic and governed by explicit tie-breaking rules.

#### Parameters

- `y_true`
  Array-like of realized outcomes.

- `y_pred`
  Array-like of forecasts aligned with `y_true`.

- `R_grid`
  Explicit candidate grid. If omitted, a documented default grid is used.

- `co`
  Over-build cost coefficient. Used only for relative balance; absolute scale cancels.

- `sample_weight`
  Optional weights applied per observation.

- `return_curve`
  If True, returns a rich artifact with full diagnostics instead of a scalar.

- `selection`
  Tie-breaking kernel used when multiple candidates are equivalent.

#### Returns

- `float` (legacy mode), or
- `CostRatioEstimate` artifact when `return_curve=True`

---

### `estimate_entity_R_from_balance`

Estimate **per-entity** cost ratios from panel data.

#### Signature

```python
estimate_entity_R_from_balance(
    df,
    *,
    entity_col,
    y_true_col,
    y_pred_col,
    ratios=None,
    co=1.0,
    sample_weight_col=None,
    return_result=False,
    selection="curve",
)
```

#### Description

Applies balance-based cost ratio estimation independently for each entity in the panel.

Entities are evaluated independently but under a shared candidate grid and selection
kernel.

#### Returns

- Legacy mode: `pd.DataFrame`
- Artifact mode: `EntityCostRatioEstimate`

---

## Estimation Artifacts

When artifact mode is enabled, tuning functions return structured objects containing:

- Selected R (`R_star`)
- Full cost curves
- Balance gaps
- Grid sensitivity diagnostics
- Identifiability signals (`is_identifiable`)

Artifacts are designed for:
- Human review
- Logging and serialization
- Downstream governance decisions

---

## Identifiability and Stability

Cost ratio tuning may be **weakly identifiable** when:

- Cost curves are flat
- Multiple grid points yield similar balance
- Small grid perturbations change the selected R

To surface this, artifacts include:

- `rel_min_gap`
- `grid_instability_log`
- `is_identifiable`

These signals:
- Do **not** change selection
- Exist solely for governance and audit

---

## Determinism Guarantees

All tuning routines guarantee:

- Deterministic outputs for identical inputs
- Explicit tie-breaking
- No stochastic behavior
- Stable results across platforms

This is critical for reproducibility.

---

## Interaction with Policies

Tuning **does not apply R**.

Instead, estimates are reviewed and frozen into policies:

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy

policy = CostRatioPolicy(R_grid=(...), co=...)
```

Policies then govern how R is applied in evaluation or production.

---

## Common Usage Patterns

- Global calibration for system-wide asymmetry
- Entity-level calibration for heterogeneous environments
- Periodic re-estimation under regime change

---

## Anti-Patterns

Avoid:

- Treating R as a model hyperparameter
- Selecting R dynamically inside evaluation
- Ignoring identifiability diagnostics
- Using raw scalar R without governance

---

## Stability Notes

- APIs are considered public and stable
- New diagnostics may be added additively
- Selection semantics are version-stable

---

## Related Documentation

- **How-To → Estimate Cost Ratio (R)**
- **Concepts → Tuning**
- **Concepts → Policies**
- **API → Policies → CostRatioPolicy**

---

*Cost ratio tuning formalizes asymmetry — it does not enforce it.*
