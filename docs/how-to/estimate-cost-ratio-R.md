# Estimating Cost Ratio (R)

This guide describes how to estimate the **cost ratio parameter (R)** used throughout the Electric Barometer ecosystem
to represent asymmetric costs between over-forecasting and under-forecasting.

In eb-optimization, cost ratio estimation is treated as a **decision calibration problem**, not a modeling task.

---

## What Is the Cost Ratio (R)?

The cost ratio, typically denoted **R**, defines the relative penalty of *under-forecasting* compared to *over-forecasting*
(or vice versa, depending on convention).

Conceptually:

- **R > 1** indicates under-forecasting is more costly than over-forecasting
- **R = 1** reduces to a symmetric error regime
- **R < 1** indicates over-forecasting is more costly

R directly parameterizes asymmetric loss metrics such as **CWSL (Cost-Weighted Service Loss)** and related EB metrics.

---

## Why R Is Estimated (Not Chosen Arbitrarily)

In many forecasting systems, asymmetry parameters are selected heuristically.
eb-optimization instead provides **data-driven estimation methods** that:

- Reflect observed service behavior
- Preserve interpretability
- Can be reproduced, audited, and re-estimated over time

This allows R to function as a *governed decision parameter* rather than a hidden tuning constant.

---

## Available Estimation Methods

eb-optimization currently provides **balance-based estimation** strategies.

### Balance-Based Estimation

Balance-based estimation selects R such that opposing service errors are balanced according to a target condition
(e.g., equalized service loss contributions).

This approach is:

- Metric-aligned (directly tied to EB loss functions)
- Deterministic
- Robust to scale differences across entities

---

## Primary APIs

### Global Cost Ratio Estimation

```python
from eb_optimization.tuning.cost_ratio import estimate_R_cost_balance

R_estimate = estimate_R_cost_balance(
    y_true=actuals,
    y_pred=forecasts,
)
```

This returns a **CostRatioEstimate** object containing:

- The estimated R value
- Diagnostic information about the balance condition
- Metadata about the estimation procedure

---

### Entity-Level Cost Ratio Estimation

```python
from eb_optimization.tuning.cost_ratio import estimate_entity_R_from_balance

entity_estimates = estimate_entity_R_from_balance(
    df=panel_df,
    entity_col="store_id",
    actual_col="y_true",
    forecast_col="y_pred",
)
```

Entity-level estimation produces one estimate per entity, enabling:

- Heterogeneous cost sensitivity
- Segment-level or store-level policies
- Downstream aggregation or clipping

---

## Output Artifacts

Estimation functions return structured records (not raw scalars), typically including:

- Estimated R value
- Convergence or balance diagnostics
- Sample size and coverage metadata

These artifacts are intended to be:

- Logged
- Reviewed
- Serialized if needed
- Used to construct frozen policies

---

## From Estimates to Policies

Estimates are *not* applied directly.

Instead, estimated values are used to construct a **CostRatioPolicy**:

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy

policy = CostRatioPolicy(R=R_estimate.R)
```

This separation ensures that:

- Estimation logic is isolated from application logic
- Policies can be versioned and governed independently

---

## Best Practices

- Re-estimate R on a fixed cadence (e.g., quarterly)
- Avoid mixing estimation windows across regime changes
- Inspect diagnostics before freezing policies
- Prefer entity-level estimation when sufficient data exists

---

## Common Pitfalls

- Treating R as a hyperparameter during model training
- Estimating R on already policy-adjusted forecasts
- Ignoring sample size effects for small entities
- Applying raw estimates without review or clipping

---

## Relationship to Other Parameters

- R governs **cost asymmetry**
- τ governs **service thresholds**
- RAL governs **readiness adjustments**

These parameters are independent but often calibrated jointly.

---

## When Not to Estimate R

You may choose not to estimate R when:

- Business costs are explicitly defined externally
- Regulatory or contractual constraints dictate asymmetry
- Data volume is insufficient for stable estimation

In these cases, R may be fixed via policy defaults.

---

## Next Steps

- See **Concepts → Policies** for how R is applied
- See **How-To → Using Policies in eb-evaluation**
- See **API → Tuning → Cost Ratio** for full reference details

---

*In Electric Barometer, cost asymmetry is a decision — and decisions deserve data.*
