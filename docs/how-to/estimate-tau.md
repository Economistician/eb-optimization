# Estimating Service Threshold (τ)

This guide describes how to estimate the **service threshold parameter (τ)** used within the Electric Barometer ecosystem
to govern hit-rate–based service behavior and threshold-sensitive metrics.

In eb-optimization, τ is treated as a **service decision parameter**, calibrated from data and frozen into policy artifacts.

---

## What Is the Service Threshold (τ)?

The service threshold, denoted **τ (tau)**, defines the boundary at which a forecast is considered to have met a service
objective.

Depending on context, τ may represent:

- A tolerance band around actuals
- A service-level requirement
- A minimum acceptable hit-rate condition

τ directly influences EB metrics such as **HR@τ**, **FRS**, and other service-oriented evaluation measures.

---

## Why τ Is Estimated

Unlike symmetric error metrics, service-based metrics require an explicit definition of *what counts as success*.

Estimating τ from data allows:

- Alignment with observed operational performance
- Consistency across entities and time
- Explicit governance of service definitions

In eb-optimization, τ is never implicit — it is always **chosen deliberately or estimated transparently**.

---

## Estimation Strategies

eb-optimization provides multiple estimation strategies for τ, each suited to different operational goals.

### Hit-Rate–Driven Estimation

This strategy selects τ such that a target hit-rate or service level is achieved.

Characteristics:

- Intuitive and business-aligned
- Stable under scale changes
- Compatible with entity-level estimation

### Metric-Aligned Estimation

τ may also be estimated to optimize or balance a service-oriented metric (e.g., FRS).

This approach ensures:

- Consistency with downstream evaluation
- Deterministic selection under ties
- Clear linkage to EB loss structures

---

## Primary APIs

### Global τ Estimation

```python
from eb_optimization.tuning.tau import estimate_tau

tau_estimate = estimate_tau(
    y_true=actuals,
    y_pred=forecasts,
    method="hr_target",
)
```

The returned **TauEstimate** object includes:

- Estimated τ value
- Estimation method
- Diagnostic and convergence metadata

---

### Entity-Level τ Estimation

```python
from eb_optimization.tuning.tau import estimate_entity_tau

entity_tau = estimate_entity_tau(
    df=panel_df,
    entity_col="store_id",
    actual_col="y_true",
    forecast_col="y_pred",
)
```

Entity-level estimation supports:

- Store- or segment-specific service thresholds
- Heterogeneous service expectations
- Aggregation or clipping before policy creation

---

## Automatic τ Selection via Hit Rate

For workflows requiring minimal configuration, eb-optimization provides an automatic hit-rate–driven estimator:

```python
from eb_optimization.tuning.tau import hr_auto_tau

tau_estimate = hr_auto_tau(
    y_true=actuals,
    y_pred=forecasts,
)
```

This method prioritizes:

- Determinism
- Minimal assumptions
- Consistency across datasets

---

## Output Artifacts

τ estimation functions return structured **TauEstimate** records rather than raw values.

These artifacts are designed to:

- Capture estimation intent
- Preserve diagnostics
- Support audit and review
- Feed directly into policy construction

---

## From Estimates to Policies

Estimated τ values are converted into a **TauPolicy**:

```python
from eb_optimization.policies.tau_policy import TauPolicy

policy = TauPolicy(tau=tau_estimate.tau)
```

Separating estimation from application ensures that service definitions remain explicit and governed.

---

## Best Practices

- Estimate τ using representative operational periods
- Revisit τ when service expectations change
- Prefer entity-level τ when heterogeneity is material
- Review hit-rate diagnostics before freezing policies

---

## Common Pitfalls

- Treating τ as a model hyperparameter
- Estimating τ on already policy-adjusted forecasts
- Using inconsistent τ definitions across teams
- Ignoring small-sample instability at the entity level

---

## Relationship to Other Parameters

- τ defines **service success**
- R defines **cost asymmetry**
- RAL modifies forecasts prior to evaluation

These parameters should be calibrated with awareness of their interactions.

---

## When Not to Estimate τ

You may choose not to estimate τ when:

- Service thresholds are contractually defined
- Regulatory constraints mandate fixed tolerances
- Data volume is insufficient for stable calibration

In these cases, τ should be set explicitly via policy defaults.

---

## Next Steps

- See **Concepts → Tuning** for estimation philosophy
- See **Concepts → Policies** for governance patterns
- See **API → Tuning → Tau** for full reference documentation
- Continue with **How-To → Tune RAL** for readiness calibration

---

*In Electric Barometer, service is a definition — and definitions must be deliberate.*
