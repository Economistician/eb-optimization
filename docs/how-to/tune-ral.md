# Tuning the Readiness Adjustment Layer (RAL)

This guide describes how to **tune the Readiness Adjustment Layer (RAL)** in eb-optimization.
RAL tuning calibrates how forecasts are adjusted based on readiness or service signals *prior*
to evaluation, enabling more realistic assessment of operational performance.

RAL is a **pre-evaluation decision layer**, not a forecasting model and not a post-hoc metric.

---

## What Is the Readiness Adjustment Layer (RAL)?

The Readiness Adjustment Layer (RAL) modifies forecasts to reflect operational readiness constraints,
such as staffing, process reliability, or execution quality.

Conceptually, RAL answers the question:

> *“Given the forecast, how much of this demand can the system realistically serve?”*

RAL operates **upstream of evaluation**, ensuring that cost and service metrics are computed against
*achievable* outcomes rather than theoretical forecasts.

---

## Why RAL Is Tuned

Unlike cost asymmetry (R) or service thresholds (τ), RAL directly alters the forecast signal.
As such, its calibration must be:

- Explicit
- Conservative
- Justifiable from data

Tuning RAL allows you to:

- Align evaluation with real execution capacity
- Prevent systematic over-crediting of forecasts
- Encode readiness assumptions as governed artifacts

---

## Conceptual Role in the EB Stack

Within the Electric Barometer ecosystem:

- **eb-metrics** computes loss and service metrics
- **eb-optimization** tunes and freezes RAL behavior
- **eb-evaluation** applies RAL-adjusted forecasts deterministically

RAL ensures that evaluation reflects *what could reasonably have happened*.

---

## Tuning Strategy Overview

RAL tuning typically involves:

1. Defining candidate readiness adjustment rules
2. Evaluating metric behavior under each rule
3. Selecting a stable, interpretable adjustment
4. Freezing the result into a RALPolicy

The goal is not to maximize a metric, but to **calibrate realism**.

---

## Primary API

### Tuning a RAL Policy

```python
from eb_optimization.tuning.ral import tune_ral_policy

ral_policy = tune_ral_policy(
    df=panel_df,
    entity_col="store_id",
    actual_col="y_true",
    forecast_col="y_pred",
)
```

The returned **RALPolicy** encapsulates:

- Adjustment parameters
- Readiness assumptions
- Metadata describing the tuning context

---

## Input Data Requirements

RAL tuning typically expects:

- Panel data with consistent entities
- Forecasts prior to any readiness adjustment
- Actual outcomes reflecting realized service

High-quality readiness signals improve tuning stability but are not strictly required.

---

## Output Artifacts

RAL tuning produces a **policy artifact**, not a scalar value.

This artifact is designed to be:

- Reviewed before deployment
- Versioned alongside code
- Applied consistently across evaluations

RAL policies should be treated as long-lived decision objects.

---

## Applying RAL Policies

Once tuned, RAL policies are applied automatically during evaluation:

```python
from eb_optimization.policies.ral_policy import apply_ral_policy

adjusted_forecasts = apply_ral_policy(
    forecasts=y_pred,
    policy=ral_policy,
)
```

This ensures consistent application across datasets and time periods.

---

## Best Practices

- Tune RAL on representative operational periods
- Prefer conservative adjustments over aggressive corrections
- Revisit RAL when execution processes materially change
- Document assumptions underlying readiness signals

---

## Common Pitfalls

- Treating RAL as a performance booster
- Tuning RAL on already-adjusted forecasts
- Overfitting readiness rules to short time windows
- Allowing RAL logic to drift without governance

---

## Relationship to Other Parameters

- R governs **cost asymmetry**
- τ governs **service success**
- RAL governs **forecast realism**

All three should be calibrated with awareness of their interactions.

---

## When Not to Use RAL

RAL may be unnecessary when:

- Forecasts already incorporate execution constraints
- Evaluation explicitly measures theoretical demand
- Readiness is constant and well-understood

In such cases, a neutral RAL policy may be appropriate.

---

## Next Steps

- See **Concepts → Policies** for governance patterns
- See **How-To → Using Policies in eb-evaluation**
- See **API → Policies → RALPolicy** for reference documentation
- Review **Run Sensitivity Sweeps** to validate RAL effects

---

*RAL exists to protect evaluation integrity — not to improve scores.*
