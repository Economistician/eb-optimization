# Running Sensitivity Sweeps

This guide describes how to run **sensitivity sweeps** in eb-optimization to understand how decision parameters
(such as cost ratio R or service threshold τ) affect cost-weighted and service-oriented metrics.

Sensitivity analysis is a **diagnostic and validation tool**, not a tuning mechanism.

---

## What Is a Sensitivity Sweep?

A sensitivity sweep evaluates one or more metrics across a **range of candidate parameter values** while holding
all other inputs fixed.

In eb-optimization, sensitivity sweeps are used to:

- Visualize metric behavior across decision regimes
- Identify stability regions
- Detect sharp inflection points
- Validate tuned parameters before freezing policies

They are intentionally **read-only** and do not modify policies.

---

## When to Use Sensitivity Sweeps

Sensitivity sweeps are most useful:

- Before freezing a newly estimated policy
- When comparing alternative tuning strategies
- When diagnosing unstable or surprising metric behavior
- During stakeholder review or documentation

They are *not* intended to replace formal tuning or estimation routines.

---

## Available Sensitivity Utilities

eb-optimization provides sensitivity helpers aligned with EB metrics.

### CWSL Sensitivity

The primary sensitivity utility evaluates **Cost-Weighted Service Loss (CWSL)** across a range of R values.

This allows you to inspect how asymmetric cost assumptions impact loss behavior.

---

## Primary APIs

### Scalar Sensitivity Sweep

```python
from eb_optimization.tuning.sensitivity import cwsl_sensitivity

results = cwsl_sensitivity(
    y_true=actuals,
    y_pred=forecasts,
    R_grid=[0.5, 1.0, 2.0, 5.0],
)
```

The result contains:

- Candidate R values
- Corresponding metric values
- Structured output suitable for plotting or inspection

---

### DataFrame-Based Sensitivity Sweep

For panel or entity-level data:

```python
from eb_optimization.tuning.sensitivity import compute_cwsl_sensitivity_df

df_sensitivity = compute_cwsl_sensitivity_df(
    df=panel_df,
    entity_col="store_id",
    actual_col="y_true",
    forecast_col="y_pred",
    R_grid=[0.5, 1.0, 2.0, 5.0],
)
```

This produces a tidy DataFrame with:

- One row per entity per candidate value
- Metric outputs suitable for aggregation
- Compatibility with downstream visualization tools

---

## Interpreting Results

When reviewing sensitivity output, look for:

- **Flat regions** indicating parameter robustness
- **Sharp slopes** indicating high sensitivity
- **Multiple local minima** suggesting ambiguity
- **Consistent optima across entities** supporting global policies

Sensitivity analysis should inform *judgment*, not dictate decisions.

---

## From Sensitivity to Decisions

Sensitivity sweeps are often used alongside tuning:

1. Run estimation to obtain a candidate parameter
2. Run sensitivity sweep around that value
3. Validate stability and interpretability
4. Freeze the parameter into a policy

This workflow preserves rigor without overfitting.

---

## Best Practices

- Use logarithmic grids for wide parameter ranges
- Center sweeps around estimated values
- Avoid interpreting noisy regions with limited data
- Separate exploratory sweeps from production workflows

---

## Common Pitfalls

- Selecting parameters directly from sensitivity minima
- Running sweeps on already policy-adjusted forecasts
- Over-interpreting minor metric differences
- Ignoring entity-level heterogeneity

---

## Relationship to Tuning

- **Tuning** selects parameters
- **Sensitivity** evaluates behavior

They serve complementary but distinct purposes.

---

## Operational Considerations

Sensitivity sweeps are:

- Deterministic
- Side-effect free
- Safe to run interactively or in notebooks

They should not be embedded in automated pipelines that freeze policies.

---

## Next Steps

- See **How-To → Estimate Cost Ratio (R)**
- See **How-To → Estimate τ**
- See **Concepts → Tuning** for methodology
- See **API → Tuning → Sensitivity** for reference documentation

---

*Sensitivity analysis does not choose for you — it reveals what your choices imply.*
