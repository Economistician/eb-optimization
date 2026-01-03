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

In many forecasting systems, asymmetry parameters are selected heuristically or embedded implicitly.
eb-optimization instead provides **data-driven estimation methods** that:

- Reflect observed service behavior
- Preserve interpretability
- Are deterministic and reproducible
- Surface stability and identifiability diagnostics
- Can be reviewed, audited, and re-estimated over time

This allows R to function as a *governed decision parameter* rather than a hidden tuning constant.

---

## Available Estimation Methods

eb-optimization currently provides **balance-based estimation** strategies.

### Balance-Based Estimation

Balance-based estimation selects R such that opposing service errors are balanced according to a target condition
(e.g., equalized cost-weighted service loss).

Key properties:

- Metric-aligned (directly tied to EB loss functions)
- Deterministic and grid-based
- Robust to scale differences across entities
- Produces explicit diagnostic artifacts

---

## Primary APIs

### Global Cost Ratio Estimation

```python
from eb_optimization.tuning.cost_ratio import estimate_R_cost_balance

est = estimate_R_cost_balance(
    y_true=actuals,
    y_pred=forecasts,
    return_curve=True,
)
```

This returns a **CostRatioEstimate** artifact containing:

- `R_star` — selected cost ratio
- `curve` — evaluated cost-balance curve over the grid
- `rel_min_gap` — relative imbalance at the optimum
- `grid_instability_log` — sensitivity to grid perturbation
- `is_identifiable` — boolean summary flag
- Additional method and diagnostic metadata

The returned artifact is intended for **inspection and review**, not direct application.

---

### Entity-Level Cost Ratio Estimation

```python
from eb_optimization.tuning.cost_ratio import estimate_entity_R_from_balance

art = estimate_entity_R_from_balance(
    df=panel_df,
    entity_col="store_id",
    y_true_col="actual",
    y_pred_col="forecast",
    return_result=True,
)
```

Entity-level estimation produces an **EntityCostRatioEstimate** artifact with:

- One calibrated R per eligible entity
- Per-entity cost curves
- Per-entity diagnostics
- Explicit handling of small-sample entities

This enables heterogeneous cost modeling while preserving governance.

---

## Identifiability and Stability Diagnostics

Cost ratio estimation may be **weakly identifiable** when:

- Cost curves are flat near the optimum
- Multiple grid points achieve similar balance
- Small grid perturbations change the selected R

To surface this, estimation artifacts include diagnostics such as:

- `rel_min_gap`
- `grid_instability_log`
- `R_min` / `R_max`
- `is_identifiable`

These diagnostics:

- **Do not alter the selected R**
- Exist purely for governance, reporting, and audit
- Are later surfaced again at the policy boundary

---

## Output Artifacts

Estimation functions return **structured artifacts**, not raw scalars.

Artifacts typically include:

- Selected R values
- Cost curves
- Diagnostic summaries
- Sample size metadata
- Method identifiers

Artifacts are designed to be:

- Logged
- Reviewed
- Serialized
- Converted into frozen policies

---

## From Estimates to Policies

Estimates are *not* applied directly.

Instead, reviewed estimates are frozen into a **CostRatioPolicy**:

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy

policy = CostRatioPolicy(
    R_grid=(0.5, 1.0, 2.0, 3.0),
    co=1.0,
)
```

The policy governs **application**, while tuning governs **estimation**.

---

## Best Practices

- Inspect curves and diagnostics before freezing policies
- Treat non-identifiable estimates as governance signals
- Prefer entity-level estimation when data volume permits
- Re-estimate on a fixed cadence or under regime change
- Version and document frozen policies

---

## Common Pitfalls

- Treating R as a model hyperparameter
- Estimating R on already policy-adjusted forecasts
- Ignoring identifiability warnings
- Applying raw scalars instead of policies
- Re-estimating R inside evaluation code

---

## When Not to Estimate R

You may choose not to estimate R when:

- Business costs are explicitly defined externally
- Regulatory or contractual constraints dictate asymmetry
- Data volume is insufficient for stable estimation

In these cases, R should be fixed via an explicit policy.

---

## Relationship to Other Parameters

- **R** governs cost asymmetry
- **τ** governs service thresholds
- **RAL** governs readiness adjustment

These parameters are independent but often reviewed jointly.

---

## Next Steps

- Review **Concepts → Policies** for application mechanics
- Review **Concepts → Tuning** for estimation philosophy
- See **API → Tuning → Cost Ratio** for full reference details

---

*In Electric Barometer, cost asymmetry is a decision — and decisions deserve diagnostics.*
