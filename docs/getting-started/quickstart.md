# Quickstart

This guide provides a **minimal, end-to-end walkthrough** of using eb-optimization
to calibrate decision parameters and apply them within the Electric Barometer ecosystem.

The goal is to demonstrate *workflow*, not exhaustively cover all options.

---

## What You Will Do

In this quickstart, you will:

1. Load actuals and forecasts
2. Estimate a cost ratio (R)
3. Estimate a service threshold (τ)
4. Freeze parameters into policies
5. Apply policies during evaluation

This mirrors a real-world EB workflow.

---

## Prerequisites

Ensure the Electric Barometer stack is installed:

```bash
pip install eb-metrics eb-optimization eb-evaluation
```

You should have:

- Historical actual values
- Corresponding forecast values
- (Optionally) entity identifiers

---

## Step 1: Prepare Data

Assume you have NumPy arrays or pandas Series:

```python
import numpy as np

y_true = np.array([...])
y_pred = np.array([...])
```

For entity-level workflows, a DataFrame is recommended.

---

## Step 2: Estimate Cost Ratio (R)

Estimate the asymmetric cost ratio using balance-based calibration:

```python
from eb_optimization.tuning.cost_ratio import estimate_R_cost_balance

R_estimate = estimate_R_cost_balance(
    y_true=y_true,
    y_pred=y_pred,
)

R_estimate.R
```

Review diagnostics before proceeding.

---

## Step 3: Estimate Service Threshold (τ)

Estimate a service threshold using hit-rate–based calibration:

```python
from eb_optimization.tuning.tau import estimate_tau

tau_estimate = estimate_tau(
    y_true=y_true,
    y_pred=y_pred,
)

tau_estimate.tau
```

τ defines what constitutes acceptable service.

---

## Step 4: Freeze Policies

Convert estimates into immutable policy artifacts:

```python
from eb_optimization.policies.cost_ratio_policy import CostRatioPolicy
from eb_optimization.policies.tau_policy import TauPolicy

cost_ratio_policy = CostRatioPolicy(R=R_estimate.R)
tau_policy = TauPolicy(tau=tau_estimate.tau)
```

Policies should be versioned and reviewed.

---

## Step 5: Apply Policies in Evaluation

Apply policies deterministically during evaluation:

```python
from eb_optimization.policies.cost_ratio_policy import apply_cost_ratio_policy
from eb_optimization.policies.tau_policy import apply_tau_policy

metrics = apply_cost_ratio_policy(
    y_true=y_true,
    y_pred=y_pred,
    policy=cost_ratio_policy,
)

service_metrics = apply_tau_policy(
    y_true=y_true,
    y_pred=y_pred,
    policy=tau_policy,
)
```

No tuning occurs at this stage.

---

## Optional: Readiness Adjustment (RAL)

If readiness effects matter, tune and apply RAL:

```python
from eb_optimization.tuning.ral import tune_ral_policy

ral_policy = tune_ral_policy(
    df=panel_df,
    entity_col="entity",
    actual_col="y_true",
    forecast_col="y_pred",
)
```

RAL is applied *before* evaluation.

---

## Key Takeaways

- Estimation and evaluation are intentionally separated
- Policies are explicit, immutable artifacts
- Evaluation is deterministic and auditable
- The workflow scales from global to entity-level use cases

---

## Where to Go Next

- **How-To → Estimate Cost Ratio (R)**
- **How-To → Estimate τ**
- **How-To → Tune RAL**
- **How-To → Run Sensitivity Sweeps**
- **Concepts → Policies**

---

*In Electric Barometer, quick does not mean careless — it means structured.*
