# Using Policies in eb-evaluation

This guide describes how to **apply policies produced by eb-optimization** within the
**eb-evaluation** layer to generate deterministic, interpretable evaluation results.

The separation between optimization and evaluation is a core design principle of the
Electric Barometer ecosystem.

---

## Purpose of Policy Application

Policies encode **frozen decision parameters** that govern how forecasts are interpreted
during evaluation.

By applying policies explicitly:

- Evaluation becomes reproducible
- Decision logic is auditable
- Results remain stable across reruns
- Governance is enforced across teams and systems

eb-evaluation never estimates or tunes parameters — it only *applies* them.

---

## Policy Types Used in Evaluation

The following policy types are commonly applied during evaluation:

- **CostRatioPolicy** – governs asymmetric cost weighting (R)
- **TauPolicy** – defines service success thresholds (τ)
- **RALPolicy** – adjusts forecasts based on readiness assumptions

Each policy type is applied independently and deterministically.

---

## Typical Evaluation Workflow

A standard Electric Barometer evaluation workflow looks like:

1. Generate or load raw forecasts
2. Tune parameters using eb-optimization (offline)
3. Freeze parameters into policies
4. Apply policies during evaluation
5. Compute metrics using eb-metrics

This workflow ensures that evaluation reflects **decisions**, not experimentation.

---

## Applying Cost Ratio Policies

Cost ratio policies control asymmetric loss behavior.

```python
from eb_optimization.policies.cost_ratio_policy import apply_cost_ratio_policy

metrics = apply_cost_ratio_policy(
    y_true=actuals,
    y_pred=forecasts,
    policy=cost_ratio_policy,
)
```

Entity-specific policies may also be applied when available.

---

## Applying Service Threshold Policies (τ)

Service threshold policies define what constitutes acceptable service.

```python
from eb_optimization.policies.tau_policy import apply_tau_policy

metrics = apply_tau_policy(
    y_true=actuals,
    y_pred=forecasts,
    policy=tau_policy,
)
```

τ policies are often used in conjunction with hit-rate–based metrics.

---

## Applying Readiness Adjustment Layer (RAL)

RAL policies modify forecasts *before* metric computation.

```python
from eb_optimization.policies.ral_policy import apply_ral_policy

adjusted_forecasts = apply_ral_policy(
    forecasts=forecasts,
    policy=ral_policy,
)
```

The adjusted forecasts are then passed into evaluation metrics.

---

## Entity-Level Policy Application

When entity-specific policies are available:

- Policies are applied per entity
- Evaluation remains vectorized and deterministic
- Missing entities fall back to default policies when defined

This supports heterogeneous decision regimes without special-case logic.

---

## Policy Ordering and Composition

Policy application follows a consistent order:

1. **RAL** – forecast adjustment
2. **τ** – service thresholding
3. **R** – cost asymmetry weighting

Maintaining this order preserves semantic correctness.

---

## Best Practices

- Always version policies alongside evaluation outputs
- Avoid applying tuning logic during evaluation
- Use defaults explicitly rather than implicitly
- Log policy identifiers with metrics

---

## Common Pitfalls

- Mixing tuning and evaluation responsibilities
- Applying policies inconsistently across datasets
- Mutating policies after deployment
- Recomputing parameters inside evaluation pipelines

---

## Governance and Auditability

Because policies are explicit artifacts:

- Evaluation results can be traced back to decisions
- Historical results can be reproduced
- Policy changes can be reviewed independently

This separation is critical in high-stakes or regulated environments.

---

## Relationship to Other Layers

- **eb-metrics** defines how metrics are computed
- **eb-optimization** defines *which parameters to use*
- **eb-evaluation** applies those parameters consistently

Each layer is intentionally constrained in scope.

---

## Next Steps

- Review **Concepts → Policies** for design philosophy
- See **How-To → Estimate R** and **Estimate τ**
- Consult **API → Policies** for reference documentation

---

*In Electric Barometer, evaluation is execution — not experimentation.*
