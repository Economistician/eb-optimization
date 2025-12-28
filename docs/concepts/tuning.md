# Tuning

This document defines what **tuning** means within the Electric Barometer ecosystem and
how tuning is deliberately constrained in eb-optimization.

Tuning in Electric Barometer is **parameter calibration for decisions**, not model optimization
and not experimentation.

---

## What Tuning Is (and Is Not)

In eb-optimization, tuning is:

- The process of **selecting decision parameters from data**
- Deterministic and reproducible
- Governed and reviewable
- Explicitly separated from evaluation

Tuning is *not*:

- Forecast model training
- Hyperparameter optimization
- Metric experimentation
- Online or adaptive adjustment

This distinction is foundational.

---

## Why Tuning Is a Separate Concept

In many systems, tuning is conflated with modeling or evaluation.
This creates ambiguity about **where decisions are made**.

Electric Barometer separates tuning so that:

- Decisions are visible
- Evaluation remains stable
- Models remain interchangeable
- Governance is enforceable

Tuning exists to make decisions *explicit*.

---

## Types of Tuning in eb-optimization

eb-optimization supports multiple tuning domains, each with a narrow scope.

### Cost Ratio Tuning (R)

Cost ratio tuning selects the asymmetric cost parameter (R) that governs
under- vs over-forecasting penalties.

Characteristics:

- Balance-based
- Metric-aligned
- Deterministic
- Produces a single interpretable parameter

---

### Service Threshold Tuning (τ)

Service threshold tuning selects the value of τ that defines service success.

Characteristics:

- Hit-rate or metric aligned
- Interpretable in operational terms
- Suitable for global or entity-level use

---

### Readiness Adjustment Layer Tuning (RAL)

RAL tuning calibrates how forecasts are adjusted to reflect execution readiness.

Characteristics:

- Conservative by design
- Applied pre-evaluation
- Produces policy artifacts, not scalars

---

## Inputs to Tuning

Tuning routines typically consume:

- Historical actuals
- Corresponding forecasts
- Optional entity identifiers
- Explicit candidate grids or methods

Tuning never modifies inputs in place.

---

## Outputs of Tuning

Tuning produces **estimation artifacts**, which include:

- Selected parameter values
- Diagnostic information
- Method metadata

These artifacts are then converted into policies.

---

## Determinism and Reproducibility

All tuning routines in eb-optimization guarantee:

- Deterministic outputs
- No stochastic components
- No dependence on execution order

This ensures that identical inputs produce identical decisions.

---

## Relationship to Search

Tuning often relies on search utilities to:

- Evaluate candidate parameter values
- Resolve ties deterministically
- Ensure stability across runs

Search supports tuning but does not define decisions.

---

## Tuning vs Sensitivity Analysis

It is critical to distinguish:

- **Tuning**: selects parameters
- **Sensitivity analysis**: explores implications

Sensitivity analysis informs judgment but does not freeze decisions.

---

## Governance Implications

Because tuning is explicit and constrained:

- Decisions can be reviewed before deployment
- Changes can be justified and documented
- Historical analyses remain reproducible

This makes tuning compatible with high-stakes environments.

---

## Common Anti-Patterns

The following patterns violate tuning principles:

- Re-tuning during evaluation
- Optimizing parameters to maximize a score
- Introducing randomness into selection
- Hiding tuning logic inside notebooks

eb-optimization is designed to prevent these failures.

---

## When to Re-Tune

Re-tuning is appropriate when:

- Cost structures materially change
- Service definitions are revised
- Execution readiness shifts
- Sufficient new data is available

Re-tuning should be deliberate, not automatic.

---

## Operational Cadence

Best practice tuning cadence is:

- Periodic (e.g., quarterly)
- Triggered by regime change
- Documented and reviewed

Avoid continuous or rolling re-tuning.

---

## Relationship to Other Concepts

- See **Concepts → Policies** for artifact creation
- See **Concepts → Search and Tie-Breaking** for selection mechanics
- See **Artifacts and Governance** for lifecycle management

---

## Next Steps

- Review **How-To → Estimate Cost Ratio (R)**
- Review **How-To → Estimate τ**
- Continue with **How-To → Tune RAL**
- Consult **API → Tuning** for reference documentation

---

*Tuning exists to formalize judgment — not to chase performance.*
