# Tuning

This document defines what **tuning** means within the Electric Barometer ecosystem and
how tuning is deliberately constrained in **eb-optimization**.

Tuning in Electric Barometer is **decision-parameter calibration**, not model optimization,
and not experimentation.

---

## What Tuning Is (and Is Not)

In eb-optimization, tuning is:

- The process of **selecting decision parameters from data**
- Deterministic and reproducible by construction
- Explicitly artifact-producing
- Reviewable and auditable before deployment
- Strictly separated from evaluation and scoring

Tuning is *not*:

- Forecast model training
- Hyperparameter optimization
- Metric gaming or score maximization
- Online, adaptive, or feedback-driven adjustment

This distinction is foundational to governance.

---

## Why Tuning Is a Separate Concept

In many analytics systems, tuning is conflated with modeling or evaluation.
This creates ambiguity about **where decisions are made** and **who owns them**.

Electric Barometer separates tuning so that:

- Decisions are explicit artifacts, not side effects
- Evaluation remains stable across time
- Models remain interchangeable
- Governance boundaries are enforceable

Tuning exists to make *judgment visible*.

---

## Types of Tuning in eb-optimization

eb-optimization supports several tuning domains, each intentionally narrow in scope.

### Cost Ratio Tuning (R)

Cost ratio tuning selects the asymmetric cost parameter **R = c_u / c_o** governing
under- vs over-forecasting penalties.

Characteristics:

- Balance-based (not optimization-based)
- Metric-aligned with EB loss functions
- Deterministic with explicit tie-breaking
- Produces interpretable estimation artifacts
- Surfaces identifiability and stability diagnostics

Outputs are reviewed before being frozen into a **CostRatioPolicy**.

---

### Service Threshold Tuning (τ)

Service threshold tuning selects the value of **τ** that defines service success or failure.

Characteristics:

- Directly interpretable in operational terms
- Suitable for global or entity-level application
- Deterministic and grid-based
- Produces policy-ready artifacts

τ tuning defines *what counts as service*, not how forecasts are built.

---

### Readiness Adjustment Layer (RAL) Tuning

RAL tuning calibrates how forecasts are adjusted to reflect execution readiness.

Characteristics:

- Conservative by design
- Explicitly policy-driven
- Applied pre-evaluation
- Produces frozen policy artifacts (not scalars)

RAL tuning affects decisions, not model accuracy metrics.

---

## Inputs to Tuning

Tuning routines typically consume:

- Historical actuals
- Corresponding forecasts (fixed at decision time)
- Optional entity identifiers
- Explicit candidate grids or kernels
- Optional sample weights

Tuning never mutates inputs in place.

---

## Outputs of Tuning

Tuning produces **estimation artifacts**, not raw numbers.

Artifacts include:

- Selected parameter values
- Diagnostic and stability information
- Sample size and coverage metadata
- Method and selection metadata

Artifacts are then converted into immutable policies.

---

## Determinism and Reproducibility

All tuning routines in eb-optimization guarantee:

- Deterministic outputs for identical inputs
- Explicit tie-breaking rules
- No stochastic components
- No dependence on execution order

These guarantees are mandatory for auditability.

---

## Relationship to Search

Tuning relies on search utilities to:

- Evaluate candidate parameter values
- Resolve ties deterministically
- Assess stability under perturbation

Search supports tuning but never defines policy decisions.

---

## Tuning vs Sensitivity Analysis

It is critical to distinguish:

- **Tuning**: selects and freezes decision parameters
- **Sensitivity analysis**: explores implications of those parameters

Sensitivity analysis informs review but does not create artifacts.

---

## Governance Implications

Because tuning is explicit and constrained:

- Decisions can be reviewed before deployment
- Changes can be justified and documented
- Historical evaluations remain reproducible
- Policy drift is prevented

This makes tuning suitable for high-stakes environments.

---

## Common Anti-Patterns

The following patterns violate tuning principles:

- Re-tuning parameters inside evaluation loops
- Optimizing parameters to maximize a score
- Introducing randomness into selection
- Hiding tuning logic inside notebooks
- Deploying estimates without review

eb-optimization is designed to prevent these failures.

---

## When to Re-Tune

Re-tuning is appropriate when:

- Cost structures materially change
- Service definitions are revised
- Execution readiness regimes shift
- Sufficient new data becomes available

Re-tuning should be deliberate, not automatic.

---

## Operational Cadence

Best-practice tuning cadence is:

- Periodic (e.g., quarterly)
- Triggered by regime change
- Explicitly documented

Avoid continuous or rolling re-tuning.

---

## Relationship to Other Concepts

- **Policies** freeze tuning outputs into governed artifacts
- **Search and tie-breaking** ensure deterministic selection
- **Artifacts and governance** define lifecycle and review

---

## Next Steps

- Review **How-To → Estimate Cost Ratio (R)**
- Review **How-To → Estimate τ**
- Continue with **How-To → Tune RAL**
- Consult **API → Tuning** for reference documentation

---

*Tuning exists to formalize judgment — not to chase performance.*
