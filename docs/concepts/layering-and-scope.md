# Layering and Scope

This document explains the **layered architecture** of the Electric Barometer ecosystem
and clarifies the **scope and responsibilities** of eb-optimization within that system.

Clear layering is a deliberate design choice to ensure correctness, governance, and long-term scalability.

---

## Why Layering Exists

In many analytics systems, responsibilities are blurred:

- Metrics embed decision logic
- Models encode business rules
- Evaluation mutates parameters implicitly

Electric Barometer rejects this approach.

Instead, it enforces **strict separation of concerns**, where each layer has a single, well-defined role.

---

## The Electric Barometer Layers

At a high level, the ecosystem consists of three primary layers:

1. **Metric Definition**
2. **Decision Optimization**
3. **Evaluation and Application**

Each layer depends on the one below it — never the reverse.

---

## eb-metrics: Metric Definition Layer

**eb-metrics** defines *what* is being measured.

Its responsibilities include:

- Cost-weighted loss functions
- Service-oriented metrics
- Readiness and hit-rate statistics
- Mathematical definitions and guarantees

eb-metrics contains **no decision-making** and **no tuning logic**.

It answers the question:

> *“Given inputs and parameters, how is a metric computed?”*

---

## eb-optimization: Decision Optimization Layer

**eb-optimization** defines *which parameters should be used*.

Its responsibilities include:

- Estimating decision parameters from data
- Selecting parameters using deterministic rules
- Freezing decisions into immutable artifacts
- Providing governed defaults and policies

eb-optimization does **not** compute metrics directly and does **not** evaluate forecasts.

It answers the question:

> *“What values should govern metric behavior?”*

---

## eb-evaluation: Evaluation and Application Layer

**eb-evaluation** defines *how decisions are applied*.

Its responsibilities include:

- Applying frozen policies to forecasts
- Producing interpretable evaluation outputs
- Preserving determinism and reproducibility

eb-evaluation never estimates or tunes parameters.

It answers the question:

> *“How do we apply decisions consistently?”*

---

## Dependency Direction

The dependency flow is strictly one-way:

```
eb-metrics → eb-optimization → eb-evaluation
```

Rules enforced by design:

- Lower layers are unaware of higher layers
- No circular dependencies
- No silent parameter mutation

This enables safe reuse and independent evolution.

---

## What Belongs in eb-optimization

The following belong **inside** eb-optimization:

- Parameter estimation routines
- Candidate search utilities
- Policy dataclasses and defaults
- Deterministic tie-breaking logic
- Sensitivity analysis helpers

---

## What Does NOT Belong in eb-optimization

The following are explicitly out of scope:

- Forecast generation or modeling
- Metric math definitions
- Ad-hoc evaluation logic
- Production orchestration code

Keeping eb-optimization narrow is intentional.

---

## Benefits of This Separation

This architecture enables:

- Clear governance boundaries
- Reproducible historical analysis
- Easier auditing and compliance
- Safer experimentation
- Scalable team ownership

Each layer can be reasoned about independently.

---

## Operational Implications

In production systems:

- Models generate forecasts
- eb-optimization artifacts are curated offline
- eb-evaluation applies decisions deterministically
- Metrics remain stable and interpretable

This reduces operational risk.

---

## Common Anti-Patterns

The following patterns violate layering:

- Estimating parameters during evaluation
- Embedding business rules inside metrics
- Modifying policies at runtime
- Using metrics to imply decisions

eb-optimization exists to prevent these failures.

---

## Layering and Governance

Strict layering is what makes governance possible:

- Decisions are explicit
- Changes are reviewable
- Responsibility is assignable

Without layering, governance collapses.

---

## When to Revisit Layer Boundaries

Layer boundaries should be reconsidered only when:

- A responsibility no longer fits clearly
- New classes of decisions emerge
- Operational requirements fundamentally change

Ad-hoc exceptions should be avoided.

---

## Next Steps

- Review **Concepts → Policies** for decision artifacts
- See **Concepts → Tuning** for estimation philosophy
- Continue with **Getting Started → Quickstart**

---

*Layering is not bureaucracy — it is how complexity stays manageable.*
