# eb-optimization

The **eb-optimization** package is the *decision and policy optimization layer* within the Electric Barometer ecosystem.

It is responsible for **estimating**, **selecting**, and **freezing** cost-sensitive decision parameters that sit between
pure metric computation and downstream evaluation or operational use.

---

## Position in the Electric Barometer Ecosystem

Electric Barometer is intentionally layered to separate concerns and ensure governance, auditability, and reuse:

- **eb-metrics**  
  Defines cost-weighted error metrics, service loss functions, and readiness statistics.

- **eb-optimization** *(this package)*  
  Determines *how* those metrics should be parameterized in practice via calibrated policies.

- **eb-evaluation**  
  Applies frozen policies to real forecasts and actuals to produce interpretable results.

This separation ensures that **decision logic is explicit, reviewable, and reproducible**, rather than hidden inside
model training or evaluation code.

---

## Core Responsibilities

eb-optimization focuses on three tightly scoped responsibilities:

### 1. Policy Definition
Policies are immutable, declarative artifacts that encode decision parameters such as:

- Cost ratio asymmetry (R)
- Service threshold parameters (τ)
- Readiness Adjustment Layer (RAL) rules

Policies are designed to be **stable contracts** that downstream systems can rely on.

### 2. Parameter Tuning
Tuning routines estimate policy parameters from historical data using principled, testable methods:

- Balance-based estimation for cost ratios
- Hit-rate–driven or service-level–driven estimation for τ
- Data-driven calibration of readiness adjustments

Importantly, *tuning is not model training* — it is **parameter selection for decision rules**.

### 3. Search & Tie-Breaking
Search utilities provide deterministic mechanisms for:

- Constructing candidate grids
- Selecting optimal parameters
- Resolving ties using explicit kernel logic

This ensures reproducibility even when multiple candidate solutions are equivalent.

---

## Package Structure

At a high level, the package is organized as:

- **policies/**  
  Policy dataclasses, defaults, and application helpers.

- **tuning/**  
  Estimation and calibration routines that produce policy parameters.

- **search/**  
  Shared utilities for candidate generation and deterministic selection.

Each submodule is documented independently in the API reference.

---

## Design Principles

The design of eb-optimization is guided by the following principles:

- **Explicitness over convenience**  
  All decision parameters are surfaced as first-class objects.

- **Reproducibility by default**  
  Identical inputs must produce identical outputs.

- **Auditability**  
  Decisions should be explainable after the fact using frozen artifacts.

- **Ecosystem composability**  
  The package is usable standalone, but designed to integrate cleanly with other EB layers.

---

## Who This Package Is For

eb-optimization is intended for:

- Data scientists designing cost-sensitive forecasting systems
- Analytics engineers operationalizing asymmetric loss functions
- Decision scientists formalizing service-level tradeoffs
- Platform teams building governed forecasting infrastructure

It is **not** intended to replace forecasting models themselves, but to sit *above* them as a decision layer.

---

## Getting Started

If you are new to eb-optimization:

1. Start with **Getting Started → Quickstart** for an end-to-end example.
2. Review **Concepts → Policies** to understand how decisions are represented.
3. Use **How-To guides** for common workflows such as estimating R or τ.
4. Refer to the **API Reference** for detailed function and class documentation.

---

## Stability & Versioning

Public APIs in eb-optimization follow semantic versioning.

- Minor versions may add new tuning strategies or policies.
- Breaking changes are reserved for major versions.
- Internal helpers are not guaranteed to be stable unless documented as public.

---

## Relationship to Real-World Operations

eb-optimization is designed to support real operational workflows:

- Policies can be estimated offline and reviewed
- Artifacts can be versioned and deployed
- Evaluation remains deterministic and explainable

This makes the package suitable for regulated or high-stakes decision environments.

---

## Further Reading

- Concepts → Layering and Scope
- Concepts → Policies
- Concepts → Tuning
- Concepts → Search and Tie-Breaking
- How-To → Using Policies in eb-evaluation

---

*Electric Barometer is an opinionated framework. eb-optimization exists to ensure those opinions are explicit, measurable, and defensible.*
