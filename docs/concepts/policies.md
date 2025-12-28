# Policies

This document defines what **policies** are within the Electric Barometer ecosystem and
explains how they function as **first-class decision artifacts** in eb-optimization.

Policies are the primary mechanism by which calibrated decisions become durable,
governed, and reusable across evaluation workflows.

---

## What Is a Policy?

A policy is an **immutable object** that encodes one or more decision parameters used
to interpret forecasts during evaluation.

In eb-optimization, policies are:

- Explicit
- Immutable
- Serializable
- Reviewable
- Deterministic in application

Policies are *not* configuration files and *not* tuning routines.

---

## Why Policies Exist

Many analytics systems embed decisions implicitly inside:

- Model training pipelines
- Metric implementations
- Notebook logic
- Ad-hoc configuration

Electric Barometer replaces this with policies so that:

- Decisions are visible
- Changes are intentional
- Results are reproducible
- Governance is enforceable

Policies make decision logic **auditable by design**.

---

## Policy Types in eb-optimization

eb-optimization defines several policy types, each governing a specific class of decisions.

### Cost Ratio Policies

**CostRatioPolicy** encodes the asymmetric cost ratio (R) used in cost-weighted metrics.

It governs:

- Relative penalty of under- vs over-forecasting
- Asymmetric loss evaluation

Cost ratio policies do not compute metrics — they parameterize them.

---

### Service Threshold Policies (τ)

**TauPolicy** defines service success thresholds used by hit-rate and service metrics.

It governs:

- What constitutes acceptable service
- Threshold-sensitive evaluation logic

τ policies formalize service definitions that would otherwise be ambiguous.

---

### Readiness Adjustment Layer Policies (RAL)

**RALPolicy** defines how forecasts are adjusted to reflect execution readiness prior
to evaluation.

It governs:

- Forecast realism
- Readiness-based adjustments
- Pre-evaluation signal modification

RAL policies are applied *before* metrics are computed.

---

## Policy Immutability

Once created, policies are treated as **immutable artifacts**:

- They are not modified in place
- Changes require re-estimation and re-freezing
- Historical policies remain valid references

Immutability ensures trustworthy historical comparison.

---

## Defaults and Fallbacks

eb-optimization provides **explicit default policies** for common use cases.

Defaults:

- Are documented and reviewable
- Serve as safe fallbacks
- Prevent implicit behavior

Fallback logic must always be explicit — never silent.

---

## Entity-Level Policies

Policies may be defined:

- Globally (one policy for all entities)
- Per-entity (heterogeneous policies)

Entity-level policies enable localized decision-making while preserving
a unified evaluation framework.

---

## From Estimation to Policy

Policies are typically constructed from estimation artifacts:

1. Estimate parameters using tuning routines
2. Review diagnostics and assumptions
3. Instantiate policy objects
4. Version and store the policy

This separation ensures clarity and governance.

---

## Applying Policies

Policies are applied **only** in the evaluation layer.

eb-optimization never applies policies to produce final metrics.

This ensures:

- Clean separation of responsibilities
- Deterministic evaluation
- No hidden side effects

---

## Policy Ordering

When multiple policies are used together, application order matters:

1. **RALPolicy** – adjusts forecasts
2. **TauPolicy** – defines service success
3. **CostRatioPolicy** – applies cost asymmetry

This ordering preserves semantic correctness.

---

## Governance and Review

Policies are designed to support governance workflows:

- Human review before deployment
- Versioned promotion across environments
- Explicit rollback paths

Policies make decision-making operationally safe.

---

## Common Misuses

The following patterns violate policy design:

- Mutating policy attributes at runtime
- Re-estimating parameters during evaluation
- Using policies as configuration toggles
- Treating policies as transient objects

eb-optimization is designed to prevent these failures.

---

## Relationship to Other Concepts

- See **Concepts → Artifacts and Governance** for lifecycle management
- See **Concepts → Tuning** for estimation philosophy
- See **Layering and Scope** for architectural boundaries

---

## When Policies Are Not Enough

In rare cases, additional governance mechanisms may be required:

- External approval workflows
- Regulatory documentation
- Business rule registries

Policies are designed to integrate with these systems, not replace them.

---

## Next Steps

- Review **How-To → Use Policies in eb-evaluation**
- See **API → Policies** for reference documentation
- Continue with **Concepts → Tuning**

---

*In Electric Barometer, policies are how intent becomes executable.*
