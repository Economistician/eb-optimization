# Artifacts and Governance

This document describes how **decision artifacts** are created, managed, and governed
within the Electric Barometer ecosystem, with a focus on eb-optimization.

The goal is to ensure that decision logic is **explicit, reviewable, reproducible, and auditable**.

---

## What Is a Decision Artifact?

A decision artifact is a **frozen, explicit representation of a decision** that affects
how forecasts are interpreted or evaluated.

In eb-optimization, artifacts typically include:

- CostRatioPolicy
- TauPolicy
- RALPolicy
- Associated metadata and diagnostics

Artifacts are *outputs of reasoning*, not intermediate calculations.

---

## Why Artifacts Matter

In many analytics systems, critical decisions are embedded implicitly in:

- Model hyperparameters
- Notebook code paths
- Ad-hoc configuration files

Electric Barometer replaces this with **first-class artifacts** so that:

- Decisions can be reviewed independently
- Results can be reproduced exactly
- Changes can be governed deliberately

---

## Artifact Lifecycle

Decision artifacts typically move through the following lifecycle:

1. **Estimation**
   Parameters are estimated using historical data.

2. **Review**
   Diagnostics and assumptions are inspected.

3. **Freezing**
   Parameters are encapsulated into immutable policy objects.

4. **Deployment**
   Policies are applied in evaluation or production workflows.

5. **Revision**
   Artifacts are periodically re-estimated and replaced.

This lifecycle mirrors mature operational decision-making.

---

## Immutability as a Design Principle

Once created, artifacts are treated as **immutable**:

- They are not modified in place
- Changes require re-estimation and re-freezing
- Historical artifacts remain valid references

Immutability enables trustworthy comparisons across time.

---

## Versioning and Traceability

Artifacts should be versioned alongside:

- Code versions
- Data windows
- Environment metadata

Recommended practices include:

- Semantic versioning for policies
- Explicit identifiers or hashes
- Storing artifacts with evaluation outputs

This ensures full traceability from results back to decisions.

---

## Separation of Responsibilities

Governance relies on clear boundaries:

- **eb-metrics** defines *how metrics behave*
- **eb-optimization** defines *which parameters to use*
- **eb-evaluation** defines *how to apply them consistently*

No layer is allowed to silently cross these boundaries.

---

## Auditability and Reproducibility

Because artifacts are explicit:

- Evaluations can be re-run deterministically
- Historical analyses can be reconstructed
- Policy changes can be audited independently of code changes

This is critical in high-stakes or regulated environments.

---

## Human Review and Judgment

Artifacts are designed to support — not replace — human judgment.

Best practice includes:

- Reviewing diagnostics before freezing artifacts
- Documenting rationale for parameter choices
- Avoiding automatic deployment without review

Governance is strongest when decisions are intentional.

---

## Common Governance Failures

The following patterns undermine governance:

- Re-tuning parameters inside evaluation code
- Mutating policies after deployment
- Failing to version artifacts
- Treating artifacts as temporary objects

eb-optimization is designed to prevent these failure modes.

---

## Operational Integration

Artifacts integrate naturally with operational systems:

- Stored in object stores or registries
- Referenced by evaluation jobs
- Promoted across environments (dev → prod)

This enables scalable and disciplined deployment.

---

## Relationship to Sensitivity Analysis

Sensitivity sweeps complement governance by:

- Providing transparency into decision implications
- Supporting informed review
- Highlighting unstable regimes

However, sensitivity analysis does not replace formal artifact creation.

---

## When Governance Is Most Critical

Strong governance is especially important when:

- Decisions affect labor, inventory, or service levels
- Forecasting drives automated actions
- Multiple teams consume shared metrics
- Regulatory or contractual scrutiny exists

eb-optimization is designed with these contexts in mind.

---

## Next Steps

- Review **Concepts → Policies** for artifact structure
- See **How-To → Use Policies in eb-evaluation**
- Consult **Concepts → Tuning** for estimation philosophy

---

*In Electric Barometer, artifacts are how judgment becomes durable.*
