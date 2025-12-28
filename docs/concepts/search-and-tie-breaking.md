# Search and Tie-Breaking

This document describes the **search utilities and tie-breaking logic** used in
eb-optimization to support deterministic parameter selection.

Search and tie-breaking are foundational to ensuring that optimization outcomes are
**reproducible, explainable, and governed**.

---

## Why Search Exists in eb-optimization

Many tuning and calibration problems do not admit closed-form solutions.
Instead, parameters are selected from **finite candidate sets**.

Search in eb-optimization exists to:

- Evaluate candidate decision parameters
- Select optimal values deterministically
- Avoid hidden randomness
- Preserve reproducibility across runs

Search is a *mechanism*, not a decision-maker.

---

## Candidate-Based Optimization

eb-optimization uses **candidate-based optimization**, where:

- A finite grid of candidate values is constructed
- Each candidate is evaluated using a metric or objective
- The best candidate is selected according to explicit rules

This approach is preferred over continuous solvers because it is:

- Transparent
- Stable
- Easy to audit

---

## Grid Construction

Candidate grids are typically constructed using helper utilities such as:

- Fixed linear grids
- Logarithmic grids
- Domain-informed discrete sets

Grid construction is intentionally explicit to avoid implicit assumptions.

---

## Deterministic Selection

Given a candidate grid and evaluation scores, eb-optimization guarantees:

- Deterministic selection
- No dependence on iteration order
- No stochastic tie resolution

Identical inputs always produce identical outputs.

---

## The Need for Tie-Breaking

In practice, multiple candidates may be equally optimal under a given objective.

Without explicit tie-breaking:

- Results may depend on iteration order
- Outputs may vary across platforms
- Reproducibility is compromised

Tie-breaking logic resolves this ambiguity.

---

## Tie-Breaking Kernels

eb-optimization uses **tie-breaking kernels** to resolve equivalent candidates.

Common patterns include:

- Selecting the smallest acceptable value
- Selecting the most conservative value
- Selecting the most stable value

The choice of kernel reflects *decision philosophy*, not numerical accident.

---

## Primary APIs

Typical tie-breaking utilities include functions such as:

```python
from eb_optimization.search.kernels import argmin_over_candidates
```

These helpers ensure that:

- Selection rules are centralized
- Behavior is consistent across tuning routines
- Decision logic is reviewable

---

## Search vs Optimization

In eb-optimization:

- **Search** evaluates candidates
- **Optimization** selects parameters
- **Policies** freeze decisions

Search is never responsible for governance decisions.

---

## Relationship to Tuning

Search utilities are used internally by tuning routines to:

- Select R values
- Select τ thresholds
- Choose RAL parameters

However, search functions are generic and reusable.

---

## Best Practices

- Use coarse grids first, then refine if needed
- Prefer conservative tie-breaking rules
- Document grid choices in artifacts
- Avoid overfitting via excessively dense grids

---

## Common Pitfalls

- Relying on implicit iteration order
- Using floating-point equality without tolerance
- Mixing stochastic and deterministic selection
- Hiding decision rules inside tuning code

eb-optimization centralizes these concerns to avoid error.

---

## Governance Implications

Explicit search and tie-breaking enable:

- Transparent parameter selection
- Post-hoc explanation of decisions
- Consistent behavior across environments

Without deterministic tie-breaking, governance collapses.

---

## When to Revisit Search Logic

Search logic should evolve only when:

- Decision philosophy changes
- New classes of parameters are introduced
- Stability or interpretability requirements shift

Changes should be deliberate and reviewed.

---

## Next Steps

- See **Concepts → Tuning** for estimation workflows
- Review **How-To → Run Sensitivity Sweeps**
- Consult **API → Search** for reference documentation

---

*In Electric Barometer, ambiguity is resolved by design — not by accident.*
