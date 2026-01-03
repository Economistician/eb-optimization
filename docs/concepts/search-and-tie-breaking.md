# Search and Tie-Breaking

This document describes the **search utilities and tie-breaking logic** used in
`eb-optimization` to support deterministic, governable parameter selection.

Search and tie-breaking are not optimization tricks — they are **decision hygiene**.
Their purpose is to ensure that parameter choices are **reproducible, explainable, and auditable**.

---

## Why Search Exists in eb-optimization

Many decision-calibration problems do not admit closed-form solutions.
Instead, parameters are selected from **finite candidate sets**.

Search exists to:

- Evaluate candidate decision parameters
- Enforce deterministic selection
- Eliminate hidden randomness
- Preserve reproducibility across runs and environments

Search is a *mechanism*, not a decision-maker.

---

## Candidate-Based Selection

eb-optimization uses **candidate-based selection**, where:

- A finite grid of candidate values is defined explicitly
- Each candidate is evaluated using a deterministic criterion
- One candidate is selected using explicit rules

This approach is preferred over continuous solvers because it is:

- Transparent
- Stable
- Easy to audit
- Resistant to numerical noise

---

## Grid Construction Philosophy

Candidate grids are always **explicit**.

Common grid construction strategies include:

- Fixed linear grids
- Logarithmic grids
- Domain-informed discrete sets

Grids are never inferred implicitly from data.

Grid design is considered a **governance decision**, not an implementation detail.

---

## Deterministic Evaluation

Given a grid and an evaluation function, eb-optimization guarantees:

- Deterministic evaluation of all candidates
- No dependence on iteration order
- No stochastic components

Identical inputs always produce identical evaluation curves.

---

## Why Tie-Breaking Is Necessary

In practice, multiple candidates may be **equally optimal** or nearly optimal.

Without explicit tie-breaking:

- Results may depend on iteration order
- Behavior may vary across platforms
- Governance becomes ambiguous

Tie-breaking logic exists to resolve this ambiguity *by design*.

---

## Tie-Breaking Kernels

eb-optimization uses **tie-breaking kernels** to resolve equivalent candidates.

Typical kernels include:

- **First / leftmost** (grid-order preference)
- **Most conservative** (risk-averse choice)
- **Stability-aware** (robust to perturbation)

The kernel reflects **decision philosophy**, not numerical accident.

---

## Kernel vs Objective

It is critical to distinguish:

- **Objective**: what is being optimized
- **Kernel**: how ties are resolved

Changing a kernel does **not** change the objective — it changes how ambiguity is handled.

This separation makes tie-breaking auditable.

---

## Stability-Aware Selection

Some tuning routines evaluate **selection stability** by perturbing:

- Candidate grids
- Pivot points
- Neighborhood structure

When multiple candidates remain viable under perturbation, diagnostics may flag
weak identifiability — but selection still remains deterministic.

Search does not *refuse* to choose; it **chooses transparently**.

---

## Relationship to Tuning

Search utilities are used internally by tuning routines to:

- Select cost ratios (R)
- Select service thresholds (τ)
- Select RAL parameters

Search utilities themselves are generic and reusable across tuning domains.

---

## Search vs Optimization

In eb-optimization:

- **Search** evaluates candidates
- **Tuning** selects parameters
- **Policies** freeze decisions

Search never performs governance and never mutates state.

---

## Best Practices

- Prefer coarse grids before refinement
- Use conservative tie-breaking by default
- Document grid choices in tuning artifacts
- Inspect stability diagnostics before freezing policies

---

## Common Pitfalls

- Relying on implicit iteration order
- Using floating-point equality without tolerance
- Mixing stochastic logic into selection
- Hiding tie-breaking inside notebooks

eb-optimization centralizes search to prevent these failures.

---

## Governance Implications

Explicit search and tie-breaking enable:

- Transparent parameter selection
- Reproducible historical analysis
- Independent audit of decision logic

Without deterministic tie-breaking, governance collapses.

---

## When to Revisit Search Logic

Search logic should change only when:

- Decision philosophy changes
- New classes of parameters are introduced
- Stability requirements evolve

Such changes should be deliberate and reviewed.

---

## Next Steps

- See **Concepts → Tuning** for estimation workflows
- Review **How-To → Sensitivity Analysis**
- Consult **API → Search** for reference documentation

---

*In Electric Barometer, ambiguity is resolved intentionally — never accidentally.*
