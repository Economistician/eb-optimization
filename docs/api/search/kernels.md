# Tie-Breaking Kernels

This document provides the **API reference** for deterministic tie-breaking kernels in
`eb_optimization.search.kernels`.

Tie-breaking kernels resolve ambiguity when multiple candidate parameters are equally
optimal under a given evaluation criterion. They are essential for reproducibility and governance.

---

## Conceptual Role

In candidate-based optimization, it is common for multiple candidates to yield identical
objective values within numerical tolerance.

Tie-breaking kernels exist to:

- Resolve ambiguity deterministically
- Make selection rules explicit and reviewable
- Prevent hidden dependence on iteration order
- Preserve reproducibility across runs and environments

Kernels encode *decision philosophy*, not numerical coincidence.

---

## Public Functions

### `argmin_over_candidates`

Select a candidate corresponding to the minimum objective value with deterministic tie-breaking.

**Signature**

```python
argmin_over_candidates(
    candidates,
    scores,
    *,
    tie_breaker=None,
    **kwargs,
)
```

**Description**

Identifies the candidate(s) achieving the minimum score and applies an explicit tie-breaking
rule when multiple candidates are equivalent.

**Parameters**

- `candidates`  
  Iterable of candidate values.

- `scores`  
  Iterable of objective values corresponding to each candidate.

- `tie_breaker` *(optional)*  
  Strategy used to resolve ties (e.g., smallest value, most conservative). If not provided,
  a documented default is used.

**Returns**

- Selected candidate value.

---

### `argmax_over_candidates`

Select a candidate corresponding to the maximum objective value with deterministic tie-breaking.

**Signature**

```python
argmax_over_candidates(
    candidates,
    scores,
    *,
    tie_breaker=None,
    **kwargs,
)
```

**Description**

Identifies the candidate(s) achieving the maximum score and applies an explicit tie-breaking
rule when multiple candidates are equivalent.

**Parameters**

- `candidates`  
  Iterable of candidate values.

- `scores`  
  Iterable of objective values corresponding to each candidate.

- `tie_breaker` *(optional)*  
  Strategy used to resolve ties.

**Returns**

- Selected candidate value.

---

## Tie-Breaking Strategies

Tie-breaking strategies may include:

- Selecting the smallest acceptable value
- Selecting the largest acceptable value
- Selecting the most conservative candidate
- Selecting based on domain-specific ordering

The chosen strategy must be explicit and documented.

---

## Determinism Guarantees

Tie-breaking kernels guarantee:

- Deterministic selection for identical inputs
- Independence from iteration order
- No stochastic behavior

These guarantees are foundational to governance.

---

## Numerical Considerations

When working with floating-point scores:

- Equality should be evaluated with tolerance
- Near-equal values should be treated explicitly
- Tie-breaking should not rely on exact float equality

Kernels are designed to handle these cases safely.

---

## Relationship to Search and Tuning

Tie-breaking kernels are used by:

- Tuning routines to select parameters
- Sensitivity analysis to report stable optima
- Any candidate-based selection requiring determinism

They are generic utilities and may be reused safely.

---

## Usage Patterns

Typical usage includes:

- Resolving R candidates with identical balance scores
- Selecting τ values with equivalent service metrics
- Choosing conservative defaults when ambiguity exists

---

## Anti-Patterns

The following patterns are discouraged:

- Relying on Python’s default `min` / `max` without control
- Allowing iteration order to influence outcomes
- Using random selection to resolve ties
- Hiding tie-breaking logic inside tuning code

---

## Stability Notes

- Kernel functions documented here are public APIs
- Behavior is stable within major versions
- New tie-breaking strategies may be added additively

---

## Related Documentation

- **Concepts → Search and Tie-Breaking**
- **Concepts → Tuning**
- **API → Search → Grid**
- **How-To → Run Sensitivity Sweeps**

---

*Tie-breaking is where ambiguity ends — by design, not accident.*
