# Candidate Grids

This document provides the **API reference** for candidate grid utilities in
`eb_optimization.search.grid`.

Candidate grids are used throughout eb-optimization to support transparent,
deterministic parameter selection.

---

## Conceptual Role

Many decision parameters in Electric Barometer are selected from finite candidate sets.
Grids provide a standardized, explicit way to construct those candidate sets so that:

- Assumptions are visible
- Selection is reproducible
- Search behavior is stable across environments

Grids are mechanisms — they do not decide outcomes.

---

## Public Functions

### `make_float_grid`

Construct a floating-point candidate grid.

**Signature**

```python
make_float_grid(
    start,
    stop,
    *,
    num=None,
    step=None,
    include_endpoints=True,
    **kwargs,
)
```

**Description**

Generates a finite list of candidate float values over a specified range. The grid may be
constructed using a fixed number of points (`num`) or a fixed spacing (`step`), depending
on the implementation and arguments provided.

**Parameters**

- `start`  
  Lower bound of the grid.

- `stop`  
  Upper bound of the grid.

- `num` *(optional)*  
  Number of points to generate.

- `step` *(optional)*  
  Step size between points.

- `include_endpoints` *(optional)*  
  Whether to include `start` and `stop` explicitly.

**Returns**

- `list[float]` (or equivalent iterable) containing candidate values.

---

## Determinism Guarantees

Grid generation guarantees:

- Deterministic output for identical inputs
- Stable ordering of candidates
- No random sampling

Grids must be safe for repeated use in tuning pipelines.

---

## Numerical Considerations

Because grids may contain floating point values:

- Treat candidate equality with tolerance when appropriate
- Prefer coarse grids for exploration
- Refine around stable regions if needed

Avoid over-dense grids that provide false precision.

---

## Usage Patterns

Grids are typically used to:

- Provide candidate R values for cost ratio tuning
- Provide candidate τ values for service threshold tuning
- Support sensitivity sweeps and stability checks

They may be used directly or indirectly via tuning routines.

---

## Defaults and Governance

When grids are not provided by the user:

- Tuning routines should use documented defaults
- Default grids should be stable across minor versions
- Changes should be reviewed and communicated

Grid defaults are governance-relevant decisions.

---

## Anti-Patterns

The following patterns are discouraged:

- Generating grids implicitly without documentation
- Using random candidate sampling for tuning
- Using excessively dense grids to “optimize” scores
- Hiding grid construction inside notebooks

---

## Stability Notes

- Grid construction utilities are public when documented here
- Signatures are stable within major versions
- Additional grid helpers may be added additively

---

## Related Documentation

- **Concepts → Search and Tie-Breaking**
- **Concepts → Tuning**
- **API → Search → Kernels**
- **How-To → Run Sensitivity Sweeps**

---

*Candidate grids make assumptions visible — which is the first step toward governance.*
