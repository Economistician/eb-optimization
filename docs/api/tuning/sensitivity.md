# Sensitivity Analysis

This document provides the **API reference** for sensitivity analysis utilities in
`eb_optimization.tuning.sensitivity`.

Sensitivity analysis evaluates how metrics respond to changes in decision parameters.
It is a **diagnostic tool**, not a tuning or optimization mechanism.

---

## Conceptual Role

Sensitivity analysis exists to answer the question:

> *“If we had chosen a different decision parameter, how would outcomes change?”*

Within the Electric Barometer ecosystem, sensitivity analysis:

- Explores parameter-response relationships
- Supports judgment and review
- Validates stability of tuned parameters
- Never freezes decisions

Sensitivity analysis does not choose parameters — people do.

---

## Public Functions

### `cwsl_sensitivity`

Evaluate Cost-Weighted Service Loss (CWSL) across a grid of cost ratio values.

**Signature**

```python
cwsl_sensitivity(
    y_true,
    y_pred,
    *,
    R_grid,
    **kwargs,
)
```

**Description**

Computes CWSL for each candidate R value in `R_grid`, holding all other inputs fixed.

**Parameters**

- `y_true`  
  Actual observed values.

- `y_pred`  
  Forecast values corresponding to `y_true`.

- `R_grid`  
  Iterable of candidate cost ratio values to evaluate.

**Returns**

- Structured result containing:
  - Candidate R values
  - Corresponding CWSL values
  - Metadata suitable for plotting or inspection

---

### `compute_cwsl_sensitivity_df`

Run CWSL sensitivity analysis on panel data.

**Signature**

```python
compute_cwsl_sensitivity_df(
    df,
    *,
    entity_col,
    actual_col,
    forecast_col,
    R_grid,
    **kwargs,
)
```

**Description**

Computes CWSL sensitivity for each entity and candidate R value in a tidy DataFrame
format suitable for aggregation and visualization.

**Parameters**

- `df`  
  Panel DataFrame containing actuals and forecasts.

- `entity_col`  
  Column identifying entities.

- `actual_col`  
  Column containing actual values.

- `forecast_col`  
  Column containing forecast values.

- `R_grid`  
  Iterable of candidate cost ratio values.

**Returns**

- pandas DataFrame with:
  - One row per entity per candidate R
  - Metric values and identifiers

---

## Determinism Guarantees

Sensitivity analysis guarantees:

- Deterministic outputs
- No stochastic behavior
- No modification of inputs or policies

Identical inputs always produce identical sensitivity curves.

---

## Relationship to Tuning

It is critical to distinguish:

- **Tuning**: selects parameters to freeze into policies
- **Sensitivity analysis**: evaluates implications of alternatives

Sensitivity analysis informs tuning but must never replace it.

---

## Interpretation Guidance

When reviewing sensitivity results:

- Look for flat regions indicating robustness
- Identify sharp inflection points
- Beware of noise in small samples
- Prefer stable regimes over marginal optima

Sensitivity curves support judgment, not automation.

---

## Common Usage Patterns

- Pre-freeze validation of tuned parameters
- Stakeholder review and communication
- Comparative analysis across entities
- Diagnostic exploration during development

---

## Anti-Patterns

The following patterns are discouraged:

- Selecting parameters directly from sensitivity minima
- Treating sensitivity sweeps as optimization routines
- Running sweeps on policy-adjusted forecasts
- Over-interpreting minor metric differences

---

## Stability Notes

- Sensitivity APIs are public and stable
- Output structure is stable within major versions
- Additional sensitivity utilities may be added additively

---

## Related Documentation

- **How-To → Run Sensitivity Sweeps**
- **Concepts → Tuning**
- **Concepts → Search and Tie-Breaking**
- **API → Tuning → Cost Ratio**

---

*Sensitivity analysis reveals consequences — it does not decide outcomes.*
