# Global vs Entity-Level Calibration

## Motivation

Calibration scope is a first-class modeling and governance decision.
In many real-world forecasting systems, data are panelized: multiple entities
share a common timeline but differ materially in scale, sparsity, behavior,
and operational consequence.

Electric Barometer (EB) supports both global and entity-level calibration
of decision parameters such as the cost ratio R = c_u / c_o.
This document clarifies when each scope is appropriate, why global calibration
can fail in heterogeneous settings, and how entity-level artifacts improve
stability, interpretability, and auditability.

## Global Calibration

### Definition

Global calibration estimates a single parameter value using all observations
pooled across entities and time. For cost-ratio tuning, this means selecting
one R* such that the aggregate underbuild and overbuild costs are balanced
across the full dataset.

### When Global Calibration Works

Global calibration can be appropriate when all of the following hold:

- Entities are homogeneous in scale and behavior
- Admissibility rules are stable and consistent
- The forecast baseline is fixed and comparable across entities
- No single entity dominates aggregate cost
- The modeling question is explicitly global

### Failure Modes of Global Calibration

In heterogeneous or sparse panels, global calibration can become fragile.

Common failure modes include:

- Aggregation dominance by high-volume entities
- Sensitivity to admissibility or filtering rules
- Dependence on baseline construction choices

These effects are consequences of aggregation, not implementation defects.

## Entity-Level Calibration

### Definition

Entity-level calibration estimates parameters independently per entity.
Each entity receives its own calibration artifact derived solely from its
observations and candidate grid.

### Advantages

Entity-level calibration provides:

- Stability: results are insensitive to other entities
- Interpretability: each parameter reflects entity-specific behavior
- Governance clarity: scope and assumptions are explicit and auditable

## Identifiability and Flat Curves

Some regimes provide weak information for calibration:

- Perfect or near-perfect forecasts
- Symmetric error distributions
- Extremely sparse demand

In these cases, the sensitivity curve is effectively flat.
Electric Barometer treats this as a valid outcome, defaulting to neutral
parameters and recording diagnostics rather than implying false precision.

## Recommendations

- Default to entity-level calibration when entities differ materially
- Use global calibration only under homogeneous, stable regimes
- Treat grid resolution as a modeling choice, not a numerical instability
- Interpret flat curves as lack of information, not failure

## Relationship to Calibration Artifacts

These principles map directly to EB artifacts:

- CostRatioEstimate represents a single global decision
- EntityCostRatioEstimate represents structured per-entity calibration
  including scalar results, curves, and governance metadata

## Summary

Calibration scope is a modeling and governance decision, not an implementation
detail. Electric Barometer supports both global and entity-level calibration,
with entity-level calibration recommended for heterogeneous or panelized data.
