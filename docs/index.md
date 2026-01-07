# eb-optimization

`eb-optimization` provides optimization, tuning, and search utilities used to transform forecast outputs into actionable decisions within the Electric Barometer ecosystem.

This package focuses on **policy application**, **parameter tuning**, and **search strategies**, rather than model training or metric definition.

## Scope

This package is responsible for:

- Applying readiness and cost-ratio policies to forecast outputs
- Estimating and tuning cost-ratio and readiness parameters
- Supporting grid-based and kernel-based search strategies
- Providing reusable optimization primitives for downstream systems

It intentionally avoids defining metrics, data contracts, or evaluation workflows.

## Contents

- **Policies**  
  Readiness and cost-ratio policies that govern decision adjustments

- **Search utilities**  
  Tools for exploring parameter spaces and optimization surfaces

- **Tuning helpers**  
  Utilities for estimating and calibrating policy parameters

## API reference

- [Core module](api/eb_optimization.md)
- [Policies](api/policies/)
- [Search](api/search/)
- [Tuning](api/tuning/)
