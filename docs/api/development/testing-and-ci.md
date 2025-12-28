# Testing and Continuous Integration (CI)

This document describes **testing expectations**, **local developer workflows**, and the
**continuous integration (CI) system** used by eb-optimization within the Electric Barometer ecosystem.

Testing and CI are treated as **contract enforcement mechanisms**, not optional quality checks.

---

## Philosophy

The testing and CI strategy for eb-optimization prioritizes:

- **Correctness over speed**
- **Determinism over convenience**
- **Contract enforcement over coverage metrics**
- **Centralization over per-repo divergence**

Every merge and release must preserve behavioral guarantees.

---

## What Is Being Protected

The test suite and CI gates protect:

- Public API contracts
- Deterministic behavior guarantees
- Policy immutability and application semantics
- Layering boundaries (metrics vs tuning vs evaluation)
- Backward compatibility across versions

Failures are treated as signal, not noise.

---

## Local Development Workflow

### Recommended Environment

- Python 3.10+
- Virtual environment (venv, conda, or equivalent)
- Editable install with development extras

```bash
pip install -e .[dev]
```

---

### Running Tests Locally

Run the full test suite:

```bash
pytest
```

Tests should pass locally before opening a pull request.

---

### Formatting and Linting

Code style and static checks are enforced consistently across the ecosystem.

Typical local commands:

```bash
ruff format
ruff check
```

Formatting and lint failures will block CI.

---

### Type Checking

Type hints are part of the public contract.

```bash
python -m pyright
```

Type errors must be resolved prior to submission.

---

## Test Design Expectations

Tests should:

- Be deterministic
- Avoid reliance on global state
- Cover edge cases and failure modes
- Reflect documented behavior

Tests are expected to encode *intent*, not implementation details.

---

## Continuous Integration Overview

eb-optimization uses **centralized CI workflows** shared across the Electric Barometer ecosystem.

Key characteristics:

- CI logic lives in a central integration repository
- Individual repos pass configuration parameters only
- Behavior is uniform across all EB packages

This avoids drift and inconsistency.

---

## CI Gates

Typical CI gates include:

- Linting and formatting checks
- Type checking
- Unit and integration tests
- Public API import validation

All gates must pass for a merge to proceed.

---

## Pull Request Validation

For pull requests:

- CI runs automatically on every update
- Failures must be resolved before review
- No bypassing of required checks

PRs that weaken guarantees will not be merged.

---

## Release-Time Validation

At release time, CI additionally enforces:

- Clean build of distribution artifacts
- Installation from built artifacts
- Smoke tests validating basic functionality

This protects downstream consumers.

---

## Determinism Checks

Special attention is paid to determinism:

- Repeated runs must produce identical outputs
- Candidate selection must not depend on ordering
- No randomness is allowed without explicit design

Violations are treated as critical issues.

---

## Handling Failures

When CI fails:

1. Identify whether the failure indicates a contract violation
2. Fix the root cause, not the symptom
3. Update tests or documentation if behavior changed intentionally

Ignoring failures is not acceptable.

---

## Relationship to Other Docs

- See **Development → Contributing** for contribution standards
- See **Development → Release Process** for publishing workflow
- See **Concepts → Tuning** for determinism philosophy

---

## Final Note

Testing and CI are how the Electric Barometer ecosystem enforces trust.

If a behavior is not tested or gated, it is not guaranteed.

---

*In Electric Barometer, CI is not bureaucracy — it is how decisions stay correct.*
