# Contributing

Thank you for your interest in contributing to **eb-optimization**.
This document outlines contribution expectations, development standards, and governance
principles for the Electric Barometer ecosystem.

Contributions are welcome — but correctness, clarity, and governance always take precedence
over convenience.

---

## Guiding Principles

All contributions to eb-optimization should adhere to the following principles:

- **Explicitness over cleverness**
- **Determinism over performance shortcuts**
- **Governance over convenience**
- **Clarity over abstraction**

This repository is a decision framework first, and a codebase second.

---

## Scope of Contributions

Appropriate contributions include:

- Bug fixes with clear reproduction cases
- New tuning strategies that align with existing philosophy
- Additional policy types with explicit governance rationale
- Improvements to documentation and examples
- Test coverage improvements

Out-of-scope contributions include:

- Forecasting models
- Online or adaptive tuning mechanisms
- Implicit or stochastic decision logic
- Evaluation shortcuts that bypass policies

---

## Development Environment

### Requirements

- Python 3.10+
- pip or equivalent environment manager
- Familiarity with NumPy and pandas
- Comfort with typed Python codebases

---

### Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ElectricBarometer/eb-optimization.git
cd eb-optimization
pip install -e .[dev]
```

---

## Code Standards

### Style and Formatting

- Code formatting is enforced via **ruff**
- Imports must be explicit and ordered
- Docstrings should be clear and complete
- Public APIs must include type hints

Formatting violations should be fixed before submitting a PR.

---

### Determinism Requirements

All code must satisfy the following:

- No stochastic behavior unless explicitly documented
- No reliance on iteration order
- No hidden global state
- Identical inputs must produce identical outputs

If determinism cannot be guaranteed, the contribution will not be accepted.

---

## Tests

### Expectations

All contributions must include:

- Unit tests covering expected behavior
- Tests for edge cases and failure modes
- Tests demonstrating determinism

Existing tests must continue to pass.

---

### Running Tests

```bash
pytest
```

Tests are treated as **contract enforcement**, not optional checks.

---

## Documentation Requirements

Any change affecting:

- Public APIs
- Policy semantics
- Tuning behavior
- Defaults or assumptions

must include corresponding documentation updates.

Documentation is considered part of the API.

---

## Governance Review

Changes affecting decision logic require additional scrutiny:

- Clear explanation of *why* the change is needed
- Description of impact on existing behavior
- Migration or compatibility considerations

Large changes may be rejected if governance implications are unclear.

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests and docs
4. Ensure all checks pass
5. Submit a pull request with a clear description

PRs should focus on a single, coherent change.

---

## Commit Guidelines

- Use clear, descriptive commit messages
- Avoid mixing unrelated changes
- Reference issues or discussions when applicable

Commits should tell a story.

---

## Review Philosophy

Code review prioritizes:

- Correctness
- Determinism
- Conceptual clarity
- Alignment with ecosystem principles

Performance optimizations without justification will be questioned.

---

## When in Doubt

If you are unsure whether a contribution fits:

- Open an issue first
- Describe the problem and proposed solution
- Discuss governance and layering implications

Early discussion prevents wasted effort.

---

## Community Expectations

- Be respectful and constructive
- Favor clarity over jargon
- Assume good intent
- Treat decisions as shared responsibilities

---

## Final Note

eb-optimization is designed to support **high-stakes decision systems**.
Every contribution becomes part of that trust boundary.

---

*Contributions are welcome — but decisions must remain defensible.*
