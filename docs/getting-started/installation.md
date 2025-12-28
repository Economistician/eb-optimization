# Installation

This guide describes how to install **eb-optimization** and its dependencies as part of the
Electric Barometer ecosystem or as a standalone decision-optimization library.

---

## Supported Python Versions

eb-optimization supports the following Python versions:

- Python **3.10**
- Python **3.11**
- Python **3.12**

Using newer Python versions is recommended to ensure compatibility with typing,
numerical libraries, and CI tooling.

---

## Installation from PyPI

The recommended way to install eb-optimization is via **pip**:

```bash
pip install eb-optimization
```

This installs the core library and its runtime dependencies.

---

## Installing as Part of the Electric Barometer Ecosystem

eb-optimization is designed to interoperate with other Electric Barometer packages.

For a typical workflow, install:

```bash
pip install eb-metrics eb-optimization eb-evaluation
```

This ensures all layers of the EB stack are available and version-compatible.

---

## Optional Dependencies

Some workflows may require additional packages for analysis or visualization.

These are intentionally kept out of the core dependency set.

Example:

```bash
pip install eb-optimization[dev]
```

Refer to the project’s `pyproject.toml` for the full list of optional extras.

---

## Installing from Source (Development)

To install eb-optimization from source:

```bash
git clone https://github.com/ElectricBarometer/eb-optimization.git
cd eb-optimization
pip install -e .
```

Editable installs are recommended for contributors and advanced users.

---

## Dependency Management Philosophy

eb-optimization follows these principles:

- Minimal runtime dependencies
- Explicit optional extras
- No hidden transitive requirements across EB layers

This ensures predictable environments and reproducible results.

---

## Verifying the Installation

After installation, verify that the package imports correctly:

```python
import eb_optimization
```

You may also run the test suite to confirm full functionality:

```bash
pytest
```

---

## Common Installation Issues

- **Version conflicts**: ensure compatible versions across EB packages
- **Outdated pip**: upgrade pip if installation fails
- **Editable installs**: avoid mixing editable and non-editable installs unintentionally

---

## Environment Recommendations

For production or research use:

- Use virtual environments (venv, conda, or similar)
- Pin package versions explicitly
- Track policy artifacts alongside environment metadata

---

## Next Steps

- Continue with **Getting Started → Quickstart**
- Review **Concepts → Layering and Scope**
- Explore **How-To guides** for common workflows

---

*eb-optimization is designed to be explicit, composable, and environment-safe.*
