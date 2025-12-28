# Release Process

This document describes the **release and versioning process** for eb-optimization within
the Electric Barometer ecosystem.

Releases are treated as **governed events** that freeze decision logic, API contracts,
and behavioral guarantees for downstream consumers.

---

## Release Philosophy

Releases in eb-optimization prioritize:

- **Stability over velocity**
- **Explicit change over silent behavior drift**
- **Governance over convenience**
- **Reproducibility over novelty**

Every release is a commitment to downstream users and systems.

---

## Semantic Versioning

eb-optimization follows **semantic versioning (SemVer)**:

```
MAJOR.MINOR.PATCH
```

### MAJOR
- Breaking API changes
- Changes to policy semantics
- Behavioral changes that affect evaluation outcomes

### MINOR
- Additive APIs
- New tuning strategies
- New policy types
- Documentation additions

### PATCH
- Bug fixes
- Internal refactors with no behavioral change
- Documentation corrections

---

## What Triggers a Release

A release may be triggered by:

- New public APIs
- Changes to tuning logic
- Updates to policy defaults
- Bug fixes affecting correctness
- Documentation changes that clarify behavior

Purely internal changes do not require a release unless behavior is affected.

---

## Pre-Release Checklist

Before cutting a release, ensure:

- All tests pass locally and in CI
- Documentation is up to date
- API changes are documented
- Defaults and assumptions are reviewed
- Version number is updated appropriately

Releases should never be rushed.

---

## CI and Quality Gates

Releases are protected by centralized CI workflows that enforce:

- Code formatting and linting
- Type checking
- Unit and integration tests
- Public API import checks

A release cannot proceed unless all gates pass.

---

## Building Release Artifacts

Release artifacts are built using standard Python tooling:

```bash
python -m build --sdist --wheel
```

Artifacts should be inspected before publication.

---

## Publishing to PyPI

Publishing is performed via a **controlled PyPI release workflow**:

- Uses scoped API tokens
- Requires explicit approval
- Publishes both source and wheel distributions

Manual publishing from local machines is discouraged.

---

## Post-Release Verification

After publication:

- Smoke tests validate installation from PyPI
- Import paths and basic workflows are verified
- Any issues trigger immediate investigation

Post-release checks protect downstream users.

---

## Backward Compatibility

When possible:

- Old APIs are deprecated before removal
- Deprecations are documented clearly
- Migration guidance is provided

Breaking changes require a major version bump.

---

## Policy and Artifact Compatibility

Special care is taken when releases affect:

- Policy constructors
- Policy defaults
- Artifact semantics

Changes that alter interpretation of historical results are treated as breaking.

---

## Coordinated Ecosystem Releases

eb-optimization is part of a larger ecosystem.

When necessary:

- Releases are coordinated with eb-metrics and eb-evaluation
- Compatibility ranges are documented
- Breaking changes are synchronized deliberately

---

## Rollbacks and Hotfixes

If a release introduces a critical issue:

- A patch release is preferred when possible
- Rollbacks are documented explicitly
- Downstream consumers are notified

Hotfixes must preserve determinism guarantees.

---

## Release Documentation

Each release should include:

- A changelog summary
- Description of behavioral changes
- Notes on compatibility and migration

Release notes are part of governance.

---

## Long-Term Support Considerations

For high-stakes environments:

- Older major versions may be supported longer
- Bug fixes may be backported selectively
- Stability is prioritized over new features

---

## Final Note

Releases define trust boundaries.

Once published, a version becomes part of the historical record and must
remain interpretable indefinitely.

---

*In Electric Barometer, releasing code is releasing decisions.*
