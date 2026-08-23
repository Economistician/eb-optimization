# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `apply_ral_policy` is hard-deprecated and always raises `ValueError`, directing callers to `electric_barometer.apply_ral` (or `eb_evaluation.apply_ral`) with a governance decisions table.

## [0.2.6] - 2026-08-23

### Added

- Root `__all__` exports `RALPolicyArtifact` (multiplicative RAL) alongside `RALTwoBandPolicy` and related RAL artifacts.
- Root `__all__` exports `enforce_snapping`, `DQCResultSummary`, and `compute_dqc`.

### Changed

- Polished policy module docstrings to concise technical overviews.
- Tightened README Overview; removed cloned Role section.
- Changelog version header now matches `pyproject.toml` (`0.2.6`).
- Public surfaces export `RALPolicyArtifact`, `TauPolicyArtifact`, and `DQCResultSummary` so they do not collide with `eb-evaluation` governance types.
- `compute_dqc` delegates the full demand series to `eb_evaluation.classify_dqc` so scoring and governance share one DQC class.
- Pinned sibling Electric Barometer packages to exact System Release 0.2.9 versions.

### Fixed

- Resolved Pyright argument-type errors in cost-ratio tuning tests.
- DQC enforcement fails closed on `UNKNOWN` or insufficient positive observations instead of mapping to `CONTINUOUS`.
- Segmented `RALPolicy.adjust_forecast` uses the same replace-or-global-fallback rule as `eb-evaluation` `ReadinessAdjustmentLayer.transform`.
- `enforce_snapping` raises when demand is `QUANTIZED`/`PACKED` but Δ* is missing or invalid, instead of returning unsnapped forecasts.
- Evaluation helpers keep `enforce="snap"` as the only implicit default; `ignore` is explicit opt-in.
- `snap_to_grid` uses the evaluation engine (`math.ceil` / `math.floor` / half-away-from-zero) instead of NumPy banker's rounding.
- `hr_at_tau_grid_units` interprets τ in raw units only for `CONTINUOUS` demand; `UNKNOWN` remains a raised fail-closed error.

## [0.2.5] - 2026-08-22

### Added

- Exposed `CostRatioPolicy`, `TauPolicy`, `DQCPolicy`, and key helpers on root `__all__`.
- Added `py.typed` marker and updated Python floor to `>=3.11`.
