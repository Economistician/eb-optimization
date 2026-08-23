# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Polished policy module docstrings to concise technical overviews.
- Tightened README Overview; removed cloned Role section.
- Changelog version header now matches `pyproject.toml` (`0.2.5`).

### Fixed

- Resolved Pyright argument-type errors in cost-ratio tuning tests.
- DQC enforcement fails closed on `UNKNOWN` or insufficient positive observations instead of mapping to `CONTINUOUS`.
- Segmented `RALPolicy.adjust_forecast` uses the same replace-or-global-fallback rule as `eb-evaluation` `ReadinessAdjustmentLayer.transform`.
- `enforce_snapping` raises when demand is `QUANTIZED`/`PACKED` but Δ* is missing or invalid, instead of returning unsnapped forecasts.
- Evaluation helpers keep `enforce="snap"` as the only implicit default; `ignore` is explicit opt-in.

## [0.2.5] - 2026-08-22

### Added

- Exposed `CostRatioPolicy`, `TauPolicy`, `DQCPolicy`, and key helpers on root `__all__`.
- Added `py.typed` marker and updated Python floor to `>=3.11`.
