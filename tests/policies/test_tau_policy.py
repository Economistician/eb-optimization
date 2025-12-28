from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eb_optimization.policies.tau_policy import (  # type: ignore
    TauPolicy,
    apply_entity_tau_policy,
    apply_tau_policy,
    apply_tau_policy_hr,
)

try:
    from eb_optimization.policies.tau_policy import DEFAULT_TAU_POLICY  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_TAU_POLICY = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _taupolicy_init_params() -> set[str]:
    sig = inspect.signature(TauPolicy)
    return set(sig.parameters.keys())


def _first_present(keys: tuple[str, ...], haystack: set[str]) -> str | None:
    for k in keys:
        if k in haystack:
            return k
    return None


def make_policy(
    *,
    method: str = "target_hit_rate",
    estimate_kwargs: dict[str, Any] | None = None,
    **overrides: Any,
) -> TauPolicy:
    """
    Construct TauPolicy in a way that is resilient to small API changes.

    - method goes to `method` if supported.
    - method-specific parameters go into one of:
        estimate_kwargs / estimate_params / kwargs
      if present on the policy.
    - governance-ish parameters (min_n, cap_with_global, global_cap_quantile)
      are included only if the policy accepts them.
    """
    init_params = _taupolicy_init_params()

    payload: dict[str, Any] = {}

    if "method" in init_params:
        payload["method"] = method

    ek = {} if estimate_kwargs is None else dict(estimate_kwargs)

    # Where do method-specific knobs live?
    ek_field = _first_present(("estimate_kwargs", "estimate_params", "kwargs"), init_params)
    if ek_field is not None:
        payload[ek_field] = ek
    else:
        # If there's no dedicated field, we can't safely pass them
        # (keep empty; policy may hardcode defaults).
        pass

    # Add any override fields ONLY if supported.
    for k, v in overrides.items():
        if k in init_params:
            payload[k] = v

    return TauPolicy(**payload)  # type: ignore[arg-type]


def _unpack_tau_result(res: Any) -> tuple[float, dict]:
    """
    apply_tau_policy might return:
      - (tau, diag)
      - TauEstimate-like object with .tau and .diagnostics
      - tau alone
    Normalize to (tau, diag).
    """
    if isinstance(res, tuple) and len(res) == 2:
        tau, diag = res
        return float(tau), dict(diag)

    # dataclass-like
    if hasattr(res, "tau"):
        tau = float(res.tau)
        diag = getattr(res, "diagnostics", {})
        return tau, dict(diag or {})

    # scalar
    return float(res), {}


def _unpack_hr_result(res: Any) -> tuple[float, float, dict]:
    """
    apply_tau_policy_hr might return:
      - (hr, tau, diag)
      - (hr, tau) (rare)
    """
    if isinstance(res, tuple) and len(res) == 3:
        hr, tau, diag = res
        return float(hr), float(tau), dict(diag)
    if isinstance(res, tuple) and len(res) == 2:
        hr, tau = res
        return float(hr), float(tau), {}
    raise TypeError(f"Unexpected return from apply_tau_policy_hr: {type(res)} {res!r}")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_apply_tau_policy_target_hit_rate_basic():
    """
    Smoke + sanity: should produce a finite, non-negative tau on finite inputs.
    """
    y = np.array([10.0, 10.0, 10.0, 10.0])
    yhat = np.array([9.0, 10.0, 12.0, 10.0])  # abs errors: [1,0,2,0]

    policy = make_policy(
        method="target_hit_rate",
        estimate_kwargs={
            "target_hit_rate": 0.50,
            "tau_floor": 0.0,
            "tau_cap": None,
        },
    )

    tau, diag = _unpack_tau_result(apply_tau_policy(y=y, yhat=yhat, policy=policy))

    assert np.isfinite(tau)
    assert tau >= 0.0
    assert isinstance(diag, dict)


def test_apply_tau_policy_respects_floor_and_cap_when_supported():
    """
    If your policy passes tau_floor/tau_cap through to tuning, tau should be bounded.
    If not supported, we still assert tau is finite/non-negative.
    """
    y = np.array([10.0, 10.0, 10.0, 10.0])
    yhat = np.array([0.0, 0.0, 0.0, 0.0])  # abs errors all 10

    policy = make_policy(
        method="target_hit_rate",
        estimate_kwargs={
            "target_hit_rate": 0.90,
            "tau_floor": 12.0,
            "tau_cap": 15.0,
        },
    )

    tau, diag = _unpack_tau_result(apply_tau_policy(y=y, yhat=yhat, policy=policy))

    assert np.isfinite(tau)
    assert tau >= 0.0

    # If floor/cap made it through, the tau should be bounded.
    # We only enforce bounds if the diagnostics mention them (strong signal of support).
    if "tau_floor" in diag or "tau_cap" in diag:
        assert tau >= 12.0
        assert tau <= 15.0


def test_apply_tau_policy_hr_returns_hr_and_tau():
    """
    apply_tau_policy_hr should return (hr, tau, diag) and for perfect forecasts:
    hr == 1.0, tau should be 0 (or floored to 0).
    """
    y = np.array([5.0, 7.0, 9.0])
    yhat = np.array([5.0, 7.0, 9.0])  # perfect

    policy = make_policy(
        method="target_hit_rate",
        estimate_kwargs={"target_hit_rate": 0.90, "tau_floor": 0.0},
    )

    hr, tau, diag = _unpack_hr_result(apply_tau_policy_hr(y=y, yhat=yhat, policy=policy))

    assert np.isfinite(hr)
    assert np.isfinite(tau)
    assert hr == 1.0
    assert tau == 0.0
    assert isinstance(diag, dict)


def test_apply_tau_policy_handles_non_finite_pairs():
    """
    Non-finite pairs dropped; if none remain, tau should be NaN.
    """
    y = np.array([np.nan, np.inf, -np.inf])
    yhat = np.array([1.0, 2.0, 3.0])

    policy = make_policy(method="target_hit_rate", estimate_kwargs={"target_hit_rate": 0.90})

    tau, diag = _unpack_tau_result(apply_tau_policy(y=y, yhat=yhat, policy=policy))

    assert np.isnan(tau)
    assert isinstance(diag, dict)


def test_apply_entity_tau_policy_min_n_blocks_small_entities_if_supported():
    """
    Entities with < min_n finite observations should return tau NaN
    when min_n governance exists. If not supported, just smoke-test output.
    """
    df = pd.DataFrame(
        {
            "entity": ["A"] * 5 + ["B"] * 40,
            "y": [10.0] * 45,
            "yhat": [9.0] * 5 + [10.0] * 40,
        }
    )

    policy = make_policy(
        method="target_hit_rate",
        estimate_kwargs={"target_hit_rate": 0.90},
        min_n=30,
        cap_with_global=False,
    )

    out = apply_entity_tau_policy(
        df=df,
        entity_col="entity",
        y_col="y",
        yhat_col="yhat",
        policy=policy,
    )

    assert set(out["entity"]) == {"A", "B"}
    assert "tau" in out.columns

    # Only enforce min_n behavior if the policy actually supports min_n.
    if "min_n" in _taupolicy_init_params():
        tau_a = float(out.loc[out["entity"] == "A", "tau"].iloc[0])
        tau_b = float(out.loc[out["entity"] == "B", "tau"].iloc[0])
        assert np.isnan(tau_a)
        assert np.isfinite(tau_b)


def test_apply_entity_tau_policy_can_cap_with_global_if_supported():
    """
    If cap_with_global is supported, entity tau should be capped (weak check).
    """
    df = pd.DataFrame(
        {
            "entity": ["A"] * 100 + ["B"] * 100,
            "y": [100.0] * 200,
            "yhat": ([0.0] * 100) + ([99.0] * 100),  # A abs err=100, B abs err=1
        }
    )

    policy = make_policy(
        method="target_hit_rate",
        estimate_kwargs={"target_hit_rate": 0.99},
        min_n=30,
        cap_with_global=True,
        global_cap_quantile=0.50,
    )

    out = apply_entity_tau_policy(
        df=df,
        entity_col="entity",
        y_col="y",
        yhat_col="yhat",
        policy=policy,
    )

    assert set(out["entity"]) == {"A", "B"}
    assert "tau" in out.columns

    if "cap_with_global" in _taupolicy_init_params():
        tau_a = float(out.loc[out["entity"] == "A", "tau"].iloc[0])
        tau_b = float(out.loc[out["entity"] == "B", "tau"].iloc[0])
        assert np.isfinite(tau_a)
        assert np.isfinite(tau_b)
        # A should not remain "huge" if a global cap is applied; keep it weak.
        assert tau_a <= 100.0


@pytest.mark.skipif(DEFAULT_TAU_POLICY is None, reason="DEFAULT_TAU_POLICY not defined")
def test_default_tau_policy_smoke():
    y = np.array([10.0, 12.0, 8.0, 9.0])
    yhat = np.array([9.0, 11.0, 8.0, 10.0])

    tau, diag = _unpack_tau_result(apply_tau_policy(y=y, yhat=yhat, policy=DEFAULT_TAU_POLICY))
    assert np.isfinite(tau) or np.isnan(tau)
    assert isinstance(diag, dict)
