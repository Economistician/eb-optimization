import pickle

import numpy as np
import pandas as pd
import pytest

from eb_optimization.policies.ral_policy import (
    RALBandThresholds,
    RALDeltas,
    RALPolicy,
    RALThresholdTwoBandPolicy,
    RALTwoBandPolicy,
)


# Test: Initialization with global uplift and segment-level uplifts
def test_initialization_with_segmented_policy():
    # Sample segment data
    uplift_table = pd.DataFrame({"segment": ["A", "B"], "uplift": [1.1, 1.2]})
    policy = RALPolicy(global_uplift=1.05, segment_cols=["segment"], uplift_table=uplift_table)

    # Validate global uplift and segment columns
    assert policy.global_uplift == 1.05, "Global uplift value mismatch"
    assert policy.segment_cols == ["segment"], "Segment columns mismatch"
    pd.testing.assert_frame_equal(
        policy.uplift_table, uplift_table, check_like=True
    )  # check_like allows order flexibility


# Test: Initialization with only global uplift (no segmentation)
def test_initialization_with_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)

    # Validate global uplift and ensure no segment columns and no uplift table
    assert policy.global_uplift == 1.05, "Global uplift value mismatch"
    assert policy.segment_cols == [], "Segment columns should be empty"
    assert policy.uplift_table is None, "Uplift table should be None"


# Test: is_segmented method returns True when there are segment-level uplifts
def test_is_segmented_with_segmented_policy():
    uplift_table = pd.DataFrame({"segment": ["A", "B"], "uplift": [1.1, 1.2]})
    policy = RALPolicy(global_uplift=1.05, segment_cols=["segment"], uplift_table=uplift_table)

    assert policy.is_segmented() is True, "Policy should be segmented"


# Test: is_segmented method returns False when there are no segment-level uplifts
def test_is_segmented_with_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)

    assert policy.is_segmented() is False, "Policy should not be segmented"


# Test: is_segmented method returns False when the uplift_table is empty
def test_is_segmented_with_empty_uplift_table():
    uplift_table = pd.DataFrame(columns=pd.Index(["segment", "uplift"]))
    policy = RALPolicy(global_uplift=1.05, segment_cols=["segment"], uplift_table=uplift_table)

    assert policy.is_segmented() is False, (
        "Policy should not be segmented when uplift_table is empty"
    )


# Test: Ensure that RALPolicy is serializable (can be pickled)
def test_serialization_of_ral_policy():
    uplift_table = pd.DataFrame({"segment": ["A", "B"], "uplift": [1.1, 1.2]})
    policy = RALPolicy(global_uplift=1.05, segment_cols=["segment"], uplift_table=uplift_table)

    # Serialize the policy to a byte stream
    serialized_policy = pickle.dumps(policy)

    # Deserialize back to ensure it is intact
    deserialized_policy = pickle.loads(serialized_policy)  # nosec

    assert deserialized_policy.global_uplift == policy.global_uplift, (
        "Global uplift mismatch after deserialization"
    )
    assert deserialized_policy.segment_cols == policy.segment_cols, (
        "Segment columns mismatch after deserialization"
    )
    pd.testing.assert_frame_equal(
        deserialized_policy.uplift_table, policy.uplift_table, check_like=True
    )  # check_like allows order flexibility


# Test: Ensure that RALPolicy is serializable (for global-only policy)
def test_serialization_of_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)

    # Serialize the policy to a byte stream
    serialized_policy = pickle.dumps(policy)

    # Deserialize back to ensure it is intact
    deserialized_policy = pickle.loads(serialized_policy)  # nosec

    assert deserialized_policy.global_uplift == policy.global_uplift, (
        "Global uplift mismatch after deserialization"
    )
    assert deserialized_policy.segment_cols == policy.segment_cols, (
        "Segment columns mismatch after deserialization"
    )
    assert deserialized_policy.uplift_table is None, (
        "Uplift table should be None after deserialization"
    )


def test_two_band_policy_global_adjustment():
    df = pd.DataFrame(
        {
            "yhat": [0.70, 0.76, 0.84, 0.85, 0.90],
        }
    )
    policy = RALTwoBandPolicy(global_deltas=RALDeltas(d_mid=0.02, d_high=0.01))

    out = policy.adjust_forecast(df, "yhat")

    expected = np.array([0.70, 0.78, 0.86, 0.86, 0.91], dtype=float)
    assert out.name == "readiness_forecast"
    np.testing.assert_allclose(out.to_numpy(dtype=float), expected, rtol=0.0, atol=1e-12)


def test_two_band_policy_per_key_override_and_fallback():
    df = pd.DataFrame(
        {
            "interface": ["A", "A", "B", "B", "C"],
            "yhat": [0.76, 0.86, 0.76, 0.86, 0.86],
        }
    )
    policy = RALTwoBandPolicy(
        global_deltas=RALDeltas(d_mid=0.01, d_high=0.01),
        per_key_deltas={
            "A": RALDeltas(d_mid=0.02, d_high=0.00),  # A: stronger mid, no high
            "B": RALDeltas(d_mid=0.00, d_high=0.02),  # B: no mid, stronger high
        },
    )

    out = policy.adjust_forecast(df, "yhat", key_col="interface").to_numpy(dtype=float)

    # A: 0.76 -> +0.02 = 0.78 (mid band), 0.86 -> +0.00 = 0.86 (high band but d_high=0.00)
    # B: 0.76 -> +0.00 = 0.76 (mid band but d_mid=0.00), 0.86 -> +0.02 = 0.88 (high band)
    # C: fallback to global: 0.86 -> +0.01 = 0.87
    expected = np.array([0.78, 0.86, 0.76, 0.88, 0.87], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_two_band_policy_missing_key_col_raises():
    df = pd.DataFrame({"yhat": [0.8]})
    policy = RALTwoBandPolicy(global_deltas=RALDeltas(d_mid=0.01, d_high=0.01))

    with pytest.raises(ValueError):
        policy.adjust_forecast(df, "yhat", key_col="interface")


def test_threshold_two_band_policy_global_adjustment():
    df = pd.DataFrame({"yhat": [0.69, 0.70, 0.83, 0.84, 0.88]})
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.70, high=0.84),
        global_deltas=RALDeltas(d_mid=0.02, d_high=0.01),
    )

    out = policy.adjust_forecast(df, "yhat")

    # mid: >=0.70 and <0.84 gets +0.02; high: >=0.84 gets +0.01
    expected = np.array([0.69, 0.72, 0.85, 0.85, 0.89], dtype=float)
    assert out.name == "readiness_forecast"
    np.testing.assert_allclose(out.to_numpy(dtype=float), expected, rtol=0.0, atol=1e-12)


def test_threshold_two_band_policy_per_key_thresholds_and_deltas_with_fallback():
    df = pd.DataFrame(
        {
            "interface": ["A", "A", "B", "B", "C", "C"],
            "yhat": [0.74, 0.86, 0.76, 0.86, 0.76, 0.86],
        }
    )

    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.75, high=0.85),
        global_deltas=RALDeltas(d_mid=0.01, d_high=0.01),
        per_key_thresholds={
            # A: make high band start earlier
            "A": RALBandThresholds(mid=0.75, high=0.80),
            # B: make mid band start earlier
            "B": RALBandThresholds(mid=0.70, high=0.85),
        },
        per_key_deltas={
            "A": RALDeltas(d_mid=0.02, d_high=0.00),  # A: stronger mid, no high
            "B": RALDeltas(d_mid=0.00, d_high=0.02),  # B: no mid, stronger high
        },
    )

    out = policy.adjust_forecast(df, "yhat", key_col="interface").to_numpy(dtype=float)

    # A thresholds: mid=0.75, high=0.80; deltas: d_mid=0.02, d_high=0.00
    #   0.74 -> low => 0.74
    #   0.86 -> high => +0.00 => 0.86
    #
    # B thresholds: mid=0.70, high=0.85; deltas: d_mid=0.00, d_high=0.02
    #   0.76 -> mid => +0.00 => 0.76
    #   0.86 -> high => +0.02 => 0.88
    #
    # C fallback: global thresholds mid=0.75 high=0.85, global deltas +0.01/+0.01
    #   0.76 -> mid => +0.01 => 0.77
    #   0.86 -> high => +0.01 => 0.87
    expected = np.array([0.74, 0.86, 0.76, 0.88, 0.77, 0.87], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_threshold_two_band_policy_missing_key_col_raises():
    df = pd.DataFrame({"yhat": [0.8]})
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.75, high=0.85),
        global_deltas=RALDeltas(d_mid=0.01, d_high=0.01),
    )

    with pytest.raises(ValueError):
        policy.adjust_forecast(df, "yhat", key_col="interface")


def test_threshold_two_band_policy_serialization_roundtrip():
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.70, high=0.84),
        global_deltas=RALDeltas(d_mid=0.02, d_high=0.01),
        per_key_thresholds={"A": RALBandThresholds(mid=0.72, high=0.82)},
        per_key_deltas={"A": RALDeltas(d_mid=0.03, d_high=0.00)},
    )

    d = policy.to_dict()
    restored = RALThresholdTwoBandPolicy.from_dict(d)

    assert restored.global_thresholds.mid == policy.global_thresholds.mid
    assert restored.global_thresholds.high == policy.global_thresholds.high
    assert restored.global_deltas.d_mid == policy.global_deltas.d_mid
    assert restored.global_deltas.d_high == policy.global_deltas.d_high

    # Narrow optionals for pyright before subscripting.
    restored_thresholds = restored.per_key_thresholds
    restored_deltas = restored.per_key_deltas
    policy_thresholds = policy.per_key_thresholds
    policy_deltas = policy.per_key_deltas

    assert restored_thresholds is not None
    assert restored_deltas is not None
    assert policy_thresholds is not None
    assert policy_deltas is not None

    assert restored_thresholds["A"].mid == policy_thresholds["A"].mid
    assert restored_thresholds["A"].high == policy_thresholds["A"].high
    assert restored_deltas["A"].d_mid == policy_deltas["A"].d_mid
    assert restored_deltas["A"].d_high == policy_deltas["A"].d_high


def test_threshold_two_band_policy_picklable():
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.70, high=0.84),
        global_deltas=RALDeltas(d_mid=0.02, d_high=0.01),
        per_key_thresholds={"A": RALBandThresholds(mid=0.72, high=0.82)},
        per_key_deltas={"A": RALDeltas(d_mid=0.03, d_high=0.00)},
    )

    serialized = pickle.dumps(policy)
    restored = pickle.loads(serialized)  # nosec

    assert restored.global_thresholds.mid == policy.global_thresholds.mid
    assert restored.global_thresholds.high == policy.global_thresholds.high
    assert restored.global_deltas.d_mid == policy.global_deltas.d_mid
    assert restored.global_deltas.d_high == policy.global_deltas.d_high

    # Narrow optionals for pyright before subscripting.
    restored_thresholds = restored.per_key_thresholds
    restored_deltas = restored.per_key_deltas
    policy_thresholds = policy.per_key_thresholds
    policy_deltas = policy.per_key_deltas

    assert restored_thresholds is not None
    assert restored_deltas is not None
    assert policy_thresholds is not None
    assert policy_deltas is not None

    assert restored_thresholds["A"].mid == policy_thresholds["A"].mid
    assert restored_thresholds["A"].high == policy_thresholds["A"].high
    assert restored_deltas["A"].d_mid == policy_deltas["A"].d_mid
    assert restored_deltas["A"].d_high == policy_deltas["A"].d_high


def test_threshold_two_band_policy_adjust_forecast_capped_caps_upper_and_lower():
    df = pd.DataFrame(
        {
            "interface": ["A", "A", "B", "B"],
            "yhat": [0.86, 0.90, -0.10, 0.50],
        }
    )

    # A gets a +0.30 high uplift; B uses global +0.10 high uplift.
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.75, high=0.85),
        global_deltas=RALDeltas(d_mid=0.0, d_high=0.10),
        per_key_deltas={"A": RALDeltas(d_mid=0.0, d_high=0.30)},
    )

    uncapped = policy.adjust_forecast(df, "yhat", key_col="interface").to_numpy(dtype=float)
    capped = policy.adjust_forecast_capped(
        df,
        "yhat",
        key_col="interface",
        lower=0.0,
        upper=1.0,
    ).to_numpy(dtype=float)

    # Uncapped:
    #  A: 0.86 -> 1.16, 0.90 -> 1.20
    #  B: -0.10 -> -0.10, 0.50 -> 0.50
    expected_uncapped = np.array([1.16, 1.20, -0.10, 0.50], dtype=float)
    np.testing.assert_allclose(uncapped, expected_uncapped, rtol=0.0, atol=1e-12)

    # Capped to [0, 1]:
    expected_capped = np.array([1.00, 1.00, 0.00, 0.50], dtype=float)
    np.testing.assert_allclose(capped, expected_capped, rtol=0.0, atol=1e-12)


def test_threshold_two_band_policy_adjust_forecast_capped_upper_none_disables_upper_cap():
    df = pd.DataFrame({"yhat": [0.90]})
    policy = RALThresholdTwoBandPolicy(
        global_thresholds=RALBandThresholds(mid=0.75, high=0.85),
        global_deltas=RALDeltas(d_mid=0.0, d_high=0.30),
    )

    uncapped = policy.adjust_forecast(df, "yhat").to_numpy(dtype=float)
    capped_no_upper = policy.adjust_forecast_capped(df, "yhat", upper=None).to_numpy(dtype=float)

    np.testing.assert_allclose(uncapped, np.array([1.20], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(capped_no_upper, uncapped, rtol=0.0, atol=1e-12)
