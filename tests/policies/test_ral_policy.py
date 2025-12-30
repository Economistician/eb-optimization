import pickle

import numpy as np
import pandas as pd
import pytest

from eb_optimization.policies.ral_policy import (
    RALDeltas,
    RALPolicy,
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
