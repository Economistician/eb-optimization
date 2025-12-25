import pytest
import pandas as pd
from eb_optimization.policies.ral_policy import RALPolicy
import pickle


# Test: Initialization with global uplift and segment-level uplifts
def test_initialization_with_segmented_policy():
    # Sample segment data
    uplift_table = pd.DataFrame({
        'segment': ['A', 'B'],
        'uplift': [1.1, 1.2]
    })
    policy = RALPolicy(global_uplift=1.05, segment_cols=['segment'], uplift_table=uplift_table)
    
    # Validate global uplift and segment columns
    assert policy.global_uplift == 1.05, "Global uplift value mismatch"
    assert policy.segment_cols == ['segment'], "Segment columns mismatch"
    pd.testing.assert_frame_equal(policy.uplift_table, uplift_table, check_like=True)  # check_like allows order flexibility


# Test: Initialization with only global uplift (no segmentation)
def test_initialization_with_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)
    
    # Validate global uplift and ensure no segment columns and no uplift table
    assert policy.global_uplift == 1.05, "Global uplift value mismatch"
    assert policy.segment_cols == [], "Segment columns should be empty"
    assert policy.uplift_table is None, "Uplift table should be None"


# Test: is_segmented method returns True when there are segment-level uplifts
def test_is_segmented_with_segmented_policy():
    uplift_table = pd.DataFrame({
        'segment': ['A', 'B'],
        'uplift': [1.1, 1.2]
    })
    policy = RALPolicy(global_uplift=1.05, segment_cols=['segment'], uplift_table=uplift_table)
    
    assert policy.is_segmented() is True, "Policy should be segmented"


# Test: is_segmented method returns False when there are no segment-level uplifts
def test_is_segmented_with_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)
    
    assert policy.is_segmented() is False, "Policy should not be segmented"


# Test: is_segmented method returns False when the uplift_table is empty
def test_is_segmented_with_empty_uplift_table():
    uplift_table = pd.DataFrame(columns=['segment', 'uplift'])
    policy = RALPolicy(global_uplift=1.05, segment_cols=['segment'], uplift_table=uplift_table)
    
    assert policy.is_segmented() is False, "Policy should not be segmented when uplift_table is empty"


# Test: Ensure that RALPolicy is serializable (can be pickled)
def test_serialization_of_ral_policy():
    uplift_table = pd.DataFrame({
        'segment': ['A', 'B'],
        'uplift': [1.1, 1.2]
    })
    policy = RALPolicy(global_uplift=1.05, segment_cols=['segment'], uplift_table=uplift_table)

    # Serialize the policy to a byte stream
    serialized_policy = pickle.dumps(policy)
    
    # Deserialize back to ensure it is intact
    deserialized_policy = pickle.loads(serialized_policy)
    
    assert deserialized_policy.global_uplift == policy.global_uplift, "Global uplift mismatch after deserialization"
    assert deserialized_policy.segment_cols == policy.segment_cols, "Segment columns mismatch after deserialization"
    pd.testing.assert_frame_equal(deserialized_policy.uplift_table, policy.uplift_table, check_like=True)  # check_like allows order flexibility


# Test: Ensure that RALPolicy is serializable (for global-only policy)
def test_serialization_of_global_only_policy():
    policy = RALPolicy(global_uplift=1.05, segment_cols=[], uplift_table=None)

    # Serialize the policy to a byte stream
    serialized_policy = pickle.dumps(policy)
    
    # Deserialize back to ensure it is intact
    deserialized_policy = pickle.loads(serialized_policy)
    
    assert deserialized_policy.global_uplift == policy.global_uplift, "Global uplift mismatch after deserialization"
    assert deserialized_policy.segment_cols == policy.segment_cols, "Segment columns mismatch after deserialization"
    assert deserialized_policy.uplift_table is None, "Uplift table should be None after deserialization"