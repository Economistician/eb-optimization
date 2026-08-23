def test_policies_public_api_imports():
    """
    Public API smoke test for `eb_optimization.policies`.

    This test intentionally imports key policy types, defaults, and helpers to ensure:
    - the public API remains stable across refactors,
    - symbols are exposed from the package as expected, and
    - import-time side effects (e.g., circular imports) are caught early.
    """
    from eb_optimization.policies import (
        DEFAULT_COST_RATIO_POLICY,
        DEFAULT_RAL_POLICY,
        CostRatioPolicy,
        DQCResultSummary,
        RALPolicyArtifact,
        RALTwoBandPolicy,
        TauPolicyArtifact,
        __all__ as policies_all,
        apply_cost_ratio_policy,
        apply_entity_cost_ratio_policy,
        apply_entity_tau_policy,
        apply_ral_policy,
        apply_tau_policy,
        apply_tau_policy_hr,
    )

    # Touch symbols so linters/optimizers can't "optimize away" imports
    assert TauPolicyArtifact and CostRatioPolicy and RALPolicyArtifact
    assert RALTwoBandPolicy is not None
    assert DQCResultSummary is not None
    assert DEFAULT_COST_RATIO_POLICY is not None
    assert DEFAULT_RAL_POLICY is not None
    assert callable(apply_tau_policy)
    assert callable(apply_tau_policy_hr)
    assert callable(apply_entity_tau_policy)
    assert callable(apply_cost_ratio_policy)
    assert callable(apply_entity_cost_ratio_policy)
    assert callable(apply_ral_policy)
    assert "RALPolicy" not in policies_all
    assert "TauPolicy" not in policies_all
    assert "DQCResult" not in policies_all
    assert "RALPolicyArtifact" in policies_all
    assert "TauPolicyArtifact" in policies_all
    assert "DQCResultSummary" in policies_all
    assert "RALTwoBandPolicy" in policies_all


def test_root_exports_disambiguated_policy_artifacts():
    """Root exports use artifact names that do not collide with eb-evaluation."""
    import eb_optimization as m
    from eb_optimization.policies.dqc_policy import DQCResult
    from eb_optimization.policies.ral_policy import RALPolicy
    from eb_optimization.policies.tau_policy import TauPolicy

    assert "RALPolicyArtifact" in m.__all__
    assert "RALTwoBandPolicy" in m.__all__
    assert "RALDeltas" in m.__all__
    assert "RALBands" in m.__all__
    assert "RALBandThresholds" in m.__all__
    assert "RALThresholdTwoBandPolicy" in m.__all__
    assert "TauPolicyArtifact" in m.__all__
    assert "DQCResultSummary" in m.__all__
    assert "enforce_snapping" in m.__all__
    assert "compute_dqc" in m.__all__
    assert callable(m.compute_dqc)
    assert "RALPolicy" not in m.__all__
    assert "TauPolicy" not in m.__all__
    assert "DQCResult" not in m.__all__
    assert m.RALPolicyArtifact is RALPolicy
    assert m.TauPolicyArtifact is TauPolicy
    assert m.RALPolicyArtifact is not None
    assert m.RALTwoBandPolicy is not None
    assert m.DQCResultSummary is DQCResult
    assert callable(m.enforce_snapping)
    from eb_optimization.policies import DQCResultSummary

    assert DQCResultSummary is DQCResult
    assert m.DQCResultSummary is DQCResultSummary


def test_search_and_tuning_module_exports():
    """
    Public API smoke test for top-level module exports.

    We export modules (not functions) at the package level to keep imports stable and
    to prevent import-time regressions from refactors (e.g., circular import paths).
    """
    from eb_optimization import search, tuning

    assert hasattr(search, "grid")
    assert hasattr(search, "kernels")

    assert hasattr(tuning, "cost_ratio")
    assert hasattr(tuning, "sensitivity")
    assert hasattr(tuning, "tau")
    assert hasattr(tuning, "ral")
