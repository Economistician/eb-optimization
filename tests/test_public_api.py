def test_policies_public_api_imports():
    from eb_optimization.policies import (
        TauPolicy,
        CostRatioPolicy,
        RALPolicy,
        DEFAULT_COST_RATIO_POLICY,
        DEFAULT_RAL_POLICY,
        apply_tau_policy,
        apply_tau_policy_hr,
        apply_entity_tau_policy,
        apply_cost_ratio_policy,
        apply_entity_cost_ratio_policy,
        apply_ral_policy,
    )

    # Touch symbols so linters/optimizers can't "optimize away" imports
    assert TauPolicy and CostRatioPolicy and RALPolicy
    assert DEFAULT_COST_RATIO_POLICY is not None
    assert DEFAULT_RAL_POLICY is not None
    assert callable(apply_tau_policy)
    assert callable(apply_tau_policy_hr)
    assert callable(apply_entity_tau_policy)
    assert callable(apply_cost_ratio_policy)
    assert callable(apply_entity_cost_ratio_policy)
    assert callable(apply_ral_policy)


def test_search_and_tuning_module_exports():
    # module-level stability (export modules, not functions)
    from eb_optimization import search, tuning

    assert hasattr(search, "grid")
    assert hasattr(search, "kernels")
    assert hasattr(search, "results")

    assert hasattr(tuning, "cost_ratio")
    assert hasattr(tuning, "sensitivity")
    assert hasattr(tuning, "tau")
    assert hasattr(tuning, "ral")