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
        RALPolicy,
        TauPolicy,
        apply_cost_ratio_policy,
        apply_entity_cost_ratio_policy,
        apply_entity_tau_policy,
        apply_ral_policy,
        apply_tau_policy,
        apply_tau_policy_hr,
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
