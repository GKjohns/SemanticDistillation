import copy

from iterative_improvement import (
    compute_dimensionality_budget,
    estimate_config_effective_dim,
    estimate_feature_effective_dim,
    enforce_dimensionality_budget_on_config,
)


def test_estimate_feature_effective_dim_bool_and_scale_are_one():
    assert estimate_feature_effective_dim({"name": "a", "type": "bool"}) == 1
    assert estimate_feature_effective_dim({"name": "b", "type": "scale", "min": 1, "max": 5}) == 1


def test_estimate_feature_effective_dim_categorical_is_k_minus_one():
    assert (
        estimate_feature_effective_dim(
            {"name": "rel_type", "type": "categorical", "values": ["a", "b", "c", "d"]}
        )
        == 3
    )
    # defensive: empty list produces 0
    assert estimate_feature_effective_dim({"name": "x", "type": "categorical", "values": []}) == 0


def test_estimate_config_effective_dim_sums_feature_costs():
    cfg = {
        "name": "X",
        "features": [
            {"name": "a", "type": "bool"},
            {"name": "b", "type": "scale", "min": 1, "max": 3},
            {"name": "c", "type": "categorical", "values": ["u", "v", "w"]},  # 2
        ],
    }
    assert estimate_config_effective_dim(cfg) == 1 + 1 + 2


def test_compute_dimensionality_budget_reasonable_defaults():
    # documented example: n_train=74 -> 18
    assert compute_dimensionality_budget(74) == 18
    # clamp lower bound
    assert compute_dimensionality_budget(20) >= 12


def test_enforce_dimensionality_budget_prunes_high_cost_features_first():
    cfg = {
        "name": "X",
        "features": [
            {"name": "cheap_bool", "type": "bool"},
            {"name": "cheap_scale", "type": "scale", "min": 1, "max": 3},
            {"name": "big_cat", "type": "categorical", "values": ["a", "b", "c", "d", "e", "f"]},  # 5
            {"name": "med_cat", "type": "categorical", "values": ["x", "y", "z"]},  # 2
        ],
    }
    cfg2, info = enforce_dimensionality_budget_on_config(copy.deepcopy(cfg), budget=3)
    # budget=3 can fit cheap_bool(1)+cheap_scale(1)+med_cat(2)=4 (too big)
    # so at least one categorical must go; big_cat should be removed first.
    assert "big_cat" in info["auto_removed_for_budget"]
    assert estimate_config_effective_dim(cfg2) <= 3

