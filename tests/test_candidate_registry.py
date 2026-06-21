"""Реестр моделей: состав build_default_candidates."""

from chemai.models.candidate_models import build_default_candidates

EXPECTED_BASE = (
    "lgb",
    "xgb",
    "ridge",
    "elastic_net",
    "hist_gbrt",
    "random_forest",
    "extra_trees",
    "grad_boosting_sklearn",
    "bayesian_ridge",
)


def test_default_candidates_base_registry() -> None:
    cands = build_default_candidates(42)
    names = [c.name for c in cands]
    assert names[: len(EXPECTED_BASE)] == list(EXPECTED_BASE)
    assert len(names) >= len(EXPECTED_BASE)
    assert len(names) == len(set(names))
