"""Реестр моделей: минимум 9 кандидатов без catboost."""

from chemai.models.candidate_models import build_default_candidates


def test_at_least_nine_candidates_without_catboost() -> None:
    cands = build_default_candidates(42)
    names = [c.name for c in cands]
    assert len(names) >= 9
    assert "lgb" in names and "bayesian_ridge" in names
