"""Реестр моделей для ансамбля (9+ алгоритмов).

См. документ: Разработка/Эпики/Анализ_модельных_вариантов.md
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNetCV

from chemai.models.lgb_model import train_lgb_regressor
from chemai.models.ridge_model import train_ridge_cv
from chemai.models.xgb_model import train_xgb_regressor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCandidate:
    """Один тип модели: обучение на фолде и финальное обучение на полном train."""

    name: str
    fit_fold: Callable[..., Any]
    fit_final: Callable[..., Any]
    short_description: str = ""


def _hold_split(
    n: int,
    rng: np.random.Generator,
    frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    n_hold = max(1, int(frac * n))
    hold, trn = order[:n_hold], order[n_hold:]
    return trn, hold


def build_default_candidates(_random_seed: int) -> list[ModelCandidate]:
    """Девять базовых семейств; при установленном catboost — десятый кандидат."""
    def lgb_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        return train_lgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs)

    def lgb_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        return train_lgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs)

    def xgb_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        return train_xgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs)

    def xgb_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        return train_xgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs)

    def ridge_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        _rs: int,
    ) -> Any:
        return train_ridge_cv(x_tr, y_tr)

    def ridge_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        _rs: int,
    ) -> Any:
        return train_ridge_cv(x_full, y_all)

    def elastic_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=rs, max_iter=5000)
        m.fit(x_tr, y_tr)
        return m

    def elastic_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=rs, max_iter=5000)
        m.fit(x_full, y_all)
        return m

    def hgb_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_depth=7,
            min_samples_leaf=15,
            l2_regularization=1e-3,
            random_state=rs,
        )
        m.fit(x_tr, y_tr)
        return m

    def hgb_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_depth=7,
            min_samples_leaf=12,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=rs,
        )
        m.fit(x_full, y_all)
        return m

    def rf_factory(rs: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=rs,
            n_jobs=-1,
        )

    def rf_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = rf_factory(rs)
        m.fit(x_tr, y_tr)
        return m

    def rf_tree_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = rf_factory(rs)
        m.fit(x_full, y_all)
        return m

    def et_factory(rs: int) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=rs,
            n_jobs=-1,
        )

    def et_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = et_factory(rs)
        m.fit(x_tr, y_tr)
        return m

    def et_final_only(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = et_factory(rs)
        m.fit(x_full, y_all)
        return m

    def gbr_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.9,
            random_state=rs,
        )
        m.fit(x_tr, y_tr)
        return m

    def gbr_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=rs,
        )
        m.fit(x_full, y_all)
        return m

    def bayes_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        _rs: int,
    ) -> Any:
        m = BayesianRidge(max_iter=500)
        m.fit(x_tr, y_tr)
        return m

    def bayes_final_fn(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        _x_tr: np.ndarray,
        _y_tr: np.ndarray,
        _x_va: np.ndarray,
        _y_va: np.ndarray,
        _rs: int,
    ) -> Any:
        m = BayesianRidge(max_iter=800)
        m.fit(x_full, y_all)
        return m

    candidates: list[ModelCandidate] = [
        ModelCandidate(
            "lgb",
            lgb_fold,
            lgb_final,
            "LightGBM — бустинг, устойчивость к шуму, ранняя остановка.",
        ),
        ModelCandidate(
            "xgb",
            xgb_fold,
            xgb_final,
            "XGBoost — второй бустинг, иной inductive bias.",
        ),
        ModelCandidate(
            "ridge",
            ridge_fold,
            ridge_final,
            "RidgeCV — сильная регуляризация, baseline при коллинеарности.",
        ),
        ModelCandidate(
            "elastic_net",
            elastic_fold,
            elastic_final,
            "ElasticNetCV — разрежение + L2, другой линейный компромисс.",
        ),
        ModelCandidate(
            "hist_gbrt",
            hgb_fold,
            hgb_final,
            "HistGradientBoosting — нативный sklearn-бустинг на гистограммах.",
        ),
        ModelCandidate(
            "random_forest",
            rf_fold,
            rf_tree_final,
            "RandomForest — бэггинг деревьев, снижение дисперсии.",
        ),
        ModelCandidate(
            "extra_trees",
            et_fold,
            et_final_only,
            "ExtraTrees — случайные пороги, меньше переобучения на малых n.",
        ),
        ModelCandidate(
            "grad_boosting_sklearn",
            gbr_fold,
            gbr_final,
            "GradientBoosting — классический бустинг sklearn.",
        ),
        ModelCandidate(
            "bayesian_ridge",
            bayes_fold,
            bayes_final_fn,
            "BayesianRidge — иной линейный угол с априором.",
        ),
    ]

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.info(
            "CatBoost не установлен — пропуск (pip install catboost или extras catboost).",
        )
        return candidates

    def cat_fold(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = CatBoostRegressor(
            depth=6,
            learning_rate=0.06,
            iterations=1200,
            loss_function="RMSE",
            random_seed=rs,
            verbose=False,
            early_stopping_rounds=60,
        )
        m.fit(x_tr, y_tr, eval_set=(x_va, y_va))
        return m

    def cat_final(
        x_full: np.ndarray,
        y_all: np.ndarray,
        _xf: np.ndarray,
        _yf: np.ndarray,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        rs: int,
    ) -> Any:
        m = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=2000,
            loss_function="RMSE",
            random_seed=rs,
            verbose=False,
            early_stopping_rounds=80,
        )
        m.fit(x_tr, y_tr, eval_set=(x_va, y_va))
        return m

    candidates.append(
        ModelCandidate(
            "catboost",
            cat_fold,
            cat_final,
            "CatBoost — доп. бустинг при наличии пакета.",
        )
    )
    return candidates


def fit_all_final(
    candidate: ModelCandidate,
    x_full: np.ndarray,
    y_all: np.ndarray,
    random_seed: int,
) -> Any:
    rng = np.random.default_rng(random_seed)
    trn, hold = _hold_split(len(x_full), rng)
    return candidate.fit_final(
        x_full,
        y_all,
        x_full,
        y_all,
        x_full[trn],
        y_all[trn],
        x_full[hold],
        y_all[hold],
        random_seed,
    )
