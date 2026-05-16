"""LightGBM с early stopping по валидации."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb


def default_lgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
    }


def train_lgb_regressor(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    stopping_rounds: int = 80,
) -> lgb.LGBMRegressor:
    p = default_lgb_params()
    if params:
        p.update(params)
    model = lgb.LGBMRegressor(random_state=random_state, **p)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model
