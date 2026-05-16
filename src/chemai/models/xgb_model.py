"""XGBoost регрессор с early stopping."""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor


def default_xgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }


def train_xgb_regressor(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    early_stopping_rounds: int = 80,
) -> XGBRegressor:
    p = default_xgb_params()
    if params:
        p.update(params)
    model = XGBRegressor(
        random_state=random_state, early_stopping_rounds=early_stopping_rounds, **p
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model
