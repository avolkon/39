"""Оценка регрессора на произвольном CV-сплиттере."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def evaluate_regressor_cv(
    build_estimator: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    cv,
) -> tuple[float, list[float]]:
    """Оценка CV: новый estimator на каждый фолд; возврат mean_rmse и списка по фолдам."""
    rmses: list[float] = []
    for fold_id, (tr, va) in enumerate(cv.split(X, y)):
        est = build_estimator()
        est.fit(X.iloc[tr], y[tr])
        pred = est.predict(X.iloc[va])
        rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
        rmses.append(rmse)
        logger.info("CV fold %d RMSE=%.6f", fold_id, rmse)
    return float(np.mean(rmses)), rmses
