"""Метрики соревнования: усреднённая RMSE по трём таргетам."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def competition_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    names: tuple[str, str, str] = ("IC50", "CC50", "SI"),
) -> tuple[float, dict[str, float]]:
    """Среднее RMSE по трём столбцам (матрицы формы (n, 3))."""
    parts = {}
    for i, name in enumerate(names):
        parts[name] = rmse(y_true[:, i], y_pred[:, i])
    return float(np.mean(list(parts.values()))), parts
