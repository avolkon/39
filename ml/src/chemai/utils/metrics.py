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


def inverse_rmse_weights(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Веса моделей обратно пропорциональны RMSE на OOF (n × m) → (m,)."""
    n_models = oof.shape[1]
    inv = np.array([1.0 / (rmse(y, oof[:, j]) + 1e-8) for j in range(n_models)])
    return inv / inv.sum()


def blend_oof_predictions(oof: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Взвешенное среднее OOF-предсказаний базовых моделей."""
    return np.asarray(oof @ weights, dtype=np.float64)


def optimize_blend_weights(
    oof_by_target: dict[str, np.ndarray],
    y_mat: np.ndarray,
    *,
    target_names: tuple[str, str, str] = ("IC50", "CC50", "SI"),
    n_random: int = 100,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], float, dict[str, float]]:
    """Подбор весов ансамбля по минимуму competition_score на OOF (не per-target RMSE)."""
    rng = np.random.default_rng(seed)
    n_models = next(iter(oof_by_target.values())).shape[1]

    best_w: dict[str, np.ndarray] = {}
    for i, name in enumerate(target_names):
        best_w[name] = inverse_rmse_weights(oof_by_target[name], y_mat[:, i])

    def _stacked(w_by_target: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack(
            [blend_oof_predictions(oof_by_target[t], w_by_target[t]) for t in target_names],
        )

    best_oof = _stacked(best_w)
    best_score, best_parts = competition_score(y_mat, best_oof)

    for _ in range(n_random):
        trial = {t: rng.dirichlet(np.ones(n_models)) for t in target_names}
        score, parts = competition_score(y_mat, _stacked(trial))
        if score < best_score:
            best_score, best_parts = score, parts
            best_w = trial

    return best_w, best_score, best_parts
