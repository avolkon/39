"""Обёртки для шкалы log1p / обратное преобразование предсказаний."""

from __future__ import annotations

import numpy as np


class Expm1Predictor:
    """Модель обучена на log1p(y); predict возвращает значения в исходной шкале."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def predict(self, x) -> np.ndarray:
        return np.expm1(np.asarray(self.inner.predict(x), dtype=np.float64))


class Log1pWrapper:
    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def fit(self, x, y: np.ndarray) -> Log1pWrapper:
        y_log = np.log1p(np.clip(y, a_min=0.0, a_max=None))
        self.estimator.fit(x, y_log)
        return self

    def predict(self, x) -> np.ndarray:
        raw = self.estimator.predict(x)
        return np.expm1(raw)
