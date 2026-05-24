"""Ансамбль: взвешенное среднее предсказаний по имени модели."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Ensemble:
    """Хранит по каждому таргету словарь обученных моделей и опциональные веса."""

    def __init__(
        self,
        models_by_target: dict[str, dict[str, Any]],
        weights_by_target: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.models_by_target = models_by_target
        self.weights_by_target = weights_by_target or {}

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {}
        for target, bundle in self.models_by_target.items():
            weights = self.weights_by_target.get(target, {})
            preds = []
            wts = []
            for name, model in bundle.items():
                preds.append(np.asarray(model.predict(x), dtype=np.float64))
                wts.append(float(weights.get(name, 1.0)))
            stacked = np.stack(preds, axis=0)
            w = np.asarray(wts, dtype=np.float64)
            cols[target] = np.average(stacked, axis=0, weights=w)
        return pd.DataFrame(cols)
