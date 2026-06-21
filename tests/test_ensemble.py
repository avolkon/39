"""Ансамбль: взвешенное усреднение предсказаний."""

from __future__ import annotations

import numpy as np

from chemai.models.ensemble import Ensemble


class _ConstPredictor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)


def test_ensemble_weighted_average_per_target() -> None:
    x = np.zeros((4, 2))
    models = {
        "IC50": {"m_a": _ConstPredictor(2.0), "m_b": _ConstPredictor(4.0)},
        "CC50": {"m_a": _ConstPredictor(10.0)},
        "SI": {"m_a": _ConstPredictor(1.0), "m_b": _ConstPredictor(3.0)},
    }
    weights = {
        "IC50": {"m_a": 1.0, "m_b": 3.0},
        "SI": {"m_a": 1.0, "m_b": 1.0},
    }
    out = Ensemble(models, weights).predict(x)

    assert list(out.columns) == ["IC50", "CC50", "SI"]
    assert len(out) == 4
    assert np.allclose(out["IC50"], 3.5)
    assert np.allclose(out["CC50"], 10.0)
    assert np.allclose(out["SI"], 2.0)
