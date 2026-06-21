"""Метрики соревнования: RMSE и competition_score."""

import numpy as np

from chemai.utils.metrics import competition_score, rmse


def test_rmse_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_competition_score_is_mean_of_column_rmse() -> None:
    y_true = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )
    y_pred = np.array(
        [
            [2.0, 11.0, 100.0],
            [3.0, 22.0, 200.0],
            [4.0, 33.0, 300.0],
        ]
    )
    score, parts = competition_score(y_true, y_pred)
    expected = float(np.mean([rmse(y_true[:, i], y_pred[:, i]) for i in range(3)]))
    assert score == expected
    assert set(parts) == {"IC50", "CC50", "SI"}
    assert parts["IC50"] == rmse(y_true[:, 0], y_pred[:, 0])
    assert parts["SI"] == 0.0
