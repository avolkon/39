"""Метрики соревнования: RMSE и competition_score."""

import numpy as np

from chemai.utils.metrics import (
    competition_score,
    inverse_rmse_weights,
    optimize_blend_weights,
    rmse,
)


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


def test_inverse_rmse_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    w = inverse_rmse_weights(oof, y)
    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)


def test_optimize_blend_weights_improves_or_matches_inverse_rmse() -> None:
    rng = np.random.default_rng(1)
    n, m = 80, 3
    y_mat = rng.standard_normal((n, 3)) * 100 + 300
    oof_by_target = {
        "IC50": y_mat[:, [0]] + rng.standard_normal((n, m)) * 20,
        "CC50": y_mat[:, [1]] + rng.standard_normal((n, m)) * 20,
        "SI": y_mat[:, [2]] + rng.standard_normal((n, m)) * 5,
    }
    inv_w = {
        t: inverse_rmse_weights(oof_by_target[t], y_mat[:, i])
        for i, t in enumerate(("IC50", "CC50", "SI"))
    }
    inv_oof = np.column_stack(
        [oof_by_target[t] @ inv_w[t] for t in ("IC50", "CC50", "SI")],
    )
    inv_score, _ = competition_score(y_mat, inv_oof)

    opt_w, opt_score, _ = optimize_blend_weights(
        oof_by_target, y_mat, n_random=50, seed=42
    )
    assert opt_score <= inv_score + 1e-9
    assert set(opt_w) == {"IC50", "CC50", "SI"}
