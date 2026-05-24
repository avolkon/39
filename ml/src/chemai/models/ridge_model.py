"""Ridge с подбором alpha на кросс-валидации."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV


def train_ridge_cv(x_train, y_train, *, alphas: np.ndarray | None = None) -> RidgeCV:
    if alphas is None:
        alphas = np.logspace(-4, 4, 25)
    model = RidgeCV(alphas=alphas, cv=None)
    model.fit(x_train, y_train)
    return model
