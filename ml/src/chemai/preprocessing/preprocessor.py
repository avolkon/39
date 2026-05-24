"""Обучение трансформаций только на train; transform для val/test."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

VARIANCE_EPS = 1e-12


class Preprocessor:
    """Медианное заполнение, отбор колонок, StandardScaler."""

    def __init__(self, missing_threshold: float = 0.3) -> None:
        self.missing_threshold = missing_threshold
        self._medians: pd.Series | None = None
        self._feature_columns: list[str] | None = None
        self._scaler = StandardScaler()
        self._dropped_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("Нет числовых колонок для обучения Preprocessor")

        miss_ratio = numeric.isna().mean()
        drop_missing = miss_ratio[miss_ratio > self.missing_threshold].index.tolist()

        medians = numeric.median(numeric_only=True)
        filled = numeric.fillna(medians)
        variances = filled.var()
        drop_zero_var = variances[variances <= VARIANCE_EPS].index.tolist()

        self._dropped_columns = sorted(set(drop_missing + drop_zero_var))
        self._feature_columns = [c for c in numeric.columns if c not in self._dropped_columns]
        self._medians = medians.reindex(self._feature_columns)

        train_matrix = numeric[self._feature_columns].fillna(self._medians)
        self._scaler.fit(train_matrix)

        logger.info(
            "Preprocessor: kept %d cols, dropped %d (missing/zero-var)",
            len(self._feature_columns),
            len(self._dropped_columns),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._feature_columns is None or self._medians is None:
            raise RuntimeError("Preprocessor: сначала вызовите fit()")
        numeric = df.select_dtypes(include=[np.number]).copy()
        missing_feats = set(self._feature_columns) - set(numeric.columns)
        if missing_feats:
            raise ValueError(f"Не хватает числовых признаков: {sorted(missing_feats)[:10]}…")
        x = numeric[self._feature_columns].fillna(self._medians)
        return self._scaler.transform(x)

    def get_feature_names(self) -> list[str]:
        if self._feature_columns is None:
            return []
        return list(self._feature_columns)

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "missing_threshold": self.missing_threshold,
            "medians": self._medians,
            "feature_columns": self._feature_columns,
            "dropped_columns": self._dropped_columns,
            "scaler": self._scaler,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        logger.info("Preprocessor сохранён: %s", path)

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        payload = joblib.load(path)
        inst = cls(missing_threshold=payload["missing_threshold"])
        inst._medians = payload["medians"]
        inst._feature_columns = payload["feature_columns"]
        inst._dropped_columns = payload["dropped_columns"]
        inst._scaler = payload["scaler"]
        return inst
