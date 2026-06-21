"""Группы строк с одинаковым числовым вектором признаков."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def feature_matrix_for_grouping(df: pd.DataFrame) -> np.ndarray:
    num = (
        df.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    return np.round(num, decimals=6)


def feature_row_groups(df: pd.DataFrame) -> np.ndarray:
    """Целочисленный group_id для каждой строки (дубликаты признаков → один id)."""
    matrix = feature_matrix_for_grouping(df)
    _, inv = np.unique(matrix, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def group_key_hash(df: pd.DataFrame) -> list[str]:
    """Стабильный hex-ключ группы для отчётов."""
    matrix = feature_matrix_for_grouping(df)
    keys: list[str] = []
    for row in matrix:
        keys.append(hashlib.sha256(row.tobytes()).hexdigest()[:16])
    return keys
