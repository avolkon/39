"""Загрузка CSV без модификации «на месте» — всегда копия."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from chemai.utils.config import get_config

logger = logging.getLogger(__name__)

TARGETS: tuple[str, ...] = ("IC50", "CC50", "SI")
INDEX_COL = "index"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Файл данных не найден: {path.resolve()}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    return df.copy()


def load_train() -> pd.DataFrame:
    cfg = get_config()
    path = cfg.data_dir / "train.csv"
    df = _read_csv(path)
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"В train.csv отсутствуют таргеты: {missing}")
    na_ratio = df[df.columns[df.columns != INDEX_COL]].isna().mean()
    hi = na_ratio[na_ratio > 0]
    if len(hi):
        top_na = hi.sort_values(ascending=False).head(10).to_dict()
        logger.info("Доля пропусков (топ-10): %s", top_na)
    return df


def load_test() -> pd.DataFrame:
    cfg = get_config()
    return _read_csv(cfg.data_dir / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    cfg = get_config()
    return _read_csv(cfg.data_dir / "sample_submission.csv")


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[list(TARGETS)].copy()
    X = df.drop(columns=list(TARGETS))
    if INDEX_COL in X.columns:
        X = X.drop(columns=[INDEX_COL])
    return X, y
