"""Разбиение CV: кластеры без утечки KMeans и группы дубликатов признаков."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from chemai.utils.config import Config
from chemai.utils.feature_groups import feature_row_groups

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_FEATURES = ("MolLogP", "RingCount", "TPSA", "HeavyAtomCount")


def _cluster_matrix(
    X: pd.DataFrame,
    cluster_feature_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in cluster_feature_names if c in X.columns]
    if len(cols) >= 2:
        sub = X[cols].select_dtypes(include=[np.number])
    else:
        sub = X.select_dtypes(include=[np.number])
        cols = list(sub.columns[: min(10, sub.shape[1])])
        sub = sub[cols] if cols else sub.iloc[:, : min(10, sub.shape[1])]
        cols = list(sub.columns)

    sub = sub.replace([np.inf, -np.inf], np.nan)
    med = sub.median(numeric_only=True)
    filled = sub.fillna(med)
    return filled.to_numpy(dtype=np.float64), cols


class LeakFreeClusterKFold:
    """R2: KMeans только на train-подвыборке; фолды — GroupKFold по кластерам."""

    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_clusters: int = 5,
        random_state: int = 42,
        cluster_feature_names: tuple[str, ...] = DEFAULT_CLUSTER_FEATURES,
        fit_fraction: float = 0.75,
    ) -> None:
        self.n_splits = n_splits
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_feature_names = cluster_feature_names
        self.fit_fraction = fit_fraction

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        matrix, used_cols = _cluster_matrix(X, self.cluster_feature_names)
        logger.info("LeakFreeClusterKFold: признаки для кластеризации: %s", used_cols)

        n = len(X)
        rng = np.random.default_rng(self.random_state)
        fit_size = max(self.n_clusters * 2, int(self.fit_fraction * n))
        fit_size = min(fit_size, n - 1)
        fit_idx = rng.choice(n, size=fit_size, replace=False)

        km = KMeans(
            n_clusters=min(self.n_clusters, fit_size),
            random_state=self.random_state,
            n_init="auto",
        )
        km.fit(matrix[fit_idx])
        groups = km.predict(matrix)

        n_unique = len(np.unique(groups))
        n_splits_eff = min(self.n_splits, n_unique)
        if n_splits_eff < self.n_splits:
            logger.warning(
                "LeakFreeClusterKFold: уникальных кластеров %d < n_splits=%d",
                n_unique,
                self.n_splits,
            )

        gkf = GroupKFold(n_splits=n_splits_eff)
        yield from gkf.split(X, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits


class DuplicateGroupKFold:
    """1.3: group_id = hash(round(X, 6)); одна группа не пересекает train/val."""

    def __init__(
        self,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = feature_row_groups(X)
        n_unique = len(np.unique(groups))
        n_splits_eff = min(self.n_splits, n_unique)
        if n_splits_eff < self.n_splits:
            logger.warning(
                "DuplicateGroupKFold: уникальных групп %d < n_splits=%d",
                n_unique,
                self.n_splits,
            )
        gkf = GroupKFold(n_splits=n_splits_eff)
        yield from gkf.split(X, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits


# Обратная совместимость: старое имя без глобального fit_predict на всём X.
ClusterKFold = LeakFreeClusterKFold


def make_cv_splitter(cfg: Config) -> LeakFreeClusterKFold | DuplicateGroupKFold:
    """Фабрика CV по CHEM_CV_STRATEGY: cluster | duplicate_group."""
    if cfg.cv_strategy == "duplicate_group":
        return DuplicateGroupKFold(n_splits=cfg.n_folds, random_state=cfg.random_seed)
    return LeakFreeClusterKFold(
        n_splits=cfg.n_folds,
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_seed,
    )
