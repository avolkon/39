"""Разбиение с учётом химического разнообразия (кластеры по дескрипторам)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_FEATURES = ("MolLogP", "RingCount", "TPSA", "HeavyAtomCount")


class ClusterKFold:
    """Кластеризация KMeans по подмножеству признаков; фолды разводят кластеры."""

    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_clusters: int = 5,
        random_state: int = 42,
        cluster_feature_names: tuple[str, ...] = DEFAULT_CLUSTER_FEATURES,
    ) -> None:
        self.n_splits = n_splits
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_feature_names = cluster_feature_names

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        matrix, used_cols = self._cluster_matrix(X)
        logger.info("ClusterKFold: признаки для кластеризации: %s", used_cols)

        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = km.fit_predict(matrix)

        rng = np.random.default_rng(self.random_state)
        cluster_ids = np.arange(self.n_clusters)
        rng.shuffle(cluster_ids)
        fold_of_cluster = np.zeros(self.n_clusters, dtype=np.int32)
        for i, cid in enumerate(cluster_ids):
            fold_of_cluster[cid] = i % self.n_splits

        sample_fold = fold_of_cluster[labels]
        idx = np.arange(len(X))

        for k in range(self.n_splits):
            val_mask = sample_fold == k
            train_idx = idx[~val_mask]
            val_idx = idx[val_mask]
            if len(val_idx) == 0:
                logger.warning("ClusterKFold: фолд %d пуст — проверьте n_clusters/n_splits", k)
            yield train_idx, val_idx

    def _cluster_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols = [c for c in self.cluster_feature_names if c in X.columns]
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

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits
