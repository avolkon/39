"""Фаза 1: честная CV, preprocessor fold, stacking shapes."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from chemai.models.stacking import fit_meta_oof_ridge, fit_meta_ridge
from chemai.preprocessing.preprocessor import Preprocessor
from chemai.utils.feature_groups import feature_row_groups
from chemai.validation.cv_splitter import (
    DuplicateGroupKFold,
    LeakFreeClusterKFold,
    make_cv_splitter,
)


def _sample_x(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "MolLogP": rng.standard_normal(n),
            "RingCount": rng.integers(0, 8, size=n),
            "TPSA": rng.random(n) * 100,
            "HeavyAtomCount": rng.integers(10, 50, size=n),
            "extra": rng.standard_normal(n),
        }
    )


def test_preprocessor_fit_fold_fixed_columns() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(80),
            "b": rng.standard_normal(80),
            "c": rng.standard_normal(80),
        }
    )
    df.loc[:10, "a"] = np.nan

    full = Preprocessor(missing_threshold=0.5)
    full.fit(df)
    schema_cols = full.get_feature_names()

    fold_pre = Preprocessor(missing_threshold=0.5)
    fold_pre.fit_fold(df.iloc[:50], full)
    assert fold_pre.get_feature_names() == schema_cols

    xt = fold_pre.transform(df.iloc[50:])
    assert xt.shape[1] == len(schema_cols)


def test_duplicate_group_kfold_no_group_overlap() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
            "b": [5.0, 5.0, 6.0, 7.0, 7.0, 8.0],
        }
    )
    groups = feature_row_groups(df)
    cv = DuplicateGroupKFold(n_splits=2, random_state=42)
    for tr, va in cv.split(df):
        tr_g = set(groups[tr])
        va_g = set(groups[va])
        assert tr_g.isdisjoint(va_g)
        assert len(set(tr) & set(va)) == 0


def test_leak_free_cluster_kfold_no_full_matrix_fit_predict() -> None:
    x = _sample_x()
    with patch.object(KMeans, "fit_predict") as mock_fp:
        list(LeakFreeClusterKFold(n_splits=5, n_clusters=6, random_state=42).split(x))
        mock_fp.assert_not_called()


def test_leak_free_cluster_kfold_covers_all_samples() -> None:
    x = _sample_x()
    cv = LeakFreeClusterKFold(n_splits=5, n_clusters=6, random_state=42)
    seen = np.zeros(len(x), dtype=bool)
    for tr, va in cv.split(x):
        assert len(set(tr) & set(va)) == 0
        seen[va] = True
    assert seen.all()


def test_make_cv_splitter_duplicate_group() -> None:
    from chemai.utils.config import load_config

    cfg = load_config()
    cfg.cv_strategy = "duplicate_group"
    cv = make_cv_splitter(cfg)
    assert isinstance(cv, DuplicateGroupKFold)


def test_nested_meta_oof_shape_and_no_nan() -> None:
    rng = np.random.default_rng(7)
    n, m = 60, 4
    oof_base = rng.standard_normal((n, m))
    y = rng.standard_normal(n) * 100 + 500
    oof_meta, meta = fit_meta_oof_ridge(oof_base, y, n_splits=5, random_state=42)
    assert oof_meta.shape == (n,)
    assert not np.isnan(oof_meta).any()
    assert meta.predict(oof_base).shape == (n,)


def test_meta_ridge_in_sample_differs_from_nested_oof() -> None:
    """In-sample meta predict не должен совпадать с nested OOF (sanity)."""
    rng = np.random.default_rng(8)
    n, m = 80, 3
    oof_base = rng.standard_normal((n, m))
    y = rng.standard_normal(n) * 50 + 200
    oof_meta, _ = fit_meta_oof_ridge(oof_base, y, n_splits=5, random_state=0)
    in_sample = fit_meta_ridge(oof_base, y).predict(oof_base)
    assert not np.allclose(oof_meta, in_sample)
