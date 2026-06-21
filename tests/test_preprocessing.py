"""Тесты: предобработка."""

import numpy as np
import pandas as pd

from chemai.preprocessing.preprocessor import Preprocessor


def test_preprocessor_fit_transform(tmp_path) -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(60),
            "b": rng.standard_normal(60),
        }
    )
    df.loc[:5, "a"] = np.nan
    pre = Preprocessor(missing_threshold=0.5)
    pre.fit(df)
    xt = pre.transform(df)
    assert xt.shape[0] == len(df)
    assert xt.shape[1] == len(pre.get_feature_names())

    p = tmp_path / "p.joblib"
    pre.save(p)
    loaded = Preprocessor.load(p)
    assert np.allclose(loaded.transform(df), xt)


def test_preprocessor_fit_transform_fold_matches_schema() -> None:
    rng = np.random.default_rng(99)
    df = pd.DataFrame({"x": rng.standard_normal(40), "y": rng.standard_normal(40)})
    full = Preprocessor(missing_threshold=0.3)
    full.fit(df)
    fold = Preprocessor(missing_threshold=0.3)
    xt = fold.fit_transform_fold(df.iloc[:25], full)
    assert xt.shape == (25, len(full.get_feature_names()))
