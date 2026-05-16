"""Кластерное CV."""

import numpy as np
import pandas as pd

from chemai.validation.cv_splitter import ClusterKFold


def test_cluster_kfold_coverage() -> None:
    rng = np.random.default_rng(0)
    n = 120
    x = pd.DataFrame(
        {
            "MolLogP": rng.standard_normal(n),
            "RingCount": rng.integers(0, 8, size=n),
            "TPSA": rng.random(n) * 100,
            "HeavyAtomCount": rng.integers(10, 50, size=n),
            "extra": rng.standard_normal(n),
        }
    )
    cv = ClusterKFold(n_splits=5, n_clusters=6, random_state=42)
    seen = np.zeros(n, dtype=bool)
    for tr, va in cv.split(x):
        assert len(set(tr) & set(va)) == 0
        seen[va] = True
    assert seen.all()
