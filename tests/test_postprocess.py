"""Постобработка предсказаний."""

import pandas as pd

from chemai.utils.postprocess import IC50_CC50_FLOOR, postprocess


def test_postprocess_non_negative() -> None:
    df = pd.DataFrame({"IC50": [-10.0], "CC50": [0.0], "SI": [-1.0]})
    out = postprocess(df)
    assert out["IC50"].iloc[0] >= IC50_CC50_FLOOR
    assert out["CC50"].iloc[0] >= IC50_CC50_FLOOR
    assert out["SI"].iloc[0] >= 0.0
