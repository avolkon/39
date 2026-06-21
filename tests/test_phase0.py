"""Тесты Фазы 0: группы дубликатов и stacking."""

from __future__ import annotations

import pandas as pd

from chemai.utils.feature_groups import feature_row_groups


def test_feature_row_groups_identical_rows_share_id() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0],
            "b": [3.0, 3.0, 4.0],
        }
    )
    groups = feature_row_groups(df)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_feature_row_groups_rounding() -> None:
    df = pd.DataFrame({"x": [1.0, 1.0 + 1e-7]})
    groups = feature_row_groups(df)
    assert groups[0] == groups[1]
