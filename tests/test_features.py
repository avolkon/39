"""Тесты: доменные признаки."""

import pandas as pd

from chemai.features.build_features import add_chem_features


def test_add_chem_features_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "MolLogP": [1.5],
            "TPSA": [40.0],
            "HeavyAtomCount": [22],
            "NumAromaticRings": [2],
            "MaxPartialCharge": [0.12],
            "MinPartialCharge": [-0.2],
            "fr_imide": [0.0],
            "fr_sulfone": [2.0],
            "RingCount": [3],
            "FractionCSP3": [0.5],
        }
    )
    out = add_chem_features(df)
    assert "LogP_TPSA" in out.columns
    assert "Arom_Heavy_ratio" in out.columns
    assert "Charge_sum" in out.columns
    assert float(out["fr_imide_flag"].iloc[0]) == 0.0
    assert float(out["fr_sulfone_flag"].iloc[0]) == 1.0
    assert "Ring_LogP" in out.columns
    assert "FractionCSP3_LogP" in out.columns
