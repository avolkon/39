"""Сквозной прогон train→predict на синтетических CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.predict import predict_pipeline
from chemai.train import train_pipeline
from chemai.utils.config import load_config


def _synthetic_dataset(
    n_train: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def make_rows(n: int, start_idx: int, *, with_targets: bool) -> pd.DataFrame:
        rows: list[dict] = []
        for i in range(n):
            row: dict = {
                "index": start_idx + i,
                "MolLogP": float(rng.standard_normal() + 1.0),
                "TPSA": float(abs(rng.standard_normal()) * 40 + 5),
                "HeavyAtomCount": float(rng.integers(15, 40)),
                "NumAromaticRings": float(rng.integers(0, 4)),
                "RingCount": float(rng.integers(1, 6)),
                "MaxPartialCharge": float(rng.random()),
                "MinPartialCharge": float(-rng.random()),
                "fr_imide": 0.0,
                "fr_sulfone": float(rng.integers(0, 2)),
            }
            for j in range(40):
                row[f"desc_{j}"] = float(rng.standard_normal())
            if with_targets:
                row["IC50"] = float(abs(rng.random()) * 2 + 0.05)
                row["CC50"] = float(abs(rng.random()) * 2 + 0.05)
                row["SI"] = float(abs(rng.random()) * 3)
            rows.append(row)
        return pd.DataFrame(rows)

    train_df = make_rows(n_train, 0, with_targets=True)
    test_df = make_rows(n_test, n_train, with_targets=False)
    return train_df, test_df


def test_train_predict_smoke(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    models = tmp_path / "models"
    sub = tmp_path / "sub"
    for d in (data, models, sub):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    train_df, test_df = _synthetic_dataset(120, 40, rng)
    train_df.to_csv(data / "train.csv", index=False)
    test_df.to_csv(data / "test.csv", index=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                f"CHEM_DATA_DIR={data.as_posix()}",
                f"CHEM_MODELS_DIR={models.as_posix()}",
                f"CHEM_SUBMISSIONS_DIR={sub.as_posix()}",
                "CHEM_N_FOLDS=3",
                "CHEM_N_CLUSTERS=4",
                "CHEM_LOG_TRANSFORM_IC50_CC50=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEM_DATA_DIR", str(data))
    monkeypatch.setenv("CHEM_MODELS_DIR", str(models))
    monkeypatch.setenv("CHEM_SUBMISSIONS_DIR", str(sub))
    monkeypatch.setenv("CHEM_N_FOLDS", "3")
    monkeypatch.setenv("CHEM_N_CLUSTERS", "4")
    monkeypatch.setenv("CHEM_LOG_TRANSFORM_IC50_CC50", "true")

    load_config(env_path)

    train_pipeline()
    out = predict_pipeline()
    assert out.is_file()
    got = pd.read_csv(out)
    assert list(got.columns) == ["index", "IC50", "CC50", "SI"]
    assert len(got) == len(test_df)
