"""Инференс: загрузка артефактов, тот же feature engineering и submission CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from chemai.features.build_features import add_chem_features
from chemai.models.ensemble import Ensemble
from chemai.utils.config import get_config
from chemai.utils.data_loader import INDEX_COL, load_test
from chemai.utils.postprocess import postprocess

logger = logging.getLogger(__name__)


def predict_pipeline(*, bundle_path: Path | None = None) -> Path:
    cfg = get_config()
    path = bundle_path or (cfg.models_dir / "pipeline_bundle.joblib")
    if not path.is_file():
        msg = f"Нет обученного бандла: {path}. Сначала выполните --train."
        raise FileNotFoundError(msg)

    artifact = joblib.load(path)
    pre = artifact["preprocessor"]
    models = artifact["models_by_target"]
    weights = artifact["weights_by_target"]

    test_df = load_test()
    if INDEX_COL in test_df.columns:
        idx = test_df[INDEX_COL].values
        x_df = test_df.drop(columns=[INDEX_COL])
    else:
        idx = pd.RangeIndex(stop=len(test_df))
        x_df = test_df.copy()

    x_df = add_chem_features(x_df)
    x_t = pre.transform(x_df)

    ensemble = Ensemble(models, weights)
    preds = ensemble.predict(x_t)
    out_df = pd.DataFrame({INDEX_COL: idx, **{c: preds[c].values for c in preds.columns}})
    out_df = postprocess(out_df, si_domain_blend=cfg.si_domain_blend)
    cols = [INDEX_COL, "IC50", "CC50", "SI"]
    out_df = out_df[cols]

    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)
    submission = cfg.submissions_dir / "final_submission.csv"
    out_df.to_csv(submission, index=False)
    logger.info("Submission: %s", submission.resolve())
    return submission
