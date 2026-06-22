"""Обучение: baseline-ансамбль или OOF stacking (Фаза 0 baseline v2)."""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np

from chemai.features.build_features import add_chem_features
from chemai.models.candidate_models import build_default_candidates, fit_all_final
from chemai.models.log_wrappers import Expm1Predictor
from chemai.models.stacking import run_oof_stacking_cv
from chemai.preprocessing.preprocessor import Preprocessor
from chemai.utils.config import get_config
from chemai.utils.data_loader import TARGETS, load_train, split_features_targets
from chemai.utils.metrics import (
    optimize_blend_weights,
    rmse,
)
from chemai.validation.cv_splitter import make_cv_splitter

logger = logging.getLogger(__name__)


def _y_train_space(y_raw: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.log1p(np.clip(y_raw, 0.0, None))
    return y_raw.copy()


def _collect_baseline_oof(
    x_raw,
    y_raw,
    candidates,
    cfg,
    full_pre,
    cv,
    use_log: bool,
) -> np.ndarray:
    n = len(x_raw)
    m = len(candidates)
    oof = np.full((n, m), np.nan, dtype=np.float64)
    y_s = _y_train_space(y_raw, use_log)

    for tr, va in cv.split(x_raw, y_raw):
        pre = Preprocessor(cfg.missing_threshold)
        pre.fit_fold(x_raw.iloc[tr], full_pre)
        x_tr = pre.transform(x_raw.iloc[tr])
        x_va = pre.transform(x_raw.iloc[va])
        y_tr = y_s[tr]
        y_va_s = y_s[va]

        for j, cand in enumerate(candidates):
            model = cand.fit_fold(x_tr, y_tr, x_va, y_va_s, cfg.random_seed)
            p = np.asarray(model.predict(x_va), dtype=np.float64)
            oof[va, j] = np.expm1(p) if use_log else p

    if np.isnan(oof).any():
        msg = "NaN в baseline OOF — проверьте CV"
        raise RuntimeError(msg)
    return oof


def _train_baseline(
    x_raw,
    y_df,
    x_full,
    full_pre,
    cfg,
    candidates,
) -> tuple[dict, dict, dict, dict]:
    cv = make_cv_splitter(cfg)

    bundle: dict[str, dict[str, object]] = {}
    weights: dict[str, dict[str, float]] = {}
    cv_report: dict[str, dict[str, float]] = {}
    oof_by_target: dict[str, np.ndarray] = {}

    for target in TARGETS:
        y_raw = y_df[target].to_numpy(dtype=np.float64)
        use_log = cfg.log_transform_ic50_cc50 and target in ("IC50", "CC50")

        fold_scores: dict[str, list[float]] = {c.name: [] for c in candidates}

        for tr, va in cv.split(x_raw, y_raw):
            pre = Preprocessor(cfg.missing_threshold)
            pre.fit_fold(x_raw.iloc[tr], full_pre)
            x_tr = pre.transform(x_raw.iloc[tr])
            x_va = pre.transform(x_raw.iloc[va])

            y_tr = _y_train_space(y_raw[tr], use_log)
            y_va_s = _y_train_space(y_raw[va], use_log)

            for cand in candidates:
                m = cand.fit_fold(x_tr, y_tr, x_va, y_va_s, cfg.random_seed)
                p = m.predict(x_va)
                pred_o = np.expm1(p) if use_log else p
                fold_scores[cand.name].append(rmse(y_raw[va], pred_o))

        cv_report[target] = {k: float(np.mean(v)) for k, v in fold_scores.items()}
        oof_by_target[target] = _collect_baseline_oof(
            x_raw, y_raw, candidates, cfg, full_pre, cv, use_log
        )
        logger.info("CV средние RMSE (%s): %s", target, cv_report[target])

        target_models: dict[str, object] = {}
        for cand in candidates:
            final_m = fit_all_final(cand, x_full, _y_train_space(y_raw, use_log), cfg.random_seed)
            target_models[cand.name] = Expm1Predictor(final_m) if use_log else final_m

        bundle[target] = target_models

    y_mat = np.column_stack([y_df[t].to_numpy(dtype=np.float64) for t in TARGETS])
    opt_w, oof_score, oof_parts = optimize_blend_weights(oof_by_target, y_mat, seed=cfg.random_seed)

    for target in TARGETS:
        w_arr = opt_w[target]
        weights[target] = {c.name: float(w_arr[i]) for i, c in enumerate(candidates)}

    logger.info(
        "Baseline OOF competition_score=%.4f parts=%s; веса по таргетам: %s",
        oof_score,
        oof_parts,
        weights,
    )

    extra = {
        "oof_competition_score": oof_score,
        "oof_parts": oof_parts,
    }
    return bundle, weights, cv_report, extra


def _train_stacking(x_raw, y_df, x_full, full_pre, cfg) -> tuple[dict, dict, dict, dict]:
    result = run_oof_stacking_cv(x_raw, y_df, x_full, cfg, full_pre=full_pre, si_blend=False)
    logger.info(
        "OOF stacking competition_score=%.4f parts=%s",
        result["oof_competition_score"],
        result["oof_parts"],
    )
    extra = {
        "stacking_mode": True,
        "meta_models_by_target": result["meta_models_by_target"],
        "base_models_by_target": result["base_models_by_target"],
        "candidate_names": result["candidate_names"],
        "oof_competition_score": result["oof_competition_score"],
        "oof_parts": result["oof_parts"],
    }
    return {}, {}, result["cv_mean_rmse"], extra


def train_pipeline() -> None:
    cfg = get_config()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    df = load_train()
    x_raw, y_df = split_features_targets(df)
    x_raw = add_chem_features(x_raw)

    full_pre = Preprocessor(cfg.missing_threshold)
    full_pre.fit(x_raw)
    x_full = full_pre.transform(x_raw)
    full_pre.save(cfg.models_dir / "preprocessor.joblib")

    candidates = build_default_candidates(cfg.random_seed)
    names = ", ".join(c.name for c in candidates)
    logger.info("Кандидатов моделей: %d (%s)", len(candidates), names)

    if cfg.use_stacking:
        bundle, weights, cv_report, stacking_extra = _train_stacking(
            x_raw, y_df, x_full, full_pre, cfg
        )
        artifact = {
            "preprocessor": full_pre,
            "stacking_mode": True,
            "models_by_target": {},
            "weights_by_target": {},
            "meta_models_by_target": stacking_extra["meta_models_by_target"],
            "base_models_by_target": stacking_extra["base_models_by_target"],
            "targets_order": list(TARGETS),
            "candidate_names": stacking_extra["candidate_names"],
        }
        metrics_payload = {
            "mode": "stacking_v2",
            "cv_mean_rmse": cv_report,
            "oof_competition_score": stacking_extra["oof_competition_score"],
            "oof_parts": stacking_extra["oof_parts"],
        }
    else:
        bundle, weights, cv_report, baseline_extra = _train_baseline(
            x_raw, y_df, x_full, full_pre, cfg, candidates
        )
        artifact = {
            "preprocessor": full_pre,
            "stacking_mode": False,
            "models_by_target": bundle,
            "weights_by_target": weights,
            "targets_order": list(TARGETS),
            "candidate_names": [c.name for c in candidates],
        }
        metrics_payload = {
            "mode": "baseline",
            "cv_mean_rmse": cv_report,
            "weights": weights,
            "oof_competition_score": baseline_extra["oof_competition_score"],
            "oof_parts": baseline_extra["oof_parts"],
        }

    bundle_path = cfg.models_dir / "pipeline_bundle.joblib"
    joblib.dump(artifact, bundle_path)
    logger.info("Сохранено: %s", bundle_path)

    metrics_path = cfg.models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Метрики CV: %s", metrics_path)
