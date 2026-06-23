"""OOF stacking второго уровня (RidgeCV + StandardScaler на OOF базовых моделей)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemai.models.candidate_models import ModelCandidate, build_default_candidates, fit_all_final
from chemai.models.log_wrappers import Expm1Predictor
from chemai.preprocessing.preprocessor import Preprocessor
from chemai.utils.config import Config
from chemai.utils.data_loader import TARGETS
from chemai.utils.metrics import competition_score, rmse
from chemai.validation.cv_splitter import make_cv_splitter

logger = logging.getLogger(__name__)

EPS = 1e-9


def y_train_space(y_raw: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.log1p(np.clip(y_raw, 0.0, None))
    return y_raw.copy()


def pred_to_original(pred: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.clip(np.expm1(np.asarray(pred, dtype=np.float64)), EPS, None)
    return np.asarray(pred, dtype=np.float64)


def fit_meta_ridge(oof: np.ndarray, y_original: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 19))),
        ]
    ).fit(np.asarray(oof, dtype=np.float64), np.asarray(y_original, dtype=np.float64))


def fit_meta_oof_ridge(
    oof_base: np.ndarray,
    y_original: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, Pipeline]:
    """Nested OOF meta: честная оценка meta-уровня (не in-sample predict на OOF)."""
    n = len(y_original)
    oof_meta = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in kf.split(oof_base):
        meta = fit_meta_ridge(oof_base[tr], y_original[tr])
        oof_meta[va] = meta.predict(oof_base[va])
    if np.isnan(oof_meta).any():
        msg = "NaN в nested meta-OOF — проверьте n_splits"
        raise RuntimeError(msg)
    final_meta = fit_meta_ridge(oof_base, y_original)
    return oof_meta, final_meta


def collect_oof_for_target(
    x_raw: pd.DataFrame,
    y_raw: np.ndarray,
    candidates: list[ModelCandidate],
    cfg: Config,
    *,
    full_pre: Preprocessor,
    use_log: bool,
) -> np.ndarray:
    """OOF-предсказания базовых моделей (n_samples × n_candidates) в исходной шкале таргета."""
    n = len(x_raw)
    c = len(candidates)
    oof = np.full((n, c), np.nan, dtype=np.float64)
    y_s = y_train_space(y_raw, use_log)

    cv = make_cv_splitter(cfg)

    for fold_id, (tr, va) in enumerate(cv.split(x_raw, y_raw)):
        pre = Preprocessor(cfg.missing_threshold)
        pre.fit_fold(x_raw.iloc[tr], full_pre)
        x_tr = pre.transform(x_raw.iloc[tr])
        x_va = pre.transform(x_raw.iloc[va])
        yt = y_s[tr]
        y_va_s = y_s[va]
        rs = cfg.random_seed + fold_id
        for j, cand in enumerate(candidates):
            model = cand.fit_fold(x_tr, yt, x_va, y_va_s, rs)
            p = np.asarray(model.predict(x_va), dtype=np.float64)
            oof[va, j] = pred_to_original(p, use_log)

    if np.isnan(oof).any():
        msg = "NaN в OOF-матрице — проверьте CV и кандидатов"
        raise RuntimeError(msg)
    return oof


def fit_base_final_models(
    x_full: np.ndarray,
    y_raw: np.ndarray,
    candidates: list[ModelCandidate],
    cfg_seed: int,
    use_log: bool,
) -> dict[str, Any]:
    ys = y_train_space(y_raw, use_log)
    out: dict[str, Any] = {}
    for cand in candidates:
        inner = fit_all_final(cand, x_full, ys, cfg_seed)
        out[cand.name] = Expm1Predictor(inner) if use_log else inner
    return out


def predict_stacked_test(
    x_test: np.ndarray,
    meta_models: dict[str, Pipeline],
    base_models: dict[str, dict[str, Any]],
    candidate_names: list[str],
) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for target in TARGETS:
        stack_cols = [
            np.asarray(base_models[target][name].predict(x_test), dtype=np.float64)
            for name in candidate_names
        ]
        oof_like = np.column_stack(stack_cols)
        cols[target] = np.asarray(meta_models[target].predict(oof_like), dtype=np.float64)
    return pd.DataFrame(cols)


def blend_si_weight(
    si_stack: np.ndarray,
    ic_hat: np.ndarray,
    cc_hat: np.ndarray,
    y_mat: np.ndarray,
    *,
    n_grid: int = 41,
) -> tuple[float, float]:
    """Подбор w: SI = w·stack + (1-w)·CC50/IC50 по минимуму competition_score на OOF."""
    y_ic, y_cc, y_si = y_mat[:, 0], y_mat[:, 1], y_mat[:, 2]
    ratio = np.clip(cc_hat, EPS, None) / np.clip(ic_hat, EPS, None)
    best_w = 1.0
    best_score = np.inf
    for w in np.linspace(0.0, 1.0, n_grid):
        si_b = w * si_stack + (1.0 - w) * ratio
        score, _ = competition_score(
            np.column_stack([y_ic, y_cc, y_si]),
            np.column_stack([ic_hat, cc_hat, si_b]),
        )
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w, float(best_score)


def run_oof_stacking_cv(
    x_raw: pd.DataFrame,
    y_df: pd.DataFrame,
    x_full: np.ndarray,
    cfg: Config,
    *,
    full_pre: Preprocessor,
    si_blend: bool = False,
    fit_final: bool = True,
) -> dict[str, Any]:
    """Полный цикл OOF stacking по трём таргетам; возвращает метрики и артефакты."""
    candidates = build_default_candidates(cfg.random_seed)
    names = [c.name for c in candidates]

    oof_pred: dict[str, np.ndarray] = {}
    meta_models: dict[str, Pipeline] = {}
    base_models: dict[str, dict[str, Any]] = {}
    cv_report: dict[str, float] = {}

    for target in TARGETS:
        y_raw = y_df[target].to_numpy(dtype=np.float64)
        use_log = cfg.log_transform_ic50_cc50 and target in ("IC50", "CC50")
        oof_base = collect_oof_for_target(
            x_raw, y_raw, candidates, cfg, full_pre=full_pre, use_log=use_log
        )
        oof_meta, meta = fit_meta_oof_ridge(
            oof_base,
            y_raw,
            n_splits=cfg.n_folds,
            random_state=cfg.random_seed,
        )
        meta_models[target] = meta
        oof_pred[target] = oof_meta
        cv_report[target] = float(rmse(y_raw, oof_pred[target]))
        if fit_final:
            base_models[target] = fit_base_final_models(
                x_full, y_raw, candidates, cfg.random_seed, use_log
            )

    y_mat = np.column_stack(
        [y_df[t].to_numpy(dtype=np.float64) for t in TARGETS],
    )

    si_blend_w: float | None = None
    if si_blend:
        si_blend_w, _ = blend_si_weight(oof_pred["SI"], oof_pred["IC50"], oof_pred["CC50"], y_mat)
        ratio = np.clip(oof_pred["CC50"], EPS, None) / np.clip(oof_pred["IC50"], EPS, None)
        si_final = si_blend_w * oof_pred["SI"] + (1.0 - si_blend_w) * ratio
        oof_matrix = np.column_stack([oof_pred["IC50"], oof_pred["CC50"], si_final])
    else:
        oof_matrix = np.column_stack([oof_pred[t] for t in TARGETS])

    mean_score, parts = competition_score(y_mat, oof_matrix)

    return {
        "candidate_names": names,
        "meta_models_by_target": meta_models,
        "base_models_by_target": base_models,
        "oof_pred_by_target": oof_pred,
        "cv_mean_rmse": cv_report,
        "oof_competition_score": float(mean_score),
        "oof_parts": parts,
        "si_blend_w": si_blend_w,
    }
