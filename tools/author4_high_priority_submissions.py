#!/usr/bin/env python3
r"""
Сабмиты автора №4: высокоприоритетные шаги **без правок базового train.py**.

Файлы в ``CHEM_SUBMISSIONS_DIR``:

- ``submission2_a4.csv`` — OOF stacking (RidgeCV + StandardScaler на OOF),
  **ClusterKFold** как в коде.
- ``submission3_a4.csv`` — как 2 + **смесь SI** с CC50/IC50 (вес на OOF).
- ``submission4_a4.csv`` — **GroupKFold** по дубликатам признаков + stacking + смесь SI.
- ``submission5_a4.csv`` — как 4 + **Optuna** для LightGBM (если ``optuna`` установлена).

Запуск из корня::

    uv run python tools/author4_high_priority_submissions.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemai.features.build_features import add_chem_features
from chemai.models.candidate_models import (
    ModelCandidate,
    build_default_candidates,
    fit_all_final,
)
from chemai.models.lgb_model import default_lgb_params, train_lgb_regressor
from chemai.models.log_wrappers import Expm1Predictor
from chemai.preprocessing.preprocessor import Preprocessor
from chemai.utils.config import get_config, load_config
from chemai.utils.data_loader import (
    INDEX_COL,
    TARGETS,
    load_test,
    load_train,
    split_features_targets,
)
from chemai.utils.logging_utils import setup_logging
from chemai.utils.metrics import competition_score, rmse
from chemai.utils.postprocess import postprocess
from chemai.validation.cv_splitter import ClusterKFold

logger = logging.getLogger("author4_hp")

EPS = 1e-9
SplitMode = Literal["cluster", "group"]


def _y_train_space(y_raw: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.log1p(np.clip(y_raw, 0.0, None))
    return y_raw.copy()


def _pred_to_original(pred: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.clip(np.expm1(np.asarray(pred, dtype=np.float64)), EPS, None)
    return np.asarray(pred, dtype=np.float64)


def feature_row_groups(X: pd.DataFrame) -> np.ndarray:
    num = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    num = np.round(num, decimals=6)
    _, inv = np.unique(num, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def cluster_splits(
    x_raw: pd.DataFrame, cfg, y_ref: np.ndarray
) -> tuple[int, Iterator[tuple[np.ndarray, np.ndarray]]]:
    cv = ClusterKFold(
        n_splits=cfg.n_folds,
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_seed,
    )
    return cfg.n_folds, cv.split(x_raw, y_ref)


def group_splits(
    x_raw: pd.DataFrame, cfg, groups: np.ndarray
) -> tuple[int, Iterator[tuple[np.ndarray, np.ndarray]]]:
    n_uid = len(np.unique(groups))
    n_sp = max(2, min(int(cfg.n_folds), int(n_uid)))
    gkf = GroupKFold(n_splits=n_sp)
    return n_sp, gkf.split(x_raw, groups=groups)


def collect_oof(
    x_raw: pd.DataFrame,
    y_raw: np.ndarray,
    candidates: list[ModelCandidate],
    cfg,
    *,
    seed: int,
    use_log: bool,
    mode: SplitMode,
    groups: np.ndarray | None,
) -> np.ndarray:
    n = len(x_raw)
    c = len(candidates)
    oof = np.full((n, c), np.nan, dtype=np.float64)
    y_s = _y_train_space(y_raw, use_log)

    if mode == "cluster":
        _, split_iter = cluster_splits(x_raw, cfg, y_raw)
    else:
        assert groups is not None
        _, split_iter = group_splits(x_raw, cfg, groups)

    for fold_id, (tr, va) in enumerate(split_iter):
        pre = Preprocessor(cfg.missing_threshold)
        pre.fit(x_raw.iloc[tr])
        x_tr = pre.transform(x_raw.iloc[tr])
        x_va = pre.transform(x_raw.iloc[va])
        yt = y_s[tr]
        y_va_s = y_s[va]
        rs = seed + fold_id
        for j, cand in enumerate(candidates):
            m = cand.fit_fold(x_tr, yt, x_va, y_va_s, rs)
            p = np.asarray(m.predict(x_va), dtype=np.float64)
            oof[va, j] = _pred_to_original(p, use_log)

    if np.isnan(oof).any():
        raise RuntimeError("NaN в OOF-matrix")
    return oof


def fit_meta_scaled(oof: np.ndarray, y_original: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 19))),
        ]
    ).fit(np.asarray(oof, dtype=np.float64), np.asarray(y_original, dtype=np.float64))


def fit_final_models_for_target(
    x_full: np.ndarray,
    y_raw: np.ndarray,
    candidates: list[ModelCandidate],
    cfg_seed: int,
    use_log: bool,
) -> dict[str, Any]:
    ys = _y_train_space(y_raw, use_log)
    out: dict[str, Any] = {}
    for cand in candidates:
        inner = fit_all_final(cand, x_full, ys, cfg_seed)
        out[cand.name] = Expm1Predictor(inner) if use_log else inner
    return out


def blend_si_oof(
    si_stack: np.ndarray,
    ic_hat: np.ndarray,
    cc_hat: np.ndarray,
    y: np.ndarray,
) -> float:
    y_ic = y[:, 0]
    y_cc = y[:, 1]
    y_si = y[:, 2]
    best_w = 0.5
    best = np.inf
    ratio = np.clip(cc_hat, EPS, None) / np.clip(ic_hat, EPS, None)
    for w in np.linspace(0.0, 1.0, 41):
        si_b = w * si_stack + (1.0 - w) * ratio
        score, _ = competition_score(
            np.column_stack([y_ic, y_cc, y_si]),
            np.column_stack([ic_hat, cc_hat, si_b]),
        )
        if score < best:
            best = score
            best_w = float(w)
    logger.info("SI blend: best_w=%.3f joint_OOF_RMSE=%.6f", best_w, best)
    return best_w


def replace_lgb_candidate(
    candidates: list[ModelCandidate],
    overlay: dict[str, Any],
) -> list[ModelCandidate]:
    out: list[ModelCandidate] = []
    for c in candidates:
        if c.name != "lgb":
            out.append(c)
            continue
        oc = dict(overlay)

        def ff(
            x_tr: np.ndarray,
            y_tr: np.ndarray,
            x_va: np.ndarray,
            y_va: np.ndarray,
            rs: int,
            _oc: dict[str, Any] = oc,
        ) -> Any:
            d = default_lgb_params()
            d.update(_oc)
            return train_lgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs, params=d)

        def fin(
            x_full: np.ndarray,
            y_all: np.ndarray,
            _xf: np.ndarray,
            _yf: np.ndarray,
            x_tr: np.ndarray,
            y_tr: np.ndarray,
            x_va: np.ndarray,
            y_va: np.ndarray,
            rs: int,
            _oc: dict[str, Any] = oc,
        ) -> Any:
            d = default_lgb_params()
            d.update(_oc)
            return train_lgb_regressor(x_tr, y_tr, x_va, y_va, random_state=rs, params=d)

        out.append(ModelCandidate("lgb", ff, fin, "LightGBM (overlay)"))

    return out


def tune_lgb_optuna(
    x_raw: pd.DataFrame,
    y_ic_raw: np.ndarray,
    cfg,
    groups: np.ndarray | None,
    mode: SplitMode,
    n_trials: int,
    seed: int,
) -> dict[str, Any] | None:
    try:
        import optuna
    except ImportError:
        logger.warning("Пакет optuna отсутствует — см. extras в pyproject.")
        return None

    ul = cfg.log_transform_ic50_cc50

    base = build_default_candidates(cfg.random_seed)

    def objective(trial: optuna.Trial) -> float:
        overlay = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 55),
            "learning_rate": trial.suggest_float("learning_rate", 0.035, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 12, 32),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 2.0, log=True),
        }
        cands = replace_lgb_candidate(base, overlay)
        oof = collect_oof(
            x_raw,
            y_ic_raw,
            cands,
            cfg,
            seed=seed,
            use_log=ul,
            mode=mode,
            groups=groups,
        )
        meta = fit_meta_scaled(oof, y_ic_raw)
        pred = np.asarray(meta.predict(oof), dtype=np.float64)
        return rmse(y_ic_raw, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    out = {
        "num_leaves": int(study.best_params["num_leaves"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "min_child_samples": int(study.best_params["min_child_samples"]),
        "lambda_l2": float(study.best_params["lambda_l2"]),
    }
    logger.info("Optuna LGB: %s (IC50 OOF stacking RMSE)=%.6f", out, study.best_value)
    return out


def run_variant(
    label: str,
    x_raw: pd.DataFrame,
    y_df: pd.DataFrame,
    x_full_np: np.ndarray,
    x_test_np: np.ndarray,
    cand: list[ModelCandidate],
    cfg,
    idx: np.ndarray,
    *,
    mode: SplitMode,
    groups: np.ndarray | None,
    si_blend: bool,
    out_path: Path,
) -> None:
    metas: dict[str, Pipeline] = {}
    oof_pred: dict[str, np.ndarray] = {}
    finals: dict[str, dict[str, Any]] = {}

    for t in TARGETS:
        y_raw = y_df[t].to_numpy(dtype=np.float64)
        ul = cfg.log_transform_ic50_cc50 and t in ("IC50", "CC50")
        oof = collect_oof(
            x_raw,
            y_raw,
            cand,
            cfg,
            seed=cfg.random_seed,
            use_log=ul,
            mode=mode,
            groups=groups,
        )
        meta = fit_meta_scaled(oof, y_raw)
        metas[t] = meta
        oof_pred[t] = np.asarray(meta.predict(oof), dtype=np.float64)
        finals[t] = fit_final_models_for_target(x_full_np, y_raw, cand, cfg.random_seed, ul)

    y_mat = np.column_stack(
        [
            y_df["IC50"].to_numpy(dtype=np.float64),
            y_df["CC50"].to_numpy(dtype=np.float64),
            y_df["SI"].to_numpy(dtype=np.float64),
        ]
    )

    w = 1.0
    if si_blend:
        w = blend_si_oof(oof_pred["SI"], oof_pred["IC50"], oof_pred["CC50"], y_mat)
        mean_rmse, parts = competition_score(
            y_mat,
            np.column_stack(
                [
                    oof_pred["IC50"],
                    oof_pred["CC50"],
                    w * oof_pred["SI"]
                    + (1.0 - w)
                    * (np.clip(oof_pred["CC50"], EPS, None) / np.clip(oof_pred["IC50"], EPS, None)),
                ]
            ),
        )
    else:
        mean_rmse, parts = competition_score(
            y_mat,
            np.column_stack([oof_pred["IC50"], oof_pred["CC50"], oof_pred["SI"]]),
        )
    logger.info("%s OOF mean RMSE (3 targets): %.6f %s", label, mean_rmse, parts)

    test_stack: dict[str, np.ndarray] = {}
    for t in TARGETS:
        cols = []
        for cm in cand:
            m = finals[t][cm.name]
            cols.append(np.asarray(m.predict(x_test_np), dtype=np.float64))
        test_stack[t] = np.column_stack(cols)

    ic_t = metas["IC50"].predict(test_stack["IC50"])
    cc_t = metas["CC50"].predict(test_stack["CC50"])
    si_s = metas["SI"].predict(test_stack["SI"])
    if si_blend:
        ratio = np.clip(cc_t, EPS, None) / np.clip(ic_t, EPS, None)
        si_t = w * si_s + (1.0 - w) * ratio
    else:
        si_t = si_s

    out = pd.DataFrame({INDEX_COL: idx, "IC50": ic_t, "CC50": cc_t, "SI": si_t})
    out = postprocess(out)
    out.to_csv(out_path, index=False)
    logger.info("Сохранено %s", out_path.resolve())

    side = out_path.with_suffix(".metrics_sidecar.json")
    side.write_text(
        json.dumps(
            {
                "label": label,
                "oof_mean_rmse": float(mean_rmse),
                "oof_parts": {k: float(v) for k, v in parts.items()},
                "si_blend_w": w if si_blend else None,
                "split_mode": mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Author4: submission2..5_a4.csv")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--optuna-trials", type=int, default=14)
    ap.add_argument("--only", type=int, choices=(2, 3, 4, 5), default=None)
    args = ap.parse_args()

    load_config(env_file=args.config)
    setup_logging()
    cfg = get_config()
    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)

    df = load_train()
    x_raw, y_df = split_features_targets(df)
    x_raw = add_chem_features(x_raw)
    groups = feature_row_groups(x_raw)

    full_pre = Preprocessor(cfg.missing_threshold)
    full_pre.fit(x_raw)
    x_full_np = full_pre.transform(x_raw)

    test_df = load_test()
    if INDEX_COL in test_df.columns:
        idx = test_df[INDEX_COL].values
        x_te = test_df.drop(columns=[INDEX_COL])
    else:
        idx = np.arange(len(test_df))
        x_te = test_df.copy()
    x_te = add_chem_features(x_te)
    x_test_np = full_pre.transform(x_te)

    base = build_default_candidates(cfg.random_seed)
    specs: list[tuple[int, SplitMode, bool, bool]] = [
        (2, "cluster", False, False),
        (3, "cluster", True, False),
        (4, "group", True, False),
        (5, "group", True, True),
    ]

    for num, mode, blend, opt in specs:
        if args.only is not None and num != args.only:
            continue
        path = cfg.submissions_dir / f"submission{num}_a4.csv"
        cand = base
        g_arg = groups if mode == "group" else None
        if opt:
            overlay = tune_lgb_optuna(
                x_raw,
                y_df["IC50"].to_numpy(dtype=np.float64),
                cfg,
                groups,
                mode,
                args.optuna_trials,
                cfg.random_seed + 501,
            )
            if overlay is not None:
                cand = replace_lgb_candidate(base, overlay)

        run_variant(
            f"submission{num}_a4",
            x_raw,
            y_df,
            x_full_np,
            x_test_np,
            cand,
            cfg,
            idx,
            mode=mode,
            groups=g_arg,
            si_blend=blend,
            out_path=path,
        )

        if args.only is not None:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
