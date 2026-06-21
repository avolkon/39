"""Диагностика Фазы 0: дубликаты, adversarial validation, SI ablation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemai.features.build_features import add_chem_features
from chemai.models.stacking import (
    blend_si_weight,
    run_oof_stacking_cv,
)
from chemai.preprocessing.preprocessor import Preprocessor
from chemai.utils.config import Config, get_config
from chemai.utils.data_loader import TARGETS, load_test, load_train, split_features_targets
from chemai.utils.feature_groups import feature_row_groups
from chemai.utils.metrics import competition_score

logger = logging.getLogger(__name__)

EPS = 1e-9


def analyze_duplicate_groups(
    x_raw: pd.DataFrame,
    y_df: pd.DataFrame,
    *,
    figures_dir: Path,
) -> dict[str, Any]:
    groups = feature_row_groups(x_raw)
    n_groups = int(groups.max()) + 1
    sizes = np.bincount(groups)
    multi_mask = sizes[groups] > 1
    n_dup_rows = int(multi_mask.sum())
    n_multi_groups = int((sizes > 1).sum())

    target_stats: dict[str, Any] = {}
    for target in TARGETS:
        y = y_df[target].to_numpy(dtype=np.float64)
        rel_spread: list[float] = []
        n_conflict = 0
        for gid in np.where(sizes > 1)[0]:
            vals = y[groups == gid]
            if len(vals) < 2:
                continue
            spread = float(np.max(vals) - np.min(vals))
            rel = spread / (float(np.mean(np.abs(vals))) + EPS)
            rel_spread.append(rel)
            if spread > 1e-6:
                n_conflict += 1
        target_stats[target] = {
            "groups_with_spread_gt_1e-6": n_conflict,
            "median_relative_spread_in_multi_groups": float(np.median(rel_spread))
            if rel_spread
            else 0.0,
            "max_relative_spread": float(np.max(rel_spread)) if rel_spread else 0.0,
        }

    si = y_df["SI"].to_numpy(dtype=np.float64)
    ic = y_df["IC50"].to_numpy(dtype=np.float64)
    cc = y_df["CC50"].to_numpy(dtype=np.float64)
    ratio = cc / np.clip(ic, EPS, None)
    si_ratio_max = float(np.max(np.abs(si - ratio)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sizes[sizes > 1], bins=30, color="#2563eb", edgecolor="white")
    axes[0].set_xlabel("Размер группы (дубликаты признаков)")
    axes[0].set_ylabel("Число групп")
    axes[0].set_title("Распределение размеров групп")

    spreads_ic50 = []
    for gid in np.where(sizes > 1)[0]:
        vals = ic[groups == gid]
        if len(vals) > 1:
            spreads_ic50.append(float(np.max(vals) - np.min(vals)))
    if spreads_ic50:
        axes[1].hist(spreads_ic50, bins=30, color="#dc2626", edgecolor="white")
    axes[1].set_xlabel("Размах IC50 внутри группы")
    axes[1].set_ylabel("Число групп")
    axes[1].set_title("Конфликт таргетов при одинаковых дескрипторах")

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "duplicate_groups_phase0.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    summary = {
        "n_rows": len(x_raw),
        "n_unique_feature_groups": n_groups,
        "n_rows_in_multi_groups": n_dup_rows,
        "pct_rows_in_multi_groups": round(100.0 * n_dup_rows / len(x_raw), 2),
        "n_multi_groups": n_multi_groups,
        "si_equals_cc50_over_ic50_max_abs_diff": si_ratio_max,
        "target_conflict_in_multi_groups": target_stats,
    }
    summary["figure"] = str(fig_path)
    return summary


def run_adversarial_validation(
    x_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    cfg: Config,
    *,
    out_json: Path,
) -> dict[str, Any]:
    pre = Preprocessor(cfg.missing_threshold)
    pre.fit(x_raw)
    x_tr = pre.transform(x_raw)
    x_te = pre.transform(x_test_raw)
    n_tr, n_te = x_tr.shape[0], x_te.shape[0]
    x_all = np.vstack([x_tr, x_te])
    y_is_test = np.array([0] * n_tr + [1] * n_te, dtype=np.int32)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=2000, C=0.1, random_state=cfg.random_seed),
            ),
        ]
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_seed)
    auc_scores = cross_val_score(clf, x_all, y_is_test, cv=skf, scoring="roc_auc")
    auc_mean = float(np.mean(auc_scores))

    clf.fit(x_all, y_is_test)
    lr = clf.named_steps["lr"]
    coef = np.abs(lr.coef_.ravel())
    feat_names = pre.get_feature_names()
    order = np.argsort(coef)[::-1][:20]
    top_features = [
        {"feature": feat_names[i], "abs_coef": float(coef[i])} for i in order if i < len(feat_names)
    ]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    rf.fit(x_all, y_is_test)
    imp = rf.feature_importances_
    imp_order = np.argsort(imp)[::-1][:20]
    top_importance = [
        {"feature": feat_names[i], "importance": float(imp[i])}
        for i in imp_order
        if i < len(feat_names)
    ]

    sample_weight = np.ones(n_tr, dtype=np.float64)
    if auc_mean > 0.55:
        proba_tr = clf.predict_proba(x_tr)[:, 0]
        sample_weight = np.clip(proba_tr, 0.05, 1.0)

    payload = {
        "auc_mean_5fold": auc_mean,
        "auc_std_5fold": float(np.std(auc_scores)),
        "interpretation": "AUC≈0.5 — слабый сдвиг; >0.6 — заметный covariate shift",
        "top_logistic_features": top_features,
        "top_rf_importance": top_importance,
        "train_sample_weights": {
            "min": float(sample_weight.min()),
            "max": float(sample_weight.max()),
            "mean": float(sample_weight.mean()),
        },
    }
    weights_path = out_json.parent / "adversarial_sample_weights.json"
    weights_path.write_text(
        json.dumps({"weights": sample_weight.tolist()}, indent=2),
        encoding="utf-8",
    )
    payload["sample_weights_file"] = str(weights_path)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_si_ablation_from_oof(
    y_df: pd.DataFrame,
    stack_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ablation SI на уже посчитанных OOF stacking-предсказаниях."""
    ic = stack_result["oof_pred_by_target"]["IC50"]
    cc = stack_result["oof_pred_by_target"]["CC50"]
    si_model = stack_result["oof_pred_by_target"]["SI"]
    y_mat = np.column_stack([y_df[t].to_numpy(dtype=np.float64) for t in TARGETS])
    ratio = np.clip(cc, EPS, None) / np.clip(ic, EPS, None)

    blend_w, _ = blend_si_weight(si_model, ic, cc, y_mat)

    rows: list[dict[str, Any]] = []
    variants = [
        ("si_model", si_model, "Stacking SI (как submission2_a4)"),
        ("si_ratio_only", ratio, "SI = CC50/IC50 из OOF IC50/CC50"),
        ("si_blend", blend_w * si_model + (1.0 - blend_w) * ratio, f"Blend w={blend_w:.3f}"),
    ]
    for key, si_vec, label in variants:
        pred = np.column_stack([ic, cc, si_vec])
        score, parts = competition_score(y_mat, pred)
        rows.append(
            {
                "variant": key,
                "label": label,
                "oof_competition_score": float(score),
                "rmse_IC50": float(parts["IC50"]),
                "rmse_CC50": float(parts["CC50"]),
                "rmse_SI": float(parts["SI"]),
                "si_blend_w": float(blend_w) if key == "si_blend" else None,
            }
        )
    return rows


def run_si_ablation(
    x_raw: pd.DataFrame,
    y_df: pd.DataFrame,
    x_full: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    stack_result = run_oof_stacking_cv(x_raw, y_df, x_full, cfg, si_blend=False)
    return run_si_ablation_from_oof(y_df, stack_result)


def write_duplicate_report_md(summary: dict[str, Any], path: Path) -> None:
    ts = summary["target_conflict_in_multi_groups"]
    lines = [
        "# Отчёт: группы дубликатов признаков (Фаза 0)",
        "",
        "**Подготовка:** Артур Сидоров",
        "**Контекст:** ChemAI Predict the Cure, train.csv",
        "",
        "## Сводка",
        "",
        "| Показатель | Значение |",
        "|------------|----------|",
        f"| Строк train | {summary['n_rows']} |",
        f"| Уникальных групп признаков | {summary['n_unique_feature_groups']} |",
        f"| Строк в группах size>1 | {summary['n_rows_in_multi_groups']} ({summary['pct_rows_in_multi_groups']}%) |",
        f"| Групп с >1 строкой | {summary['n_multi_groups']} |",
        f"| max \\|SI − CC50/IC50\\| на train | {summary['si_equals_cc50_over_ic50_max_abs_diff']:.3e} |",
        "",
        "## Конфликт таргетов внутри групп",
        "",
        "| Таргет | Групп с разбросом >1e-6 | Медиана отн. разброса | Max отн. разброс |",
        "|--------|-------------------------|----------------------|------------------|",
    ]
    for t in TARGETS:
        s = ts[t]
        lines.append(
            f"| {t} | {s['groups_with_spread_gt_1e-6']} | "
            f"{s['median_relative_spread_in_multi_groups']:.4f} | "
            f"{s['max_relative_spread']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- Одинаковый вектор RDKit-дескрипторов **не гарантирует** одинаковые IC50/CC50/SI — "
            "это label noise / скрытые факторы (сольват, протокол, стереоизомер не в таблице).",
            "- GroupKFold «в лоб» может **ухудшить LB** при другом смешении test — см. историю submission4.",
            "- Для Фазы 1: веса `1/√(размер группы)` и robust loss предпочтительнее жёсткого group split.",
            "",
            f"![duplicate groups]({Path(summary['figure']).name})",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_log_md(
    *,
    dup_summary: dict[str, Any],
    adv: dict[str, Any],
    stacking_oof: dict[str, Any],
    si_rows: list[dict[str, Any]],
    lb_reference: dict[str, float],
    path: Path,
) -> None:
    lines = [
        "# Журнал экспериментов — Фаза 0",
        "",
        "**Подготовка:** Артур Сидоров, Максим Власюк (stacking / ablation)",
        "",
        "## 0.2 Adversarial validation (train vs test)",
        "",
        f"- AUC (5-fold CV): **{adv['auc_mean_5fold']:.4f}** ± {adv['auc_std_5fold']:.4f}",
        f"- Веса train: `{adv['sample_weights_file']}`",
        "",
        "Топ признаков (|coef| logistic):",
        "",
    ]
    for item in adv["top_logistic_features"][:8]:
        lines.append(f"- `{item['feature']}`: {item['abs_coef']:.4f}")

    lines.extend(
        [
            "",
            "## 0.3 OOF stacking (baseline v2, `CHEM_USE_STACKING=true`)",
            "",
            f"- **OOF competition_score:** {stacking_oof['oof_competition_score']:.4f}",
            f"- RMSE IC50: {stacking_oof['oof_parts']['IC50']:.4f}",
            f"- RMSE CC50: {stacking_oof['oof_parts']['CC50']:.4f}",
            f"- RMSE SI: {stacking_oof['oof_parts']['SI']:.4f}",
            "",
            "## 0.4 OOF vs public LB (исторические отправки)",
            "",
            "| Вариант | OOF (этот прогон) | Public LB (зафикс.) | Δ OOF−LB |",
            "|---------|-------------------|---------------------|----------|",
            f"| Stacking cluster (≈submission2) | {stacking_oof['oof_competition_score']:.2f} | "
            f"{lb_reference.get('submission2_a4', float('nan')):.2f} | "
            f"{stacking_oof['oof_competition_score'] - lb_reference.get('submission2_a4', 0):+.2f} |",
            f"| Group split (≈submission4) | — | "
            f"{lb_reference.get('submission4_a4', float('nan')):.2f} | — |",
            "",
            "## 0.5 Ablation SI (на OOF stacking IC50/CC50)",
            "",
            "| Вариант | OOF mean RMSE | RMSE SI |",
            "|---------|---------------|---------|",
        ]
    )
    for row in si_rows:
        lines.append(
            f"| {row['label']} | {row['oof_competition_score']:.4f} | {row['rmse_SI']:.4f} |"
        )

    gate = stacking_oof["oof_competition_score"]
    gate_ok = gate <= 320
    lines.extend(
        [
            "",
            "## Gate Фазы 0",
            "",
            f"- OOF stacking v2 = **{gate:.2f}** → "
            f"{'**PASS** (≤320), переход к Фазе 1' if gate_ok else '**FAIL** (>320), упор на Фазы 2–3'}",
            "",
            f"- Дубликаты: {dup_summary['pct_rows_in_multi_groups']}% строк в multi-groups",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase0_diagnostics(
    *, docs_dir: Path | None = None, models_dir: Path | None = None
) -> dict[str, Any]:
    cfg = get_config()
    docs = docs_dir or Path("docs/eda")
    models = models_dir or cfg.models_dir

    df = load_train()
    test_df = load_test()
    x_raw, y_df = split_features_targets(df)
    x_raw = add_chem_features(x_raw)
    x_test = add_chem_features(test_df.drop(columns=["index"], errors="ignore"))

    dup_summary = analyze_duplicate_groups(x_raw, y_df, figures_dir=docs / "figures")
    write_duplicate_report_md(dup_summary, docs / "duplicate_groups_report.md")

    adv = run_adversarial_validation(
        x_raw,
        x_test,
        cfg,
        out_json=models / "adversarial_validation.json",
    )

    full_pre = Preprocessor(cfg.missing_threshold)
    full_pre.fit(x_raw)
    x_full = full_pre.transform(x_raw)

    stacking_oof = run_oof_stacking_cv(x_raw, y_df, x_full, cfg, si_blend=False, fit_final=False)
    si_rows = run_si_ablation_from_oof(y_df, stacking_oof)

    lb_ref = {
        "submission2_a4": 349.30995,
        "submission4_a4": 373.03730,
        "baseline_repo_approx": 365.0,
    }
    write_experiment_log_md(
        dup_summary=dup_summary,
        adv=adv,
        stacking_oof=stacking_oof,
        si_rows=si_rows,
        lb_reference=lb_ref,
        path=docs / "phase0_experiment_log.md",
    )

    phase0_metrics = {
        "duplicate_groups": dup_summary,
        "adversarial_validation": adv,
        "stacking_oof": {
            "oof_competition_score": stacking_oof["oof_competition_score"],
            "oof_parts": stacking_oof["oof_parts"],
            "cv_mean_rmse": stacking_oof["cv_mean_rmse"],
        },
        "si_ablation": si_rows,
        "lb_reference": lb_ref,
    }
    metrics_path = models / "phase0_diagnostics.json"
    metrics_path.write_text(
        json.dumps(phase0_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Фаза 0: артефакты записаны в %s", docs)
    return phase0_metrics
