#!/usr/bin/env python3
"""Полный EDA по train/test: таблицы, графики и Markdown-отчёт в docs/eda/.

Запуск из корня репозитория:
    uv run python tools/eda_generate_report.py
    uv run python tools/eda_generate_report.py --data-dir ml/data --out-dir docs/eda
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11

TARGETS = ("IC50", "CC50", "SI")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сгенерировать EDA отчёт ChemAI.")
    p.add_argument("--data-dir", type=Path, default=Path("ml/data"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/eda"))
    return p.parse_args()


def feature_columns(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in TARGETS and c != "index"]


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def train_test_shift(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Относительный сдвиг средних |mu_test - mu_train| / sigma_train для признаков."""
    tr = train_df[feats].astype(float)
    te = test_df[feats].astype(float)
    mu_t = tr.mean()
    sigma = tr.std(ddof=0).replace(0, np.nan)
    shift = (te.mean() - mu_t).abs() / sigma
    shift = shift.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    return shift


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feats = feature_columns(train)
    test_feats = [c for c in test.columns if c != "index"]
    feat_set_ok = set(feats) == set(test_feats)

    ratio_cc_ic = train["CC50"].astype(float) / train["IC50"].astype(float)
    si_residual = (train["SI"].astype(float) - ratio_cc_ic).abs().max()

    # --- Targets ---
    targ_stats_rows = []
    for t in TARGETS:
        s = train[t].astype(float)
        targ_stats_rows.append(
            {
                "target": t,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "min": float(s.min()),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "max": float(s.max()),
                "std": float(s.std(ddof=0)),
                "skew": float(s.skew()),
            }
        )
    targ_df = pd.DataFrame(targ_stats_rows)

    ratio_label = "CC50/IC50"
    corr_si_ratio = train[[*TARGETS]].assign(**{ratio_label: ratio_cc_ic}).corr(method="pearson")

    # --- Features: missing ---
    feat_missing = train[feats].isna().mean().sort_values(ascending=False)
    missing_any = feat_missing[feat_missing > 0]

    # Variance
    v = train[feats].astype(float).var(ddof=0)
    zero_var = (v <= 0) | (~np.isfinite(v))
    nv = int(zero_var.sum())

    # Constant / quasi-constant (uniq <= 3 for float — грубо «почти бинарные/константы»)
    uniq = train[feats].nunique(dropna=True)
    quasi = uniq[uniq <= 2].sort_values()

    # fr_* summaries
    fr_cols = [c for c in feats if c.startswith("fr_")]
    fr_means = (
        train[fr_cols].mean(numeric_only=True).sort_values(ascending=False)
        if fr_cols
        else pd.Series(dtype=float)
    )

    # Duplicates по признакам
    dup_feats = train[feats].duplicated().sum()
    dup_pct = 100 * float(dup_feats) / len(train)

    # Correlations таргеты vs признаки (Pearson), на полных pairwise drops NA — используем fillna медиана для устойчивости топа
    tr_num = train[feats].astype(float)
    medians = tr_num.median(numeric_only=True)
    filled = tr_num.fillna(medians)
    top_corr_chunks = []
    for tgt in TARGETS:
        corr = (
            filled.corrwith(train[tgt].astype(float), method="pearson")
            .abs()
            .sort_values(ascending=False)
        )
        top_corr_chunks.append(
            corr.head(20)
            .rename("corr_abs")
            .rename_axis("feature")
            .reset_index()
            .assign(target=tgt)[["target", "feature", "corr_abs"]]
        )
    top_corr = pd.concat(top_corr_chunks, ignore_index=True)

    # Shift train vs test
    shift_scores = train_test_shift(train, test, feats)

    # --- Figures ---
    def save_targets_hist() -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, tgt in zip(axes, TARGETS):
            train[tgt].astype(float).hist(bins=40, ax=ax, edgecolor="white", alpha=0.85)
            ax.set_title(tgt)
            ax.set_ylabel("count")
        plt.suptitle("Распределения таргетов (сырые)", y=1.05)
        p = fig_dir / "targets_raw_hist.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    def save_targets_log_hist() -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
        for ax, tgt in zip(axes, ("IC50", "CC50")):
            s = np.log1p(train[tgt].astype(float).clip(lower=0))
            s.hist(bins=40, ax=ax, edgecolor="white", alpha=0.85)
            ax.set_title(f"log1p({tgt})")
            ax.set_ylabel("count")
        plt.suptitle("Лог-сжатие активности IC50/CC50", y=1.05)
        p = fig_dir / "targets_log1p_hist.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    def save_si_vs_ratio() -> Path:
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ratio_q = ratio_cc_ic.clip(upper=ratio_cc_ic.quantile(0.99))
        ax.scatter(np.log1p(ratio_q), train["SI"].astype(float), alpha=0.35, s=12)
        ax.set_xlabel("log1p(CC50 / IC50), клип верхних 1% для читаемости")
        ax.set_ylabel("SI")
        ax.set_title("SI vs отношение CC50/IC50")
        p = fig_dir / "si_vs_ratio.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    def save_corr_heatmap_subset() -> Path:
        # топ-15 признаков по max(|corr|) с любым таргетом
        score = pd.Series(0.0, index=feats)
        for tgt in TARGETS:
            score = score.add(filled.corrwith(train[tgt].astype(float)).abs(), fill_value=0)
        cand = score.sort_values(ascending=False).index.tolist()
        top15 = []
        for col in cand:
            if filled[col].std(ddof=0) < 1e-12:
                continue
            top15.append(col)
            if len(top15) >= 15:
                break
        if not top15:
            raise RuntimeError("Heatmap subset: ни одной колонки с ненулевой дисперсией.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sub = pd.concat([train[[*TARGETS]].astype(float), filled[top15]], axis=1).corr()
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns, rotation=85, fontsize=8)
        ax.set_yticks(range(len(sub.columns)))
        ax.set_yticklabels(sub.columns, fontsize=8)
        ax.set_title("Корреляции: таргеты + топ-15 признаков")
        p = fig_dir / "heatmap_targets_top_feats.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    def save_missing_bar() -> Path:
        if missing_any.empty:
            return Path("")
        tail = missing_any.head(35)
        fig, ax = plt.subplots(figsize=(8, max(5, len(tail) * 0.18)))
        ax.barh(tail.index[::-1], (tail.values[::-1] * 100))
        ax.set_xlabel("Доля пропусков, %")
        ax.set_title("Топ признаков с пропусками (train)")
        p = fig_dir / "missingness_top.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    def save_shift_top() -> Path:
        tt = shift_scores.head(35)
        fig, ax = plt.subplots(figsize=(8, max(5, len(tt) * 0.18)))
        ax.barh(tt.index[::-1], tt.values[::-1])
        ax.set_xlabel("|mean_test − mean_train| / std_train")
        ax.set_title("Топ признаков: сдвиг распределения train → test")
        p = fig_dir / "train_test_shift_top.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p.relative_to(out_dir)

    fp_raw = save_targets_hist()
    fp_log = save_targets_log_hist()
    fp_si = save_si_vs_ratio()
    fp_hm = save_corr_heatmap_subset()
    fp_mis = save_missing_bar()
    fp_sh = save_shift_top()

    # Markdown body
    lines: list[str] = []
    lines.append("# Exploratory Data Analysis — ChemAI: Predict the Cure\n")
    lines.append("Отчёт сгенерирован автоматически (`tools/eda_generate_report.py`).\n")
    lines.append("## 1. Объём и целостность данных\n")
    lines.append("| Набор | Строки | Колонки | Комментарий |\n|---|---|---|---|\n")
    lines.append(
        f"| train | {len(train)} | {len(train.columns)} | включает `index`, три таргета, {len(feats)} числовых признаков |\n"
    )
    lines.append(
        f"| test | {len(test)} | {len(test.columns)} | `index` + те же признаки что в train |\n"
    )
    fs = "**да**" if feat_set_ok else "**нет**"
    lines.append(
        f"| Совпадение имён признаков train/test | {fs} | — | симметричная разность пустая |\n"
    )
    lines.append(
        f"| Дубликаты строк по 210 признакам (train) | **{dup_feats}** | ~{dup_pct:.1f}% объектов |\n"
    )
    lines.append(
        f"| Константные признаки (variance≤0 или нечисловая дисперсия) | **{nv}** | — | |\n"
    )

    lines.append("\n## 2. Целевые переменные (IC50, CC50, SI)\n")
    lines.append("| target | n | missing | min | median | mean | max | std | skew |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in targ_df.iterrows():
        lines.append(
            f"| {r.target} | {r.n} | {r['missing']} | {r['min']:.6g} | {r['median']:.6g} | "
            f"{r['mean']:.6g} | {r['max']:.6g} | {r['std']:.6g} | {r['skew']:.3g} |\n"
        )

    lines.append("\nНаблюдения:\n")
    lines.append(
        "- **IC50/CC50**: **асимметрия вправо** (skew ≫ 2) — лог-преобразование `log1p` смягчает шкалу и совпадает с решением пайплайна.\n"
    )
    lines.append(
        "- **SI**: высокая асимметрия (skew велик); при этом при проверке **на train выполняется тождество** "
        f"`SI = CC50 / IC50` с max $|SI - CC50/IC50| \\approx {si_residual:.3e}$.\n"
    )

    lines.append(f"\n![Распределения таргетов](figures/{fp_raw.name})\n")
    lines.append(f"\n![log1p IC50/CC50](figures/{fp_log.name})\n")

    corr_cols = [*TARGETS, ratio_label]
    lines.append("\n### Матрица корреляций: таргеты и отношение CC50/IC50\n")
    lines.append("|  | " + " | ".join(str(c) for c in corr_cols) + " |\n")
    lines.append("|" + "|".join(["---"] * (1 + len(corr_cols))) + "|\n")
    part = corr_si_ratio.round(4)
    for i in corr_cols:
        vals = []
        for j in corr_cols:
            vals.append(f"{float(part.loc[i, j]):.4f}")
        lines.append("| " + md_escape(str(i)) + " | " + " | ".join(vals) + " |\n")

    lines.append(
        "- На **`test`** метки SI недоступны; в сабмите по регламенту платформы всё равно нужно выдать **три столбца**. "
        "Практический вариант: обучаться по факту на `SI`, либо **согласовывать** предсказания "
        "`IC50/CC50/SI`, например задавая SI как функцию двух первых после инференса — но это уже инженерный выбор, не условие EDA.\n"
    )
    lines.append(f"\n![SI vs ratio](figures/{fp_si.name})\n")

    lines.append("\n## 3. Пропуски и «плоские» признаки\n")
    lines.append(f"- Колонок с пропусками в train: **{len(missing_any)}**\n")
    if missing_any.empty:
        lines.append(
            "- Пропусков в числовых признаках не обнаружено (полезно для простых шкалёров).\n"
        )
    else:
        lines.append("\nТоп по доле NA:\n")
        lines.append("| feature | pct_missing |\n|---|---|\n")
        for name, pct in missing_any.head(20).items():
            lines.append(f"| {md_escape(name)} | {100 * pct:.4f}% |\n")
        if fp_mis.name:
            lines.append(f"\n![Пропуски](figures/{fp_mis.name})\n")

    lines.append(
        f"\nПризнаки с числом **уникальных значений ≤2** в train (примерно константы/бинари с редкой меткой): **{len(quasi)}**\n\n"
    )
    if len(quasi):
        lines.append("| feature | nunique |\n|---|---|\n")
        for name, u in quasi.head(25).items():
            lines.append(f"| {md_escape(name)} | {int(u)} |\n")

    lines.append("\n## 4. Функциональные группы `fr_*`\n")
    if len(fr_cols) == 0:
        lines.append("- Колонки `fr_*` не найдены.\n")
    else:
        lines.append(f"- Всего `fr_*` колонок: **{len(fr_cols)}**.\n")
        lines.append(
            "- В RDKit-дескрипторах столбцы `fr_*` — как правило **счётчики** наличий/повторений фрагментов "
            "(могут принимать целые значения > 1), поэтому «доля наличия» интерпретируется как среднее значение, а не только {0,1}.\n"
        )
        lines.append(
            "\nНаибольшая средняя «частота» фрагмента (может подсказывать редкий сигнал):\n"
        )
        lines.append("| fr_* feature | mean(train) |\n|---|---|\n")
        for name, mu in fr_means.head(15).items():
            lines.append(f"| {md_escape(name)} | {mu:.4f} |\n")

    lines.append("\n## 5. Линейные связи признаков с таргетами (Pearson |corr|)\n")
    lines.append(
        "Таблицы: топ-10 по каждому таргету (NaN в признаках заполнены **медианой выборки только для этого ранжирования**).\n"
    )

    def table_top(tname: str) -> None:
        sub = top_corr[top_corr["target"] == tname].head(10).copy()
        lines.append(f"\n### {tname}\n")
        lines.append("| rank | feature | |corr| |\n|:---:|:---|:---:|\n")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            lines.append(
                f"| {rank} | {md_escape(row['feature'])} | {float(row['corr_abs']):.4f} |\n"
            )

    for t in TARGETS:
        table_top(t)

    lines.append(f"\n![heatmap subset](figures/{fp_hm.name})\n")

    lines.append("\n## 6. Сдвиг распределений train ↔ test\n")
    lines.append(
        "Эвристика для каждого признака: отношение |среднее_test − среднее_train| к std_train по train. "
        "Большие значения намекают на возможный covariate shift между выборками.\n"
    )
    lines.append("\nТоп-15:\n")
    lines.append("| feature | shift_score |\n|:---|:---:|\n")
    for name, score in shift_scores.head(15).items():
        lines.append(f"| {md_escape(name)} | {score:.4f} |\n")
    lines.append(f"\n![train-test shift](figures/{fp_sh.name})\n")

    lines.append("\n## 7. Выводы и рекомендации для моделирования\n")
    bullets = [
        "**Предобработка:** есть редкие пропуски (заряды BCUT*) — нужны заполнение + масштабирование; "
        f"**{nv} констант** и множество «плоских» колонок (≤2 различных значений в train) желательно либо убрать, либо не усиливать в линейных моделях.",
        "**IC50/CC50:** тяжёлые правые хвосты → `log1p` по IC50/CC50 уместен.",
        f"**SI на train совпадает с CC50/IC50** (max остаток ≈ `{si_residual:.3e}`). Для финального файла платформе всё равно нужен столбец SI — стоит обсудить **согласованность** его с предсказанными IC50/CC50.",
        "Много дубликатов по вектору признаков (см. §1): при CV учитывать утечку «соседа» между фолдами по желанию; кластерный CV частично с этим уже помогает.",
        "**Train vs test:** смотреть топ сдвига средних (§6): при сильном shift ансамбли и деревья обычно устойчивее простых линейных методов без регуляризации.",
        "`fr_*` — целочисленные счётчики / индексы фрагментов → хорошо улавливаются **деревьями**, при линейных моделях полезно квантировать или использовать регуляризацию.",
    ]
    for b in bullets:
        lines.append(f"- {b}\n")

    lines.append(
        "\nЭкспертная аналитика по выбору моделей и путям улучшения: "
        "**[Аналитическая_записка_архитектура_моделей.md]"
        "(Аналитическая_записка_архитектура_моделей.md)**.\n"
    )

    md_path = out_dir / "EDA_Report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {md_path} and figures under {fig_dir}")


if __name__ == "__main__":
    main()
