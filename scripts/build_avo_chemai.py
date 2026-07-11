#!/usr/bin/env python
"""Generate standalone notebooks/39_chemai.ipynb (no repo dependency)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "39_chemai.ipynb"

_conc: dict = {}
exec((ROOT / "scripts" / "notebook_conclusions.py").read_text(encoding="utf-8"), _conc)

_s7: dict = {}
exec((ROOT / "scripts" / "notebook_section7.py").read_text(encoding="utf-8"), _s7)

CHEM_FEATURES = _s7["CHEM_FEATURES"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


TITLE = """\
# Хакатон ChemAI: Predict the Cure

**НИЯУ МИФИ, группа М25-555**

**Состав команды:**

| Участник | Роль в проекте |
|----------|----------------|
| Анастасия Волконская | тимлид, интегратор ноутбука |
| Мария Макарова | EDA и данные (§2–§3) |
| Артур Сидоров | матчасть PCA/ICA (§5–§6) |
| Максим Власюк | модели и CV, stacking (§7–§8) |
| Алина Давыденко | тексты выводов, вычитка |

**Kaggle:** Задача хакатона. ChemAI: Predict the Cure, команда 39

https://www.kaggle.com/competitions/chem-ai-predict-the-cure/overview

Единый самодостаточный ноутбук → submission **349.31** (OOF stacking, ClusterKFold).
"""

CELLS = [
    md(TITLE),
    code(
        """# @title Установка зависимостей
\"\"\"
Проверка и установка pip-пакетов.
Репозиторий Git не нужен — ноутбук полностью автономный.
\"\"\"
import importlib
import subprocess
import sys

# --- список зависимостей для EDA, ML и stacking ---
REQUIRED = (
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "lightgbm", "xgboost",
)
for pkg in REQUIRED:
    mod = "sklearn" if pkg == "scikit-learn" else pkg
    try:
        importlib.import_module(mod)
    except ImportError:
        # Colab / чистое окружение: тихая установка недостающего пакета
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("Зависимости OK")
"""
    ),
    code(
        """# @title §1. Импорты и константы
\"\"\"
Общие библиотеки, seed и настройки графиков для всего ноутбука.
\"\"\"
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- воспроизводимость экспериментов ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- таргеты соревнования (метрика = mean RMSE по ним) ---
TARGETS = ("IC50", "CC50", "SI")

# --- единый стиль графиков EDA ---
plt.rcParams.update({"figure.figsize": (8, 4), "font.size": 10})
sns.set_theme(style="whitegrid")
print("RANDOM_STATE =", RANDOM_STATE)
"""
    ),
    md(
        "⚠️ **Данные:** нужны `train.csv` и `test.csv` с Kaggle.\n\n"
        "- Положите файлы рядом с ноутбуком, в папку `data/` или загрузите через Colab (ячейка §2).\n"
        "- Репозиторий Git **не требуется**."
    ),
    code(
        """# @title §2. Загрузка данных
\"\"\"
Ищем train.csv / test.csv по нескольким путям.
В Google Colab — fallback через files.upload().
\"\"\"
TRAIN_NAME, TEST_NAME = "train.csv", "test.csv"


def find_data_file(name: str) -> Path | None:
    \"\"\"Ищет CSV в cwd, data/, /content/ — типичные пути Colab и локального запуска.\"\"\"
    for p in (
        Path(name),
        Path("data") / name,
        Path("/content") / name,
        Path("/content/data") / name,
    ):
        if p.is_file():
            return p.resolve()
    return None


# --- поиск файлов на диске ---
train_path = find_data_file(TRAIN_NAME)
test_path = find_data_file(TEST_NAME)

# --- fallback: ручная загрузка в Google Colab ---
if train_path is None or test_path is None:
    try:
        from google.colab import files  # type: ignore
        print("Загрузите train.csv и test.csv с Kaggle:")
        uploaded = files.upload()
        for fname, data in uploaded.items():
            Path(fname).write_bytes(data)
        train_path = find_data_file(TRAIN_NAME)
        test_path = find_data_file(TEST_NAME)
    except ImportError:
        pass

if train_path is None or test_path is None:
    raise FileNotFoundError("Не найдены train.csv / test.csv — скачайте с Kaggle.")

# --- чтение таблиц ---
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- список признаков (всё кроме index и таргетов) ---
feature_cols = [c for c in train_df.columns if c not in ("index", *TARGETS)]
print("train:", train_df.shape, "| test:", test_df.shape)
print("числовых признаков:", len(feature_cols))
train_df.head(3)
"""
    ),
    md(_conc["AFTER_LOAD"]),
    code(
        """# @title §3. EDA — распределения таргетов
\"\"\"
Гистограммы IC50, CC50, SI (log-шкала по оси Y для наглядности хвостов).
Проверка тождества SI = CC50/IC50 на train.
\"\"\"
# --- гистограммы таргетов ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, t in zip(axes, TARGETS, strict=True):
    ax.hist(train_df[t], bins=40, color="steelblue", edgecolor="white")
    ax.set_title(t)
    ax.set_yscale("log")  # частоты на log-шкале — видны редкие большие значения
plt.tight_layout()
plt.show()

# --- проверка определения SI на train ---
ratio = train_df["CC50"] / train_df["IC50"].clip(lower=1e-8)
max_err = (train_df["SI"] - ratio).abs().max()
print("max |SI - CC50/IC50|:", max_err)
"""
    ),
    code(
        """# @title §3. EDA — describe и skew
\"\"\"
Сводная статистика таргетов; skew показывает асимметрию (обоснование log1p в §7).
\"\"\"
desc = train_df[list(TARGETS)].describe().T
desc["skew"] = train_df[list(TARGETS)].skew()  # skew >> 0 → правый хвост
display(desc.round(3))
"""
    ),
    md(_conc["AFTER_EDA"]),
    code(CHEM_FEATURES),
    code(
        """# @title §4. Domain features
\"\"\"
Добавляем 7 QSPR-признаков (функция add_chem_features из §4.0).
\"\"\"
# --- признаки train без таргетов и index ---
x_raw = add_chem_features(
    train_df.drop(columns=list(TARGETS)).drop(columns=["index"], errors="ignore")
)
new_cols = [c for c in x_raw.columns if c not in feature_cols]
print("добавлено признаков:", len(new_cols))
print("список:", new_cols)
x_raw[new_cols].describe().round(4)
"""
    ),
    md(_conc["AFTER_FEATURES"]),
    code(
        """# @title §4. Heatmap корреляций
\"\"\"
Корреляции таргетов с domain-признаками и CC50/IC50.
\"\"\"
# --- таблица для корреляционной матрицы ---
corr_df = train_df[list(TARGETS)].copy()
for c in new_cols:
    corr_df[c] = x_raw[c].values
corr_df["CC50/IC50"] = train_df["CC50"] / train_df["IC50"].clip(lower=1e-8)

# --- heatmap: насколько SI связан с отношением CC50/IC50 ---
cm = corr_df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
plt.title("Корреляции: таргеты и ключевые признаки")
plt.tight_layout()
plt.show()
print("corr(SI, CC50/IC50):", round(float(cm.loc["SI", "CC50/IC50"]), 4))
"""
    ),
    md(_conc["AFTER_CORR"]),
    code(
        """# @title §5. PCA — scree plot
\"\"\"
PCA на масштабированных числовых признаках: scree plot первых 20 компонент.
\"\"\"
# --- масштабирование числовых дескрипторов ---
num = train_df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num)

# --- PCA и scree plot ---
pca = PCA(random_state=RANDOM_STATE).fit(X_scaled)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(1, 21), evr[:20], alpha=0.7, label="компонента")
ax.plot(range(1, 21), cum[:20], "o-", color="crimson", label="накопленная")
ax.set_xlabel("PC")
ax.legend()
plt.title("Scree plot (первые 20 PC)")
plt.tight_layout()
plt.show()
print("PC1+PC2 накопленная доля дисперсии:", round(float(cum[1]), 4))
"""
    ),
    code(
        """# @title §5. PCA — scatter PC1 vs PC2
\"\"\"
Визуализация объектов в плоскости PC1–PC2; цвет = SI.
\"\"\"
# --- проекция на первые две главные компоненты ---
xy = pca.transform(X_scaled)[:, :2]
plt.figure(figsize=(6, 4))
sc = plt.scatter(xy[:, 0], xy[:, 1], c=train_df["SI"], cmap="viridis", s=12, alpha=0.7)
plt.colorbar(sc, label="SI")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: PC1 vs PC2")
plt.tight_layout()
plt.show()
"""
    ),
    md(_conc["AFTER_PCA"]),
    code(
        """# @title §6. FastICA
\"\"\"
FastICA — матчасть: независимые компоненты vs PCA (ортогональные).
\"\"\"
# --- ICA на тех же масштабированных признаках, что и PCA ---
ica = FastICA(n_components=3, random_state=RANDOM_STATE, max_iter=500)
ic = ica.fit_transform(X_scaled)

# --- scatter IC1 vs IC2, цвет = SI ---
plt.figure(figsize=(5, 4))
plt.scatter(ic[:, 0], ic[:, 1], c=train_df["SI"], cmap="plasma", s=10, alpha=0.7)
plt.xlabel("IC1")
plt.ylabel("IC2")
plt.title("ICA: IC1 vs IC2 (цвет = SI)")
plt.tight_layout()
plt.show()
"""
    ),
    md(_conc["AFTER_ICA"]),
    *_s7["build_section7_cells"](md, code, _conc),
    code(
        """# @title §8. Сравнение экспериментов
\"\"\"
Сводная таблица: baseline vs stacking vs исторические варианты команды.
\"\"\"
# --- сравнение зафиксированных экспериментов команды 39 ---
compare_df = pd.DataFrame([
    {
        "variant": "baseline weighted ensemble",
        "oof_competition_score": 607.0,
        "public_LB": 363.07,
        "note": "9 моделей, inverse-RMSE веса",
    },
    {
        "variant": "stacking (этот ноутбук)",
        "oof_competition_score": round(stack_oof, 2),
        "public_LB": 349.31,
        "note": "ClusterKFold + Ridge meta",
    },
    {
        "variant": "submission3 (+ SI blend)",
        "oof_competition_score": round(stack_oof, 2),
        "public_LB": 349.31,
        "note": "w≈1 на OOF — смесь не помогла",
    },
    {
        "variant": "submission4 (GroupKFold)",
        "oof_competition_score": None,
        "public_LB": 373.04,
        "note": "лучше OOF, хуже LB",
    },
])
display(compare_df)
"""
    ),
    md(_conc["AFTER_COMPARE"]),
    code(
        """# @title §9. Сохранение submission
\"\"\"
CSV для сдачи: index, IC50, CC50, SI (250 строк).
Эталон = author4 submission2 (ClusterKFold + 9 моделей + Ridge meta).
\"\"\"
import json

OUT_PATH = Path("39_chemai_submission.csv")
INTEGRATED_PATH = Path("submission_avo_chemai_integrated.csv")
FINGERPRINT_PATH = Path("submission2_reference_fingerprint.json")

# --- эталон до перезаписи (файл из репозитория / предыдущий прогон) ---
reference_df = pd.read_csv(INTEGRATED_PATH) if INTEGRATED_PATH.is_file() else None

submission_df.to_csv(OUT_PATH, index=False)
submission_df.to_csv(INTEGRATED_PATH, index=False)

# --- проверки формата Kaggle ---
assert submission_df.shape == (250, 4), f"ожидалось (250,4), получено {submission_df.shape}"
assert list(submission_df.columns) == ["index", "IC50", "CC50", "SI"]
assert submission_df["IC50"].min() >= 0
assert submission_df["CC50"].min() >= 0
assert submission_df["SI"].min() >= 0

# --- OOF stacking из §7.10 (author4 submission2 ≈ 557.22) ---
assert abs(stack_oof - 557.22) < 0.5, (
    f"OOF stacking {stack_oof:.2f} ≠ эталон ~557.22 — перезапустите §7 (Colab: Restart runtime → Run All)"
)

# --- численная сверка с эталоном submission2 ---
if reference_df is not None:
    max_diff = float(
        (submission_df[["IC50", "CC50", "SI"]] - reference_df[["IC50", "CC50", "SI"]])
        .abs()
        .max()
        .max()
    )
    assert max_diff < 1e-5, f"Расхождение с {INTEGRATED_PATH.name}: max |Δ|={max_diff:.2e}"
    print(f"Совпадает с эталоном {INTEGRATED_PATH.name} (max |Δ|={max_diff:.2e})")
elif FINGERPRINT_PATH.is_file():
    fp = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    chk = float(submission_df[["IC50", "CC50", "SI"]].to_numpy().sum())
    assert abs(chk - fp["sample_checksum"]) < 1e-3, "sample_checksum не совпал с эталоном submission2"
    for col, stats in fp["describe"].items():
        for key, expected in stats.items():
            got = float(submission_df[col].describe()[key])
            assert abs(got - expected) < 1e-2, f"{col}.{key}: {got} vs эталон {expected}"
    print("Статистики совпадают с", FINGERPRINT_PATH.name)
else:
    print("Эталон не найден — только проверки формата и OOF. Положите fingerprint рядом с ноутбуком.")

print("Файлы сохранены:")
print(" ", OUT_PATH.resolve())
print(" ", INTEGRATED_PATH.resolve())
print(submission_df.describe().round(2))
"""
    ),
    md(_conc["AFTER_VERIFY"]),
    md(_conc["AFTER_SUBMISSION"]),
    md(_conc["LEADERBOARD_MD"]),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": CELLS,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written {OUT} ({len(CELLS)} cells)")
