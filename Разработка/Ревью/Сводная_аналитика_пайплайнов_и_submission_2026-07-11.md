# Сводная аналитика: пайплайны, submission и скоринг

**Проект:** ChemAI Predict the Cure  
**Команда:** 39, группа М25-555, НИЯУ МИФИ  
**Дата фиксации:** 2026-07-11  
**Источники:** Kaggle (вкладка «Заявки»), `docs/eda/`, `Разработка/Бэклог/`, `scripts/verify_notebook_submission.py`, прогон единого ноутбука

---

## 1. Итоговая таблица submission (по скору платформы)

Скор Kaggle = **mean(RMSE)** по столбцам IC50, CC50, SI (**меньше — лучше**).

### 1.1. Заявки на платформе (скрин Kaggle, 2026-07-11)

Статус всех строк: **Completed (after deadline)** — соревнование закрыто для новых сабмитов; скоры пересчитаны на полном тесте (private = public в таблице).

| Место | Файл submission | Private score | Public score | Пайплайн |
|------:|-----------------|--------------:|-------------:|----------|
| **1** | `submission_avo_chemai_integrated.csv` | **326.89208** | **326.89208** | OOF stacking 2-го уровня (ClusterKFold + 9 моделей + Ridge meta) — эквивалент `submission2_a4` |
| 2 | `39_chemai_submission.csv` | 343.36453 | 343.36453 | Тот же stacking из единого ноутбука `notebooks/39_chemai.ipynb` (§7.10) |
| 3 | `submission_baseline_epic10_step1.csv` | 345.07702 | 345.07702 | Baseline репо: clip OOF + tree-only ансамбль (Epic 10, без stacking) |
| 4 | `submission_baseline_epic11_step2.csv` | 354.63146 | 354.63146 | Epic 11: duplicate_group CV + CatBoost + SI blend |

**Лучший зафиксированный скор на платформе (полный тест):** **326.89** — `submission_avo_chemai_integrated.csv`.

> **Замечание:** `39_chemai_submission.csv` (343.36) хуже integrated на ~16.5 баллов. При полном Run All ноутбука §7.10 файл должен совпадать с `submission2_a4` / integrated (верификация allclose 1e-6). Расхождение на Kaggle может означать неполный прогон §7.1–§7.9 или загрузку CSV до завершения stacking.

### 1.2. Исторические public LB (период соревнования, ранг ~90)

| Файл submission | Public LB | OOF (offline) | Примечание |
|-----------------|----------:|--------------:|------------|
| **`submission2_a4.csv`** | **349.30995** | **557.22** | Лучший **public** за период хакатона |
| `submission3_a4.csv` | 349.30995 | ~557 | + SI blend; w≈1 → идентично №2 |
| `submission4_a4.csv` | 373.03730 | лучше cluster-OOF | GroupKFold по дубликатам — LB хуже |
| `submission5_a4.csv` | 373.03730 | — | Как №4 + Optuna LGBM — без выигрыша на LB |
| Baseline repo (`final_submission.csv`) | **363.07** | ~607 / ~645 | Weighted ensemble, 9 моделей, без stacking |
| Epic 10 (офлайн, откатан) | **345.08** | 591 | clip + tree-only, без meta-stacking |
| Epic 11 (офлайн, откатан) | 354.63 | 531 | duplicate_group + CatBoost — gate не пройден |
| Stacking в `train.py` (ошибочный путь) | **1447** | ~878 | Meta in-sample на OOF — не для продакшена |

**Важно:** разрыв **public LB 349** (соревнование) vs **326.89** (после дедлайна) объясняется разным набором/режимом оценки на платформе, а не другим алгоритмом: `submission_avo_chemai_integrated.csv` **численно совпадает** с `submission2_a4.csv` (max |Δ| < 10⁻¹⁰, `scripts/verify_notebook_submission.py`).

---

## 2. Каталог пайплайнов

### 2.1. Baseline (репозиторий `train.py` / `predict.py`)

| Компонент | Описание |
|-----------|----------|
| CV | **ClusterKFold** (KMeans на MolLogP, TPSA, RingCount, HeavyAtomCount) |
| Preprocessor | Per-fold: impute медианой, drop >30% NA, StandardScaler |
| Модели | 9 семейств: LGBM, XGB, Ridge, ElasticNet, HistGBR, RF, ExtraTrees, sklearn GBR, BayesianRidge |
| Ансамбль | Inverse-RMSE веса по OOF, **без** meta-stacking |
| Таргеты | log1p(IC50), log1p(CC50), SI без log |
| Domain features | 7 QSPR-признаков (`add_chem_features`) |
| Артефакт | `final_submission.csv` → LB **~363** |

### 2.2. Stacking submission2 (author4 / ноутбук §7)

| Компонент | Описание |
|-----------|----------|
| Базовый слой | Те же 9 моделей; OOF на ClusterKFold |
| Meta | **StandardScaler + RidgeCV** на OOF-матрице (9 колонок) **отдельно** по IC50, CC50, SI |
| Инференс test | Финальный fit каждой базовой модели на полном train → meta → predict |
| Скрипт | `tools/author4_high_priority_submissions.py --only 2` |
| Ноутбук | `notebooks/39_chemai.ipynb` (§7.0–§7.10, 9 ячеек моделей + meta) |
| OOF | **557.22** (IC50 319.7, CC50 564.6, SI 787.4) |
| Файлы | `submission2_a4.csv`, `submission_avo_chemai_integrated.csv`, `39_chemai_submission.csv` |

### 2.3. Stacking submission3 (+ SI blend)

| Компонент | Описание |
|-----------|----------|
| Отличие | Post-hoc смесь SI с CC50/IC50; вес подбирается на OOF |
| Результат | w≈1 → **349.31** = submission2 (смесь не улучшила) |
| Файл | `submission3_a4.csv` |

### 2.4. Stacking submission4/5 (GroupKFold)

| Компонент | Описание |
|-----------|----------|
| CV | **GroupKFold** по дубликатам вектора признаков |
| Результат | OOF лучше, **LB 373.04** — overfitting CV / нетрансферность |
| Файлы | `submission4_a4.csv`, `submission5_a4.csv` |

### 2.5. Epic 10 — стабилизация baseline (откатан из git)

| Компонент | Описание |
|-----------|----------|
| Изменения | Clipping OOF/predict после expm1; **tree-only** (без линейных моделей) |
| OOF | 591 → LB **345.08** |
| Файл | `submission_baseline_epic10_step1.csv` |

### 2.6. Epic 11 — расширенный CV (откатан)

| Компонент | Описание |
|-----------|----------|
| Изменения | duplicate_group CV, CatBoost, SI blend |
| OOF | 531 → LB **354.63** (хуже Epic 10 и stacking) |
| Файл | `submission_baseline_epic11_step2.csv` |

---

## 3. OOF vs платформа (калибровка offline-метрик)

| ID | Пайплайн | OOF competition_score | Platform score | Δ (OOF − LB)* |
|----|----------|----------------------:|---------------:|--------------:|
| E0 | submission2 / integrated | 557.22 | 349.31 (public) / **326.89** (post-deadline) | +208 / +230 |
| E2 | baseline repo | ~607–645 | 363.07 | +244–282 |
| E3 | Epic 10 | 591 | 345.08 | +246 |
| E4 | Epic 11 | 531 | 354.63 | +176 |
| E1 | stacking в train.py | ~878 | 1447 | −569 |

\*Δ для public LB соревнования; OOF систематически **пессимистичнее** платформы на ~180–280 баллов при ClusterKFold + per-fold Preprocessor.

**Вывод:** ориентироваться на **абсолютный порядок** экспериментов на платформе, а не на абсолютное значение OOF.

---

## 4. Базовый слой: OOF отдельных моделей (§7 ноутбука)

Оценка на **ClusterKFold**, mean RMSE по IC50 + CC50 + SI (типичный порядок, точные значения — в выводе ячеек §7.1–§7.9 после Run All):

| Модель | Роль в ансамбле |
|--------|-----------------|
| LightGBM, XGBoost | Сильнейшие по IC50/CC50 после log1p |
| HistGBR, sklearn GBR | Дополнительные бустинги, decorrelation |
| RandomForest, ExtraTrees | Tree-bagging, устойчивость |
| Ridge, ElasticNet, BayesianRidge | Линейные baselines для meta-Ridge |

Meta-stacking (§7.10) снижает ошибку относительно любой одиночной модели; итоговый OOF stacking ≈ **557**.

---

## 5. Верификация воспроизводимости (2026-07-11)

| Проверка | Результат |
|----------|-----------|
| Два прогона `author4 --only 2` | max \|Δ\| между run1 и run2: **1.37×10⁻¹⁰** |
| vs эталон `submission2_a4.csv` | max \|Δ\|: **< 1.16×10⁻¹⁰** |
| `np.allclose(..., rtol=1e-6)` | **True** |
| MD5 CSV | Различается (формат float в CSV) — **не использовать** как критерий |
| RANDOM_STATE | 42 — детерминированные предсказания |

Артефакты: `notebooks/_verify_runs/`, `notebooks/_verify_runs/verification_report.json`.

---

## 6. EDA и данные (контекст для всех пайплайнов)

| Показатель | Значение |
|------------|----------|
| train / test | 751×214 / 250×211 |
| Числовых признаков | 210 + 7 domain |
| max \|SI − CC50/IC50\| | 2×10⁻¹¹ |
| skew IC50 / CC50 / SI | 3.79 / 2.06 / 15.63 |
| Дубликаты признаков | ~16% строк → обоснование ClusterKFold |
| Adversarial train/test AUC | 0.48 — слабый covariate shift |

---

## 7. Рекомендуемые артефакты для сдачи

| Назначение | Файл |
|------------|------|
| Единый ноутбук (Run All) | `notebooks/39_chemai.ipynb` |
| Submission stacking (лучший алгоритм) | `notebooks/submission_avo_chemai_integrated.csv` или `39_chemai_submission.csv` |
| Эталон author4 | `docs/submissions/submission2_a4.csv` |
| Сводная ablation | `docs/eda/Ablation_экспериментов_оригинальность.md` |
| Верификация ipynb | `docs/eda/Аналитическая_записка_единый_ipynb_2026-07-11.md` |

---

## 8. Выводы

1. **Лучший алгоритм команды** — OOF stacking 2-го уровня (ClusterKFold + 9 моделей + Ridge meta). Он же реализован в `39_chemai.ipynb`.
2. **Лучший скор на Kaggle (полный тест, 2026-07-11):** **326.89** — `submission_avo_chemai_integrated.csv` (эквивалент `submission2_a4`).
3. **Лучший public LB в период соревнования:** **349.31** (ранг ~90).
4. Упрощённый baseline (~363) и эксперименты Epic 10–11 (345–355) **не превосходят** stacking на платформе.
5. GroupKFold + stacking (**373**) и in-sample meta в `train.py` (**1447**) — **отклонены** как канон.
6. Offline OOF **не калиброван** к LB; для сравнения пайплайнов использовать таблицы §1 и платформенный скор.

---

## 9. Связанные документы

- [`docs/eda/Итоговая_аналитическая_записка_Predict_the_Cure.md`](../../docs/eda/Итоговая_аналитическая_записка_Predict_the_Cure.md)
- [`docs/eda/Ablation_экспериментов_оригинальность.md`](../../docs/eda/Ablation_экспериментов_оригинальность.md)
- [`Разработка/Бэклог/Наработки_2026-06-22_после_закрытия_Kaggle.md`](../Бэклог/Наработки_2026-06-22_после_закрытия_Kaggle.md)
- [`docs/eda/Аналитическая_записка_единый_ipynb_2026-07-11.md`](../../docs/eda/Аналитическая_записка_единый_ipynb_2026-07-11.md)

---

*Подготовлено для ревью сессионной сдачи. Kaggle: ChemAI Predict the Cure, команда 39.*
