# План рефакторинга: ChemAI → единый ipynb (полная версия)

**Дата:** 2026-07-10 (обновлено)  
**Принцип:** **не упрощать** — собрать в один ноутбук все наработки репозитория и **воспроизвести submission2_a4 (LB 349.31)**.  
**Итоговый файл:** `notebooks/avo_chemai_predict_the_cure.ipynb`  
**Роли:** [`Роли_рефакторинг_сессия_2026.md`](Роли_рефакторинг_сессия_2026.md)

---

## 1. Цель (уточнение от команды)

| Было (ошибочно) | Стало (правильно) |
|-----------------|-------------------|
| Упростить до 5 моделей sklearn | Все 9+ кандидатов + stacking 2-го уровня |
| Ноутбук ≈ baseline, LB 349 «упомянуть» | **Run All → submission2_a4.csv** (тот же скрипт, что дал 349.31) |
| Не импортировать chemai | `pip install -e .` + вызов **существующего** кода |

---

## 2. Канонический путь к LB 349.31

**Файл:** `docs/submissions/submission2_a4.csv`  
**Генератор:** `tools/author4_high_priority_submissions.py --only 2`

Архитектура (из `docs/eda/Итоговая_аналитическая_записка`):

1. `add_chem_features` — 7 доменных признаков
2. `Preprocessor` — impute, scale, drop high-missing / zero-var
3. **ClusterKFold** (LeakFreeClusterKFold)
4. Per-fold OOF от **9 базовых моделей** (`build_default_candidates`)
5. Meta: **StandardScaler + RidgeCV** на OOF-матрице
6. Финальный fit: `fit_all_final` + `Expm1Predictor` (log1p IC50/CC50)
7. `postprocess` → CSV

> **Примечание:** `run_pipeline.py` с `CHEM_USE_STACKING=true` — попытка интеграции в `train.py`; для submission2 используется **author4** (исторический скрипт, восстановлен из git `e6ff309`).

---

## 3. Структура ноутбука (~28 ячеек)

| § | Содержание | Источник в репо |
|---|------------|-----------------|
| deps | pip + `pip install -e .` | `pyproject.toml` |
| §2 | Загрузка train/test | `data_loader.py` |
| §3 | EDA таргетов, SI=CC50/IC50 | `docs/eda/EDA_Report.md` |
| §4 | Domain features | `build_features.py` |
| §5 | PCA (матчасть) | `Матчасть_PCA_LDA_ICA.md` |
| §6 | ICA (матчасть) | то же |
| §7 | Список 9+ моделей | `candidate_models.py` |
| §7 | Baseline ensemble | `train.py` (`CHEM_USE_STACKING=false`) |
| §7 | **submission2 stacking** | `tools/author4 --only 2` |
| §8 | compare_df (baseline vs stacking vs submission3/4) | `Ablation_экспериментов_оригинальность.md` |
| §9 | `notebooks/submission.csv` ← submission2_a4 | assert формата |
| §9 | Скрин LB 349.31 | `docs/eda/figures/leaderboard_team39_best_score.png` |

---

## 4. Сборка и пересборка

```powershell
python scripts/build_avo_chemai.py
# правка текстов выводов:
# scripts/notebook_conclusions.py → python scripts/build_avo_chemai.py
```

---

## 5. Воспроизводимость

| Проверка | Ожидание |
|----------|----------|
| `author4 --only 2` | `docs/submissions/submission2_a4.csv`, OOF ~557 |
| Формат CSV | 250 × 4, `index,IC50,CC50,SI` |
| Public LB (истор.) | **349.30995** (Kaggle закрыт — сверка по скрину) |
| Baseline для сравнения | OOF ~607, LB ~363 |

---

## 6. Что **не** делать

- Не выдумывать новые модели или упрощённые пайплайны
- Не скрывать stacking за «студенческим baseline»
- Не менять логику author4 / train.py при сборке ipynb

---

## 7. Статус

- [x] Восстановлен `tools/author4_high_priority_submissions.py`
- [x] `scripts/build_avo_chemai.py` + `notebook_conclusions.py`
- [x] `notebooks/avo_chemai_predict_the_cure.ipynb`
- [x] `notebooks/submission.csv` (копия submission2_a4)
- [ ] Run All на чистой машине (Colab / платформа) — проверка команды

---

*Полная версия: все наработки в одном ipynb, результат = submission2_a4 (LB 349.31).*
