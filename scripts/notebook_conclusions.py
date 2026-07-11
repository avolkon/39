# Тексты markdown-выводов для notebooks/avo_chemai_predict_the_cure.ipynb
# Редактировать здесь; пересборка: python scripts/build_avo_chemai.py

AFTER_LOAD = """\
## 📊 Выводы: загрузка данных (§2)

| Набор | Строк | Колонок |
|-------|-------|---------|
| train | 751 | 214 |
| test | 250 | 211 |

**Интерпретация:**

- В train есть три таргета (`IC50`, `CC50`, `SI`) и 210 числовых дескрипторов молекул.
- Имена признаков в train и test совпадают — можно обучать и предсказывать без переименования.
- Пропусков в таргетах нет; часть дескрипторов имеет NA (см. EDA).

**Вывод:** данные готовы к feature engineering и CV.

**✅ Следующий шаг:** EDA таргетов и проверка связи SI ≈ CC50/IC50.
"""

AFTER_EDA = """\
## 📊 Выводы: EDA (§3)

| target | skew (train) | комментарий |
|--------|--------------|-------------|
| IC50 | ~3.8 | правый хвост |
| CC50 | ~2.1 | правый хвост |
| SI | ~15.6 | сильная асимметрия |

**Интерпретация:**

- На train выполняется **SI = CC50 / IC50** (макс. расхождение ~2e-11).
- ~16% строк train — дубликаты по вектору признаков → для CV нужен **ClusterKFold**, а не случайный split.
- Для IC50 и CC50 в пайплайне используем **log1p** при обучении и **expm1** при предсказании.

**Вывод:** таргеты нелинейны и асимметричны — одной линейной модели мало.

**✅ Следующий шаг:** доменные признаки и PCA/ICA (матчасть).
"""

AFTER_FEATURES = """\
## 📊 Выводы: признаки (§4)

Добавлены **7 доменных признаков** (QSPR поверх дескрипторов RDKit):

- `LogP_TPSA`, `Arom_Heavy_ratio`, `Charge_sum`
- флаги `fr_imide`, `fr_sulfone`
- `Ring_LogP`, `FractionCSP3_LogP`

**Вывод:** feature engineering выполнен в ноутбуке (§4.0), без внешних модулей.

**✅ Следующий шаг:** heatmap корреляций.
"""

AFTER_CORR = """\
## 📊 Выводы: корреляции (§4)

**Интерпретация:**

- **SI** с **CC50/IC50** коррелирует сильнее, чем SI с IC50 или CC50 по отдельности — согласуется с SI = CC50/IC50.
- Доменные признаки дают умеренные связи с IC50/CC50; одного признака недостаточно.

**✅ Следующий шаг:** PCA (§5).
"""

AFTER_PCA = """\
## 📊 Выводы: PCA (§5)

- PC1–PC2 объясняют ~33% дисперсии; SI не линейно отделяется двумя компонентами.
- PCA — матчасть и EDA; финальный stacking использует полный набор после `Preprocessor`.

**✅ Следующий шаг:** ICA (§6).
"""

AFTER_ICA = """\
## 📊 Выводы: ICA (§6)

- FastICA на табличных QSPR-данных менее интерпретируем, чем PCA.
- В финальном пайплайне (stacking) ICA не использовался.

**✅ Следующий шаг:** обучение моделей (§7).
"""

AFTER_MODEL_LGB = """\
## 📊 Выводы: LightGBM (§7.1)

**Интерпретация OOF (см. вывод ячейки выше):**

- Градиентный бустинг хорошо ловит нелинейности IC50/CC50 после **log1p**.
- Обычно входит в **топ-3** базовых моделей по competition_score.
- На **SI** (без log) ошибка выше — хвосты и асимметрия сложнее для одного регрессора.

**✅ Следующий шаг:** XGBoost (§7.2).
"""

AFTER_MODEL_XGB = """\
## 📊 Выводы: XGBoost (§7.2)

**Интерпретация OOF:**

- Близок к LightGBM по качеству; early stopping стабилизирует число деревьев.
- Даёт **диверсификацию** stacking: корреляция предсказаний с LGBM < 1.
- Линейные таргеты IC50/CC50 — сильная сторона; SI — слабее.

**✅ Следующий шаг:** RidgeCV (§7.3).
"""

AFTER_MODEL_RIDGE = """\
## 📊 Выводы: RidgeCV (§7.3)

**Интерпретация OOF:**

- Линейный **baseline**: competition_score заметно выше (хуже), чем у бустингов.
- Полезен в stacking как «якорь» — meta-модель может взвешивать его ниже.
- На масштабированных признаках после Preprocessor работает стабильно.

**✅ Следующий шаг:** ElasticNetCV (§7.4).
"""

AFTER_MODEL_ELASTIC = """\
## 📊 Выводы: ElasticNetCV (§7.4)

**Интерпретация OOF:**

- L1+L2: часть весов обнуляется → умеренная регуляризация на ~165 признаках.
- Качество близко к Ridge; различия в OOF небольшие.
- В ансамбле добавляет слабо коррелированный линейный сигнал.

**✅ Следующий шаг:** HistGradientBoosting (§7.5).
"""

AFTER_MODEL_HGB = """\
## 📊 Выводы: HistGradientBoosting (§7.5)

**Интерпретация OOF:**

- Быстрый бустинг sklearn; по OOF часто между линейными моделями и LGBM/XGB.
- Хорош на IC50/CC50; на SI — типичный разрыв с лучшими tree-моделями.
- Увеличивает разнообразие базового слоя без дублирования LGBM.

**✅ Следующий шаг:** RandomForest (§7.6).
"""

AFTER_MODEL_RF = """\
## 📊 Выводы: RandomForest (§7.6)

**Интерпретация OOF:**

- Bagging снижает дисперсию, но **bias** выше, чем у бустингов.
- OOF RMSE обычно хуже LGBM/XGB; модель всё равно полезна для stacking.
- Устойчив к выбросам в дескрипторах.

**✅ Следующий шаг:** ExtraTrees (§7.7).
"""

AFTER_MODEL_ET = """\
## 📊 Выводы: ExtraTrees (§7.7)

**Интерпретация OOF:**

- Случайные сплиты → ещё одна «ветка» tree-семейства с иной ошибкой.
- Сравним с RandomForest по OOF; вместе дают decorrelated predictions.
- На SI редко лидирует, но улучшает meta-слой.

**✅ Следующий шаг:** GradientBoosting sklearn (§7.8).
"""

AFTER_MODEL_GBR = """\
## 📊 Выводы: GradientBoosting sklearn (§7.8)

**Интерпретация OOF:**

- Классический GBR медленнее LGBM, но даёт третий независимый бустинг.
- OOF между HistGBR и LGBM по качеству.
- В финальном stacking (§7.10) вклад определяет Ridge meta.

**✅ Следующий шаг:** BayesianRidge (§7.9).
"""

AFTER_MODEL_BAYES = """\
## 📊 Выводы: BayesianRidge (§7.9)

**Интерпретация OOF:**

- Последняя базовая модель: вероятностная линейная регрессия.
- OOF близок к Ridge/ElasticNet — слабее tree-моделей.
- Завершает базовый слой из **9 моделей**; далее — meta-stacking (§7.10).

**✅ Следующий шаг:** Ridge meta-stacking (§7.10).
"""

AFTER_MODELS = """\
## 📊 Сводная аналитическая записка: §7 (все модели)

### Базовый слой (OOF, ClusterKFold)

| Модель | Роль в ансамбле |
|--------|-----------------|
| LightGBM, XGBoost | сильнейшие по IC50/CC50 |
| HistGBR, sklearn GBR | доп. бустинги, decorrelation |
| RandomForest, ExtraTrees | tree-bagging, устойчивость |
| Ridge, ElasticNet, BayesianRidge | линейные baselines для meta |

Точные **OOF competition_score** и RMSE по таргетам — в выводах ячеек §7.1–§7.10 и таблице §7.10.

### Meta-stacking (§7.10)

- **ClusterKFold** (KMeans на MolLogP, TPSA, RingCount, HeavyAtomCount)
- Per-fold **Preprocessor** (impute + scale, без утечки)
- OOF 9 моделей → **StandardScaler + RidgeCV** на каждый таргет
- **log1p / expm1** для IC50 и CC50

**Итог:** OOF stacking ~557, public LB **349.31** (Kaggle, команда 39).

**✅ Следующий шаг:** сводная таблица экспериментов (§8).
"""

AFTER_COMPARE = """\
## 📊 Выводы: сравнение (§8)

| Вариант | OOF | Public LB |
|---------|-----|-----------|
| Baseline ensemble | ~607 | ~363 |
| **Stacking (этот ноутбук)** | **~557** | **349.31** |
| + SI blend (submission3) | ~557 | 349.31 (w≈1) |
| GroupKFold (submission4) | — | 373.04 |

**Вывод:** stacking + ClusterKFold — лучший зафиксированный результат команды.

**✅ Следующий шаг:** сохранение `39_chemai_submission.csv` (§9).
"""

AFTER_VERIFY = """\
## 📊 Выводы: верификация submission (§9)

| Проверка | Результат |
|----------|-----------|
| Формат | 250 строк, `index, IC50, CC50, SI` |
| Файлы | `39_chemai_submission.csv` и `submission_avo_chemai_integrated.csv` (одинаковое содержимое) |
| OOF stacking | ≈ **557.22** (author4 submission2) |
| Эталон | max \\|Δ\\| < 10⁻⁵ vs `submission_avo_chemai_integrated.csv` или fingerprint JSON |

**Канонический путь к integrated:** `python tools/author4_high_priority_submissions.py --only 2` — тот же алгоритм, что §7.10.

**Colab:** перед §7 — **Runtime → Restart session**, затем **Run All** (иначе возможен «грязный» kernel).

**✅ Следующий шаг:** итоговые выводы.
"""

AFTER_SUBMISSION = """\
## §9. Итоговые выводы

1. Загрузили train/test с Kaggle, провели EDA (§2–§3).
2. Добавили domain-признаки, PCA и ICA (§4–§6).
3. Обучили **OOF stacking** из 9 моделей + Ridge meta (§7) — тот же подход, что дал **LB 349.31**.
4. Сохранили **`39_chemai_submission.csv`** и **`submission_avo_chemai_integrated.csv`** (эквивалент submission2).
5. Kaggle: команда **39**, public score **349.30995**, ранг ~90.

*Команда 39, группа М25-555: Анастасия Волконская, Мария Макарова, Артур Сидоров, Максим Власюк, Алина Давыденко.*
"""

LEADERBOARD_MD = """\
## Kaggle Leaderboard

Соревнование: [ChemAI: Predict the Cure](https://www.kaggle.com/competitions/chem-ai-predict-the-cure/overview)

- Команда **39**, лучший public score **349.30995**, ранг **~90**
- *(При сдаче можно приложить скриншот вкладки Leaderboard.)*
"""
