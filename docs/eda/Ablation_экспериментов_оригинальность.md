# Ablation: эксперименты и элементы оригинальности

**Проект:** ChemAI Predict the Cure, команда 39  
**Назначение:** обоснование для критерия «оригинальность» и «обоснование решения»

---

## 1. Элементы оригинальности в текущем коде

| Элемент | Где | Зачем |
|---------|-----|-------|
| **ClusterKFold** | `validation/cv_splitter.py` | ~16% дубликатов признаков — смягчение оптимистичной CV |
| **Per-fold Preprocessor** | `train.py` | Честная валидация без утечки scaler/imputer |
| **Доменные признаки** | `features/build_features.py` | LogP/TPSA, arom/heavy, заряд, флаги fr_*, Ring×LogP, FractionCSP3×LogP |
| **log1p / expm1** для IC50/CC50 | `train.py`, `Expm1Predictor` | Асимметрия концентраций (EDA) |
| **Опциональная SI-смесь** | `postprocess.py`, `CHEM_SI_DOMAIN_BLEND` | Связь SI≈CC50/IC50 на train; по умолчанию **выкл.** |
| **competition_score** | `utils/metrics.py` | Метрика платформы = mean(RMSE×3) |

---

## 2. Таблица экспериментов (архив 2026-06-22)

| ID | Гипотеза | Public LB | Итог |
|----|----------|-----------|------|
| baseline repo | Weighted ensemble, 9 моделей | 363.07 | Воспроизводимое ядро |
| submission2_a4 | OOF stacking 2-го уровня | **349.31** | Лучший LB команды |
| submission3_a4 | + SI blend на OOF | 349.31 | w≈1 — смесь не помогла |
| submission4_a4 | GroupKFold по дубликатам | 373.04 | OOF↑, LB↓ |
| Epic 10 (откат) | clip + tree-only | **345.08** | Лучший репо-LB офлайн |
| Epic 11 (откат) | duplicate_group + CatBoost | 354.63 | OOF↓, LB↑ — не канон |

Подробности: [`Разработка/Бэклог/Наработки_2026-06-22_после_закрытия_Kaggle.md`](../../Разработка/Бэклог/Наработки_2026-06-22_после_закрытия_Kaggle.md).

---

## 3. Что **не** включено в git (и почему)

| Наработка | Причина отката |
|-----------|----------------|
| Stacking в `train.py` | LB 1447 — нестабильно |
| duplicate_group CV | LB хуже cluster |
| Полный SI blend в train | w=1 на OOF — нет эффекта |

---

## 4. Безопасное использование SI blend

```powershell
# Эксперимент (как submission3_a4); по умолчанию 0 — поведение baseline
$env:CHEM_SI_DOMAIN_BLEND="0.5"
python run_pipeline.py --predict
```

Полный подбор веса — только по OOF на train, не на test.
