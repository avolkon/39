# Журнал экспериментов — Фаза 0

**Подготовка:** Артур Сидоров, Максим Власюк (stacking / ablation)

## 0.2 Adversarial validation (train vs test)

- AUC (5-fold CV): **0.4806** ± 0.0194
- Веса train: `ml\models_saved\adversarial_sample_weights.json`

Топ признаков (|coef| logistic):

- `VSA_EState7`: 0.3318
- `fr_unbrch_alkane`: 0.2893
- `Arom_Heavy_ratio`: 0.2673
- `fr_nitro_arom`: 0.2519
- `VSA_EState2`: 0.2469
- `VSA_EState5`: 0.2374
- `fr_nitro_arom_nonortho`: 0.2344
- `PEOE_VSA2`: 0.2228

## 0.3 OOF stacking (baseline v2, `CHEM_USE_STACKING=true`)

- **OOF competition_score:** 556.9457
- RMSE IC50: 327.2171
- RMSE CC50: 555.5920
- RMSE SI: 788.0281

## 0.4 OOF vs public LB (исторические отправки)

| Вариант | OOF (этот прогон) | Public LB (зафикс.) | Δ OOF−LB |
|---------|-------------------|---------------------|----------|
| Stacking cluster (≈submission2) | 556.95 | 349.31 | +207.64 |
| Group split (≈submission4) | — | 373.04 | — |

## 0.5 Ablation SI (на OOF stacking IC50/CC50)

| Вариант | OOF mean RMSE | RMSE SI |
|---------|---------------|---------|
| Stacking SI (как submission2_a4) | 556.9457 | 788.0281 |
| SI = CC50/IC50 из OOF IC50/CC50 | 558.6055 | 793.0075 |
| Blend w=1.000 | 556.9457 | 788.0281 |

**Gate Фазы 0**

- OOF stacking v2 = **556.95** → **FAIL** (порог ≤320 не достигнут)
- **Решение:** переход к **Фазе 1** (fix CV R1/R2, OOF meta) — см. [`ML_ревью_2026-06-22_фаза0_диагностика.md`](../../Разработка/Ревью/ML_ревью_2026-06-22_фаза0_диагностика.md)
- Дубликаты: **24.1%** строк в multi-groups; **49/60** групп с конфликтом IC50
