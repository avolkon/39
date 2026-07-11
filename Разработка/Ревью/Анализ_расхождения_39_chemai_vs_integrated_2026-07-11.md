# Анализ расхождения: `39_chemai_submission.csv` vs `submission_avo_chemai_integrated.csv`

**Дата:** 2026-07-11  
**Kaggle score:** integrated **326.89** vs 39_chemai **343.36** (Δ = **+16.5**, хуже)  
**Папка артефактов:** `notebooks/`

---

## 1. Главный вывод

Файлы **не являются одним и тем же прогнозом**. `submission_avo_chemai_integrated.csv` — **точная копия** канонического `submission2_a4.csv` (author4, OOF stacking). `39_chemai_submission.csv` — **другой прогноз**, сильнее всего испорчен столбец **SI**; его скор на Kaggle (**343.36**) близок к **Epic 10 baseline (~345)**, а не к stacking (**~327**).

---

## 2. Сравнение файлов (локально)

| Файл | MD5 | IC50 mean | CC50 mean | SI mean | SI std | SI max |
|------|-----|--------:|----------:|--------:|-------:|-------:|
| `submission_avo_chemai_integrated.csv` | `4516ab8d…` | 298.28 | 752.98 | **106.67** | **34.47** | 215.19 |
| `docs/submissions/submission2_a4.csv` | `4516ab8d…` | 298.28 | 752.98 | 106.67 | 34.47 | 215.19 |
| `39_chemai_submission.csv` | `f8eb1375…` | 300.99 | 767.20 | **125.54** | **72.39** | **831.26** |

**Вывод:** integrated ≡ submission2_a4 (байтово/численно). 39_chemai — отдельный результат.

### 2.1. Попарные отличия (39 vs integrated)

| Метрика | IC50 | CC50 | SI |
|---------|-----:|-----:|---:|
| Корреляция | 0.990 | 0.995 | **0.534** |
| mean \|Δ\| | 24.8 | 42.1 | **31.5** |
| max \|Δ\| | 648 | 599 | **831** |
| Доля строк с \|Δ\|/ref > 5% | **64%** | **44%** | **88%** |
| Строк с \|Δ SI\| > 50 | — | — | **39 / 250** |

Пример выброса (index **182**):

| | IC50 | CC50 | SI |
|---|-----:|-----:|---:|
| integrated | 60.75 | 161.42 | **0.00** (clip после meta) |
| 39_chemai | 53.76 | 148.01 | **831.26** |

**Интерпретация:** meta-stacking по SI в каноническом пайплайне сглаживает предсказания; в 39_chemai SI раздувается на части строк → рост RMSE на платформе.

---

## 3. Два ноутбука — две линии происхождения

| | `avo_chemai_predict_the_cure.ipynb` | `39_chemai.ipynb` |
|---|-------------------------------------|-------------------|
| **Структура §7** | Один блок §7.0 + `run_stacking_submission2()` | 9 ячеек моделей + §7.10 meta |
| **Источник кода** | Inline (копия author4) | `scripts/notebook_section7.py` |
| **Имя CSV** | `submission_avo_chemai_integrated.csv` | `39_chemai_submission.csv` |
| **mtime ноутбука** | 2026-07-11 15:36 | 2026-07-11 13:53 |
| **mtime CSV в repo** | 2026-07-10 23:31 | **2026-07-11 15:23** |

### 3.1. Откуда взялся `integrated` (326.89)

- MD5 **совпадает** с `docs/submissions/submission2_a4.csv` и эталоном author4.
- Создан **2026-07-10** — до финальной сборки split-ноутбука `39_chemai.ipynb`.
- Происхождение: прогон **`tools/author4_high_priority_submissions.py --only 2`** (или копирование готового `submission2_a4.csv`), **не** успешный Run All ноутбука `avo_chemai` после 11.07.

### 3.2. Откуда взялся `39_chemai_submission` (343.36)

- Создан **2026-07-11 15:23** — после пересборки split-ноутбука.
- Сохранён из §9 `39_chemai.ipynb` (`submission_df.to_csv`).
- Численно **не совпадает** с submission2 → прогон §7.10 дал **иной** результат, чем author4.

---

## 4. Причины худшего скоринга (по приоритету)

### Причина 1 — разные CSV, не тот же пайплайн на выходе

Integrated = проверенный author4 stacking.  
39_chemai = другие предсказания; Kaggle score **343.36** логично хуже **326.89**.

### Причина 2 — SI/meta-слой повреждён в прогоне 39_chemai

- SI: corr с integrated **0.53** (IC50/CC50 ~0.99).
- 88% строк SI отличаются >5%.
- Скор **343 ≈ Epic 10 (345)** — типично для **ослабленного stacking / tree-heavy** прогноза без корректного meta по SI.

### Причина 3 — баги `fit_fold` в ранних версиях ноутбука

В `avo_chemai_predict_the_cure.ipynb` **до сих пор** (на 2026-07-11):

```python
def ridge_fold(x_tr, y_tr, *_a, _rs):  # TypeError при 5-м позиционном аргументе
```

Аналогично `bayes_fold`, в monolithic §7.0 — `elastic_fold`, `hgb_fold` с `*_a, rs`.

В `39_chemai.ipynb` это **исправлено** (`_x_va, _y_va, rs`), но:

- пользователь в Colab мог гонять **старую версию** до fix;
- при падении на Ridge/Elastic **OOF_STORE** остаётся неполным;
- §7.10 с `assert len(OOF_STORE)==9` должен падать — если assert обошли или kernel содержал **смешанное состояние** (часть моделей от старой сессии), meta Ridge обучается на **битой OOF-матрице** → искажённый SI на test.

### Причина 4 — архитектура split-ноутбука (риск Colab)

`39_chemai.ipynb` требует **строго по порядку**:

1. §7.0 (9 ячеек инфраструктуры)
2. §7.1–§7.9 (каждая модель ~1–2 мин)
3. §7.10 meta
4. §9 save

**Риски:**

- перезапуск не с начала → `OOF_STORE` / `CANDIDATES` из старой сессии;
- пропуск ячейки после ошибки;
- §9 без повторного §7.10;
- в §9 **нет** `assert allclose` к `submission2_a4` — битый CSV сохраняется без отлова.

### Причина 5 — `avo_chemai` не воспроизводит integrated при Run All

В `avo_chemai` остался баг `ridge_fold` → monolithic stacking **упадёт** на Ridge.  
Файл `integrated` появился **не** из этого Run All, а из author4.  
Если бы сдали CSV из упавшего/частичного прогона avo — результат был бы похож на 39_chemai.

### Причина 6 — не путать с «разными LB public/private»

326.89 vs 349.31 (public в соревновании) — разные **режимы оценки Kaggle**.  
326.89 vs 343.36 — сравнение **двух файлов в одной таблице заявок** (after deadline); здесь разница из‑за **содержимого CSV**, а не только public/private.

---

## 5. Хронология (реконструкция)

| Время | Событие |
|-------|---------|
| 2026-07-10 23:31 | `submission_avo_chemai_integrated.csv` = копия `submission2_a4` (author4) |
| 2026-07-11 ~13:53 | Пересборка `39_chemai.ipynb` (split §7, fix fit_fold) |
| 2026-07-11 15:23 | `39_chemai_submission.csv` — прогон split-ноутбука (Colab/локально), **некорректный stacking** |
| 2026-07-11 15:36 | Обновление `avo_chemai_predict_the_cure.ipynb` (bug ridge_fold **не** исправлен) |
| Kaggle upload | integrated → **326.89**; 39_chemai → **343.36** |

---

## 6. Рекомендации

1. **На Kaggle сдавать только** `submission_avo_chemai_integrated.csv` (или свежий author4 `submission2_a4.csv`).
2. **Перегенерировать** `39_chemai_submission.csv`: Run All `39_chemai.ipynb` с нуля (Runtime → Restart session), все §7.0–§7.10.
3. **Добавить в §9** проверку:
   ```python
   ref = pd.read_csv("submission_avo_chemai_integrated.csv")  # или submission2_a4
   assert np.allclose(submission_df[["IC50","CC50","SI"]], ref[["IC50","CC50","SI"]], rtol=1e-5)
   ```
4. **Синхронизировать или удалить** `avo_chemai_predict_the_cure.ipynb` — в нём остаётся баг `ridge_fold`.
5. **Не полагаться на MD5 CSV** — только `allclose` по колонкам прогноза.

---

## 7. Связанные документы

- [`Сводная_аналитика_пайплайнов_и_submission_2026-07-11.md`](Сводная_аналитика_пайплайнов_и_submission_2026-07-11.md)
- [`docs/eda/Аналитическая_записка_единый_ipynb_2026-07-11.md`](../../docs/eda/Аналитическая_записка_единый_ipynb_2026-07-11.md)
- `scripts/verify_notebook_submission.py`

---

*Подготовлено по сравнению CSV в `notebooks/` и анализу исходников ноутбуков.*
