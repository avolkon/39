# ChemAI: Predict the Cure (репозиторий команды «39»)

Табличная регрессия **IC50**, **CC50**, **SI** по числовым дескрипторам молекул: пакет **`chemai`** в **`ml/src/chemai/`**, интерфейс командной строки — **`run_pipeline.py`** из корня репозитория.

## Что воспроизводит код (финальный пайплайн)

После **`--train`** в каталог моделей попадают (пути по умолчанию см. переменные `CHEM_*`):

- **`ml/models_saved/preprocessor.joblib`**
- **`ml/models_saved/pipeline_bundle.joblib`**
- **`ml/models_saved/metrics.json`**

После **`--predict`** записывается **`docs/submissions/final_submission.csv`** (`index`, `IC50`, `CC50`, `SI`) — см. конфиг **`CHEM_SUBMISSIONS_DIR`**.

Обучение: **`ml/src/chemai/train.py`** через **`run_pipeline.py --train`** — кросс‑валидация **`ClusterKFold`**, фильтрация признаков и масштабирование через **`Preprocessor`**, для каждого таргета **ансамбль** базовых регрессоров; из фолдовых RMSE выводятся **веса** (обратное среднее, нормировка по таргету). Лог‑признак **`log1p`** для IC50/CC50 включается флагом из конфига (в `.env` — **`CHEM_LOG_TRANSFORM_IC50_CC50`**).

## Требования

- **Python ≥ 3.11** (для Windows 11 удобно **3.12+**).

## Конфигурация

Жёсткий **набор по умолчанию** задаёт **`ml/src/chemai/utils/config.py`** (переменные окружения с префиксом **`CHEM_`** или файл **`.env`**). Полный список переменных — в **`.env.example`**; здесь главные каталоги:

| Переменная / смысл | По умолчанию |
|--------------------|---------------|
| `CHEM_DATA_DIR` | `ml/data` |
| `CHEM_MODELS_DIR` | `ml/models_saved` |
| `CHEM_SUBMISSIONS_DIR` | `docs/submissions` |
| `CHEM_N_FOLDS` | `5` |
| `CHEM_N_CLUSTERS` | `5` |
| остальные (`CHEM_RANDOM_SEED`, `CHEM_LOG_TRANSFORM_IC50_CC50`, `CHEM_MISSING_THRESHOLD`, …) | см. **`.env.example`** |

Практично скопировать **`.env.example` → `.env`** и поправить пути при необходимости:

```powershell
Copy-Item .env.example .env
```

Пример явного указания конфига: `python run_pipeline.py --train --config .env`

## Данные

Положите выдачу платформы в **`ml/data/`** (CSV **не коммитятся** — см. `.gitignore`).

| Файл | Назначение |
|------|------------|
| **`train.csv`** | обучение (**обязателен** для `--train`) |
| **`test.csv`** | инференс (**обязателен** для `--predict**) |
| `sample_submission.csv` | необязательно, подсказка по столбцам ответа |

Имена `train.csv` и `test.csv` заданы в **`ml/src/chemai/utils/data_loader.py`** (без переименования или правок кода / `CHEM_DATA_DIR`).

## Пошаговый запуск (Windows, PowerShell)

Рабочая директория — **корень репозитория** (рядом `run_pipeline.py`, `pyproject.toml`).

1. **Данные** — убедитесь, что существуют `.\ml\data\train.csv` и `.\ml\data\test.csv`:
   ```powershell
   Test-Path .\ml\data\train.csv
   Test-Path .\ml\data\test.csv
   ```
2. **Виртуальное окружение** (пример под Python 3.12):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Зависимости:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```
   Опционально **CatBoost** (дополнительный кандидат в ансамбле при установленном пакете):
   ```powershell
   pip install -e ".[dev,catboost]"
   ```
4. **Обучение:**
   ```powershell
   python run_pipeline.py --train
   ```
5. **Предсказание и submission:**
   ```powershell
   python run_pipeline.py --predict
   ```
   Результат: **`docs/submissions/final_submission.csv`** (или путь из `CHEM_SUBMISSIONS_DIR`).

При изменении исходников полезная проверка:

```powershell
ruff check ml/src tests run_pipeline.py
pytest -q
```

## Запуск через `uv` и `Makefile`

```bash
uv venv && uv pip install -e ".[dev]"
uv run python run_pipeline.py --train
uv run python run_pipeline.py --predict
```

или из корня: **`make train`**, **`make predict`**, **`make lint`**, **`make test`** (см. `Makefile`).

## Состав базовых моделей (англ. имена совпадают с кодом)

Девять семейств в **`build_default_candidates`**: LightGBM, XGBoost, RidgeCV (+ `StandardScaler`), ElasticNetCV (+ `StandardScaler`), HistGradientBoosting, RandomForest, ExtraTrees, GradientBoosting (sklearn), BayesianRidge (+ `StandardScaler`). При наличии пакета **catboost** добавляется дополнительный кандидат.

## Метрика платформы (как при скоринге)

`score = (RMSE(IC50) + RMSE(CC50) + RMSE(SI)) / 3`.

## Оригинальные элементы решения

Помимо стандартного QSPR-стека команда явно учитывает домен и структуру данных:

- **ClusterKFold** — кластеризация по MolLogP/TPSA/HeavyAtomCount/RingCount вместо наивного KFold при ~16% дубликатов признаков.
- **Доменные производные признаки** (`build_features.py`): LogP/TPSA, arom/heavy, зарядовые суммы, бинарные флаги функциональных групп, Ring×LogP, FractionCSP3×LogP.
- **Опциональная post-hoc согласованность SI** с отношением CC50/IC50 (`CHEM_SI_DOMAIN_BLEND`, по умолчанию `0`) — воспроизводит идею эксперимента submission3_a4 без изменения обучения.
- Таблица ablation и архив экспериментов: **`docs/eda/Ablation_экспериментов_оригинальность.md`**.

## Документация и отчётность команды

- Разведочный анализ: **`docs/eda/EDA_Report.md`**, графики в **`docs/eda/figures/`** (пересборка автоматическим скриптом в этом репозитории не подключена; при обновлении данных правьте вручную при необходимости).
- Аналитика по архитектуре моделей: **`docs/eda/Аналитическая_записка_архитектура_моделей.md`**.
- Итоговая сводка (датасет, эксперименты, лидерборд — по состоянию на сохранённые материалы): **`docs/eda/Итоговая_аналитическая_записка_Predict_the_Cure.md`**.
- Требования организаторов см. **`Разработка/подготовка/ТЗ_Хакатон.txt`**.
