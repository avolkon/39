# 39 ChemAI: Predict the Cure

Команда фармацевтической разработки ML и Data-инженеров строит модели, предсказывающие **IC50**, **CC50** и **SI** по 214 молекулярным дескрипторам (см. `ТЗ_Хакатон.txt` в `Разработка/подготовка/`).

## Требования

- Python **3.11+**
- Файлы соревнования: `data/train.csv`, `data/test.csv` (и опционально `data/sample_submission.csv`)

## Установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

С [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e ".[dev]"
```

## Обучение

```bash
python run_pipeline.py --train
```

Сохраняется: `models_saved/preprocessor.joblib`, `models_saved/pipeline_bundle.joblib`, `models_saved/metrics.json`.

## Предсказание

```bash
python run_pipeline.py --predict
```

Результат: `submissions/final_submission.csv` с колонками `index,IC50,CC50,SI`.

## Модели (ансамбль)

На каждый таргет обучаются **не менее девяти** различных типов регрессоров; для IC50/CC50/SI **отдельно** задокументирована применимость (`Разработка/Эпики/Анализ_модельных_вариантов.md`), **веса** в ансамбле считаются по CV **независимо для каждого таргета**. Состав: LightGBM, XGBoost, RidgeCV, ElasticNetCV, HistGradientBoosting, RandomForest, ExtraTrees, GradientBoosting (sklearn), BayesianRidge; при установленном пакете — **CatBoost** (`pip install -e ".[dev,catboost]"`).

## Конфигурация

Скопируйте `.env.example` в `.env` и при необходимости измените пути (`CHEM_DATA_DIR`, `CHEM_MODELS_DIR`, `CHEM_SUBMISSIONS_DIR`) и `CHEM_N_FOLDS`, `CHEM_N_CLUSTERS`.

## Качество кода

```bash
ruff check src tests run_pipeline.py
pytest -q
```

## План и эпики

См. `Разработка/Эпики/Эпики_и_план_реализации.md`, принципы ревью — `ПРИНЦИПЫ_РЕВЬЮ_и_перенос_на_ChemAI.md`, ИБ — `ИБ_анализ.md`.

## Метрика соревнования

`score = (RMSE(IC50) + RMSE(CC50) + RMSE(SI)) / 3`.
