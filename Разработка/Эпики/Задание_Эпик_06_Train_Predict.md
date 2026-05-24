# Задание на разработку: Эпик 6 — Train / Predict пайплайн

## Контекст
Связывает все слои в воспроизводимый сценарий Kaggle submission.

## Файлы
`ml/src/chemai/train.py`, `ml/src/chemai/predict.py`, `run_pipeline.py`

## Задача
Сохранение `pipeline_bundle.joblib`, генерация `final_submission.csv`, `metrics.json`.

## Критерии приемки
- [x] Сквозной тест `test_pipeline_smoke.py` на синтетических CSV

**Зависимости:** Эпик 5
