.PHONY: install lint format test train predict eda-report

install:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check ml/src tests run_pipeline.py tools

format:
	uv run ruff format ml/src tests run_pipeline.py tools

test:
	uv run pytest -q

train:
	uv run python run_pipeline.py --train

predict:
	uv run python run_pipeline.py --predict

eda-report:
	uv run python tools/eda_generate_report.py
