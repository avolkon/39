.PHONY: install lint format test train predict

install:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check ml/src tests run_pipeline.py

format:
	uv run ruff format ml/src tests run_pipeline.py

test:
	uv run pytest -q

train:
	uv run python run_pipeline.py --train

predict:
	uv run python run_pipeline.py --predict
