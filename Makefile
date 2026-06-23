.PHONY: install lint format test train predict phase0 train-stacking

install:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check ml/src tests run_pipeline.py run_phase0_diagnostics.py

format:
	uv run ruff format ml/src tests run_pipeline.py run_phase0_diagnostics.py

test:
	uv run pytest -q

train:
	uv run python run_pipeline.py --train

train-stacking:
	CHEM_USE_STACKING=true uv run python run_pipeline.py --train

predict:
	uv run python run_pipeline.py --predict

phase0:
	uv run python run_phase0_diagnostics.py
