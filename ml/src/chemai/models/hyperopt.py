"""Гиперпараметры: шаблон для Optuna (опционально)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def default_params_lgb() -> dict[str, Any]:
    from chemai.models.lgb_model import default_lgb_params

    return default_lgb_params()


def default_params_xgb() -> dict[str, Any]:
    from chemai.models.xgb_model import default_xgb_params

    return default_xgb_params()


def save_params(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_params(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_optuna_stub() -> None:
    logger.info("Optuna: задайте CHEM_USE_OPTUNA=true и реализуйте objective в hyperopt.py.")
