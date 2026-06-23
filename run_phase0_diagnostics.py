"""Фаза 0: диагностика дубликатов, adversarial validation, SI ablation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chemai.diagnostics.phase0 import run_phase0_diagnostics
from chemai.utils.config import load_config
from chemai.utils.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="ChemAI — диагностика Фазы 0")
    parser.add_argument("--config", type=Path, default=None, help="Путь к .env")
    args = parser.parse_args()

    load_config(env_file=args.config)
    setup_logging()
    log = logging.getLogger("chemai.phase0")
    log.info("Запуск диагностики Фазы 0")
    run_phase0_diagnostics()
    return 0


if __name__ == "__main__":
    sys.exit(main())
