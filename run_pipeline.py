"""CLI: обучение и инференс пайплайна ChemAI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chemai.predict import predict_pipeline
from chemai.train import train_pipeline
from chemai.utils.config import load_config
from chemai.utils.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="ChemAI train / predict pipeline")
    parser.add_argument("--train", action="store_true", help="Обучить модели и сохранить артефакты")
    parser.add_argument(
        "--predict", action="store_true", help="Сгенерировать docs/submissions/final_submission.csv"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к .env (опционально)",
    )
    args = parser.parse_args()

    load_config(env_file=args.config)
    setup_logging()
    log = logging.getLogger("chemai.run")

    if args.train == args.predict:
        log.error("Укажите ровно один режим: --train или --predict")
        return 2

    if args.train:
        train_pipeline()
    else:
        predict_pipeline()

    return 0


if __name__ == "__main__":
    sys.exit(main())
