"""Опциональная многозадачная сеть (PyTorch). Требует зависимость [nn]."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def train_multitask_mlp_stub(*args, **kwargs) -> None:  # noqa: ARG001
    """Зарезервировано: полноценное обучение при CHEM_USE_NEURAL_NET=true."""
    logger.warning("Нейросеть: заглушка; установите torch и доработайте train_multitask_mlp.")
    return None
