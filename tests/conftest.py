"""Сброс глобальной конфигурации между тестами."""

import pytest

from chemai.utils.config import reset_config_for_tests


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    reset_config_for_tests()
    yield
    reset_config_for_tests()
