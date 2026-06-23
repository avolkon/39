"""Конфигурация через переменные окружения / .env (Pydantic Settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("ml/data"))
    models_dir: Path = Field(default=Path("ml/models_saved"))
    submissions_dir: Path = Field(default=Path("docs/submissions"))
    random_seed: int = Field(default=42)
    n_folds: int = Field(default=5, ge=2)
    n_clusters: int = Field(default=5, ge=2)
    log_transform_ic50_cc50: bool = Field(default=True)
    missing_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    use_neural_net: bool = Field(default=False)
    use_optuna: bool = Field(default=False)
    use_stacking: bool = Field(default=False)
    cv_strategy: Literal["cluster", "duplicate_group"] = Field(default="cluster")
    # 0 = без смеси (по умолчанию); (0,1] — SI := w·SI_model + (1-w)·CC50/IC50 (доменная связь)
    si_domain_blend: float = Field(default=0.0, ge=0.0, le=1.0)


_config_instance: Config | None = None


def load_config(env_file: Path | None = None) -> Config:
    """Инициализирует (или переустанавливает) глобальную конфигурацию."""
    global _config_instance
    _config_instance = Config(_env_file=env_file) if env_file is not None else Config()
    return _config_instance


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config_for_tests() -> None:
    global _config_instance
    _config_instance = None
