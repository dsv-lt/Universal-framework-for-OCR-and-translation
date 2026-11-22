"""Application settings and configuration persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


def detect_project_root() -> Path:
    """Resolve the repository/workspace root directory."""
    env_home = os.environ.get("IM_TRANSLATOR_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "packs").exists() and (cwd / "pyproject.toml").exists():
        return cwd

    pkg_root = Path(__file__).resolve().parents[3]
    if (pkg_root / "packs").exists() or (pkg_root / "pyproject.toml").exists():
        return pkg_root

    return cwd


class AppSettings(BaseModel):
    """Persistent runtime configuration for CLI and Web UI."""

    packs_dir: str = Field(default_factory=lambda: str(detect_project_root() / "packs"))
    models_dir: str = Field(default_factory=lambda: str(detect_project_root() / "models"))
    datasets_dir: str = Field(default_factory=lambda: str(detect_project_root() / "datasets"))
    device: Literal["auto", "cuda", "cpu", "dml", "mps"] = "auto"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "local-model"
    default_target_lang: str = "ru"

    @property
    def packs_path(self) -> Path:
        return Path(self.packs_dir).expanduser().resolve()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser().resolve()

    @property
    def datasets_path(self) -> Path:
        return Path(self.datasets_dir).expanduser().resolve()

    @property
    def templates_path(self) -> Path:
        return detect_project_root() / "docs" / "templates"


def get_config_path() -> Path:
    env_path = os.environ.get("IM_TRANSLATOR_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return detect_project_root() / "config.yaml"


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load AppSettings from YAML file or return defaults."""
    path = config_path or get_config_path()
    if not path.exists():
        return AppSettings()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        return AppSettings()
    return AppSettings.model_validate(raw)


def save_settings(settings: AppSettings, config_path: Path | None = None) -> Path:
    """Persist AppSettings to YAML file."""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(settings.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)
    return path
