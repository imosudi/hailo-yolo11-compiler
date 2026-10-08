"""Unit tests for configuration loading, precedence, and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yolo_hailo_mlops.config import AppConfig, load_config
from yolo_hailo_mlops.exceptions import ConfigurationError
from yolo_hailo_mlops.state import ExecutionMode


def test_default_config_loading(temp_workspace: Path) -> None:
    """Verify that configuration loads with defaults when file is empty."""
    empty_yaml = temp_workspace / "empty.yaml"
    empty_yaml.write_text("{}", encoding="utf-8")

    cfg = load_config(empty_yaml)
    assert cfg.project.name == "yolo11-hailo"
    assert cfg.project.seed == 42
    assert cfg.model.variant == "n"
    assert cfg.hailo.target == "hailo8l"
    assert cfg.export.batch == 1
    assert cfg.export.dynamic is False


def test_cli_precedence_over_file_and_env(temp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify precedence: CLI overrides > Env vars > YAML."""
    yaml_file = temp_workspace / "test.yaml"
    yaml_file.write_text(
        "training:\n  epochs: 50\n  imgsz: 640\nproject:\n  seed: 10\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("YOLO_HAILO_EPOCHS", "75")
    monkeypatch.setenv("YOLO_HAILO_SEED", "20")

    # 1. With env vars active, env overrides file
    cfg1 = load_config(yaml_file)
    assert cfg1.training.epochs == 75
    assert cfg1.project.seed == 20

    # 2. With CLI overrides, CLI overrides both env and file
    cli_overrides = {"epochs": 120, "seed": 99}
    cfg2 = load_config(yaml_file, cli_overrides=cli_overrides)
    assert cfg2.training.epochs == 120
    assert cfg2.project.seed == 99


def test_config_validation_invalid_target(temp_workspace: Path) -> None:
    """Hailo target must be 'hailo8l'."""
    invalid_yaml = temp_workspace / "invalid.yaml"
    invalid_yaml.write_text("hailo:\n  target: hailo8\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(invalid_yaml)
    assert "Target must be 'hailo8l'" in str(exc_info.value)


def test_config_validation_dynamic_export_rejected(temp_workspace: Path) -> None:
    """Dynamic export must be rejected."""
    cfg = AppConfig()
    cfg.export.dynamic = True
    with pytest.raises(ConfigurationError):
        cfg.validate()


def test_config_validation_accuracy_gate_bounds(temp_workspace: Path) -> None:
    """Accuracy gate thresholds must be between 0.0 and 1.0."""
    cfg = AppConfig()
    cfg.validation.accuracy.map50_max_drop = 1.5
    with pytest.raises(ConfigurationError):
        cfg.validate()
