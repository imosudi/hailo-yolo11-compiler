"""Pytest configuration and common testing fixtures."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

# Ensure src/ and repo root are on Python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from PIL import Image

from yolo_hailo_mlops.config import AppConfig, load_config
from yolo_hailo_mlops.utils.filesystem import atomic_write


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provide isolated temporary directory for test executions."""
    tmp = Path(tempfile.mkdtemp(prefix="test_yolo_hailo_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def dummy_dataset(temp_workspace: Path) -> Path:
    """Create a minimal valid YOLO dataset with images and labels."""
    ds_root = temp_workspace / "data"
    images_train = ds_root / "images" / "train"
    images_val = ds_root / "images" / "val"
    labels_train = ds_root / "labels" / "train"
    labels_val = ds_root / "labels" / "val"

    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    # Create 4 train images and labels
    for i in range(4):
        img_p = images_train / f"train_{i:02d}.jpg"
        img = Image.new("RGB", (640, 640), color=(i * 40, 100, 150))
        img.save(img_p)

        lbl_p = labels_train / f"train_{i:02d}.txt"
        with open(lbl_p, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")

    # Create 2 val images and labels
    for i in range(2):
        img_p = images_val / f"val_{i:02d}.jpg"
        img = Image.new("RGB", (640, 640), color=(200, i * 50, 100))
        img.save(img_p)

        lbl_p = labels_val / f"val_{i:02d}.txt"
        with open(lbl_p, "w", encoding="utf-8") as f:
            f.write("0 0.4 0.6 0.15 0.25\n")

    yaml_path = ds_root / "dataset.yaml"
    yaml_content = (
        f"path: {ds_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['object']\n"
    )
    atomic_write(yaml_path, yaml_content)
    return yaml_path


@pytest.fixture
def sample_config(dummy_dataset: Path, temp_workspace: Path) -> AppConfig:
    """Provide a validated AppConfig instance configured for test runs."""
    config_dict = {
        "project": {"name": "test-yolo-hailo", "seed": 42},
        "model": {"source": "yolo11n.pt", "variant": "n", "task": "detect"},
        "dataset": {
            "yaml": str(dummy_dataset),
            "calibration": {"source": "train", "count": 2, "seed": 42},
        },
        "training": {
            "epochs": 1,
            "imgsz": 640,
            "batch": 2,
            "device": "cpu",
            "optimizer": "auto",
            "patience": 5,
            "deterministic": True,
        },
        "export": {
            "imgsz": 640,
            "batch": 1,
            "dynamic": False,
            "precision": "fp32",
            "simplify": True,
            "nms": "auto",
        },
        "hailo": {
            "target": "hailo8l",
            "sdk_mode": "auto",
            "require_hardware": False,
        },
        "quantization": {
            "mode": "int8_ptq",
            "optimisation": True,
        },
        "validation": {
            "enabled": True,
            "accuracy": {"map50_max_drop": 0.05, "map5095_max_drop": 0.05},
            "performance": {"warmup_iterations": 2, "measurement_iterations": 5},
        },
        "artifacts": {
            "root": str(temp_workspace / "artifacts"),
        },
    }
    cfg_path = temp_workspace / "config.yaml"
    import yaml
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    return load_config(cfg_path)
