"""Unit tests for calibration sampling and preprocessing contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from yolo_hailo_mlops.calibration.preprocessor import letterbox_image, preprocess_calibration_image
from yolo_hailo_mlops.calibration.selector import CalibrationSelector
from yolo_hailo_mlops.config import AppConfig


def test_letterbox_image_dimensions() -> None:
    """Verify that letterboxing produces exact target spatial dimensions with padding."""
    # Create rectangular image 800x400
    orig = Image.new("RGB", (800, 400), color=(255, 0, 0))
    boxed, scale, pad = letterbox_image(orig, target_size=(640, 640), pad_color=(114, 114, 114))

    assert boxed.size == (640, 640)
    # Check scale factor: 640 / 800 = 0.8
    assert round(scale, 2) == 0.8
    # Height became 400 * 0.8 = 320, padding in Y is (640 - 320) / 2 = 160
    assert pad == (0, 160)

    # Pixel in pad area should be (114, 114, 114)
    pixel_pad = boxed.getpixel((320, 10))
    assert pixel_pad == (114, 114, 114)


def test_preprocess_calibration_image_contract(temp_workspace: Path) -> None:
    """Verify preprocessing contract: returns uint8 array of shape (640, 640, 3)."""
    img_path = temp_workspace / "sample.png"
    Image.new("RGB", (300, 500), color=(50, 100, 150)).save(img_path)

    arr = preprocess_calibration_image(img_path, target_size=(640, 640))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (640, 640, 3)
    assert arr.dtype == np.uint8


def test_calibration_selector_deterministic_generation(sample_config: AppConfig, temp_workspace: Path) -> None:
    """Calibration selector must generate manifest, statistics, and calib_data.npy."""
    run_dir = temp_workspace / "runs" / "test_run"
    selector = CalibrationSelector(sample_config, run_dir)
    npy_path, manifest_path = selector.generate()

    assert npy_path.is_file()
    assert manifest_path.is_file()

    data = np.load(str(npy_path))
    assert data.shape[0] == sample_config.dataset.calibration.count
    assert data.shape[1:] == (640, 640, 3)
