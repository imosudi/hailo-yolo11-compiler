"""Unit tests for dataset pre-flight validation and format checks."""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image
import pytest

from yolo_hailo_mlops.dataset.validator import DatasetValidator
from yolo_hailo_mlops.exceptions import DatasetValidationError


def test_valid_dataset_validation(dummy_dataset: Path) -> None:
    """Verify that a properly structured YOLO dataset passes pre-flight validation."""
    validator = DatasetValidator(dummy_dataset)
    report = validator.validate(fail_fast=True)

    assert report.is_valid is True
    assert report.status == "PASSED"
    assert report.train_stats.num_images == 4
    assert report.val_stats.num_images == 2
    assert report.duplicate_images == 0
    assert report.train_val_leakage == 0
    assert report.num_classes == 1
    assert report.calibration_candidates == 4


def test_missing_dataset_yaml(temp_workspace: Path) -> None:
    """Validator must raise DatasetValidationError if YAML does not exist."""
    missing = temp_workspace / "missing_dataset.yaml"
    validator = DatasetValidator(missing)
    with pytest.raises(DatasetValidationError) as exc_info:
        validator.validate()
    assert "does not exist" in str(exc_info.value)


def test_dataset_train_val_leakage_detection(dummy_dataset: Path) -> None:
    """Validator must detect identical images across train and val splits."""
    ds_root = dummy_dataset.parent
    train_img = ds_root / "images" / "train" / "train_00.jpg"
    val_img = ds_root / "images" / "val" / "val_00.jpg"

    # Copy identical image bytes into val to simulate data leakage
    shutil.copy2(train_img, val_img)

    validator = DatasetValidator(dummy_dataset)
    with pytest.raises(DatasetValidationError) as exc_info:
        validator.validate(fail_fast=True)
    assert "leakage" in str(exc_info.value).lower()


def test_dataset_out_of_bounds_class_id(dummy_dataset: Path) -> None:
    """Validator must reject labels containing class IDs >= num_classes."""
    lbl_file = dummy_dataset.parent / "labels" / "train" / "train_00.txt"
    # Class ID 99 when nc is 1
    lbl_file.write_text("99 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    validator = DatasetValidator(dummy_dataset)
    report = validator.validate(fail_fast=False)
    assert report.train_stats.missing_labels > 0
