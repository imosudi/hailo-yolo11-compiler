"""Unit tests for typed exception taxonomy and diagnostic formatting."""

from __future__ import annotations

import pytest

from yolo_hailo_mlops.exceptions import (
    AccuracyValidationError,
    CalibrationError,
    ConfigurationError,
    DatasetValidationError,
    HailoCompilationError,
    HailoEnvironmentError,
    YoloHailoError,
)


def test_exception_diagnostic_formatting() -> None:
    """Exception string must contain code, phase, cause, and remediation."""
    err = ConfigurationError(
        message="Invalid batch size",
        code="E-CFG-999",
        phase="configuration",
        remediation="Set training.batch to a positive integer.",
        artifact="config/config.yaml",
    )
    formatted = str(err)
    assert "[E-CFG-999]" in formatted
    assert "Phase: configuration" in formatted
    assert "Cause: Invalid batch size" in formatted
    assert "Affected Artifact: config/config.yaml" in formatted
    assert "Remediation:" in formatted
    assert "Set training.batch" in formatted


def test_exception_hierarchy() -> None:
    """All domain exceptions must inherit from YoloHailoError."""
    assert issubclass(DatasetValidationError, YoloHailoError)
    assert issubclass(CalibrationError, YoloHailoError)
    assert issubclass(HailoCompilationError, YoloHailoError)
    assert issubclass(HailoEnvironmentError, YoloHailoError)
    assert issubclass(AccuracyValidationError, YoloHailoError)
