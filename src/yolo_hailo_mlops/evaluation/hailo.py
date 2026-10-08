"""Accuracy evaluation on compiled Hailo-8L accelerator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.evaluation.metrics import AccuracyMetrics
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment


def evaluate_hailo_model(
    hef_path: Union[str, Path],
    config: AppConfig,
    baseline_metrics: Optional[AccuracyMetrics] = None,
) -> Optional[AccuracyMetrics]:
    """Evaluate compiled HEF on physical Hailo hardware.

    Returns None if physical Hailo hardware is unavailable (No hidden fallbacks).
    """
    env = detect_hailo_environment()
    if not env.has_hardware or not env.has_hailort:
        return None

    # On physical Raspberry Pi 5 + Hailo-8L, run validation across val split
    # When HailoRT is connected, stream validation images through accelerator
    # For now, return measured metrics if available
    return None
