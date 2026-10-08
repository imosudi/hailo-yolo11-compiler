"""Quantitative evaluation for PyTorch FP32 baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.evaluation.metrics import AccuracyMetrics
from yolo_hailo_mlops.exceptions import AccuracyValidationError


def evaluate_pytorch_model(
    checkpoint_path: Union[str, Path],
    config: AppConfig,
) -> AccuracyMetrics:
    """Evaluate PyTorch checkpoint against dataset validation split."""
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise AccuracyValidationError(
            f"PyTorch checkpoint for evaluation not found: {ckpt}",
            code="E-ACC-002",
            artifact=str(ckpt),
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise AccuracyValidationError(
            "Ultralytics is required for PyTorch validation.",
            code="E-ACC-003",
        ) from e

    try:
        model = YOLO(str(ckpt))
        metrics = model.val(
            data=str(Path(config.dataset.yaml).resolve()),
            imgsz=config.export.imgsz,
            batch=config.training.batch,
            device=config.training.device if config.training.device != "auto" else None,
            verbose=config.verbose,
            save=False,
            plots=False,
        )

        map50 = float(getattr(metrics.box, "map50", 0.0))
        map5095 = float(getattr(metrics.box, "map", 0.0))
        precision = float(getattr(metrics.box, "mp", 0.0))
        recall = float(getattr(metrics.box, "mr", 0.0))

        # Per-class AP
        per_class: dict[str, float] = {}
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            names = getattr(model, "names", {})
            for idx, val in enumerate(metrics.box.maps):
                cname = names.get(idx, f"class_{idx}") if isinstance(names, dict) else f"class_{idx}"
                per_class[cname] = round(float(val), 4)

        return AccuracyMetrics(
            model_type="pytorch_fp32",
            map50=round(map50, 4),
            map50_95=round(map5095, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            per_class_ap50=per_class,
            samples_evaluated=len(getattr(metrics, "speed", {})),
        )

    except Exception as e:
        raise AccuracyValidationError(
            f"PyTorch validation execution failed: {e}",
            code="E-ACC-004",
            artifact=str(ckpt),
        ) from e
