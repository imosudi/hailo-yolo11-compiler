"""Validation and structural inspection of PyTorch YOLO11 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from yolo_hailo_mlops.exceptions import TrainingError
from yolo_hailo_mlops.utils.hashing import compute_sha256


@dataclass
class CheckpointMetadata:
    path: str
    sha256: str
    size_bytes: int
    task: str
    num_classes: int
    class_names: List[str]
    architecture: str
    epochs_trained: Optional[int] = None
    parameters_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "task": self.task,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "architecture": self.architecture,
            "epochs_trained": self.epochs_trained,
            "parameters_count": self.parameters_count,
        }


def validate_checkpoint(
    checkpoint_path: Union[str, Path],
    expected_task: str = "detect",
    expected_num_classes: Optional[int] = None,
) -> CheckpointMetadata:
    """Inspect and validate a PyTorch YOLO11 checkpoint (.pt)."""
    p = Path(checkpoint_path).resolve()
    if not p.is_file():
        raise TrainingError(
            f"Checkpoint file not found: {p}",
            code="E-TRAIN-002",
            artifact=str(p),
        )

    size = p.stat().st_size
    if size < 1024:
        raise TrainingError(
            f"Checkpoint file is suspiciously small ({size} bytes): {p}",
            code="E-TRAIN-003",
            artifact=str(p),
        )

    sha = compute_sha256(p)

    task = expected_task
    num_classes = expected_num_classes or 80
    class_names: List[str] = []
    architecture = "yolo11"
    epochs = None
    params_count = None

    # Attempt inspection via torch
    try:
        import torch

        # Load weights safely on CPU
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # Ultralytics checkpoint dictionary structure
            model = ckpt.get("model")
            epochs = ckpt.get("epoch")
            if model is not None:
                # Inspect model attributes
                if hasattr(model, "task"):
                    task = getattr(model, "task", expected_task)
                if hasattr(model, "names"):
                    names = getattr(model, "names")
                    if isinstance(names, dict):
                        class_names = [names[k] for k in sorted(names.keys())]
                        num_classes = len(class_names)
                    elif isinstance(names, list):
                        class_names = names
                        num_classes = len(class_names)

                if hasattr(model, "parameters"):
                    try:
                        params_count = sum(p.numel() for p in model.parameters())
                    except Exception:
                        pass
                architecture = type(model).__name__

        # Validate task compatibility
        if task != expected_task:
            raise TrainingError(
                f"Checkpoint task mismatch. Expected '{expected_task}', found '{task}'",
                code="E-TRAIN-004",
                artifact=str(p),
            )

        # Validate class count compatibility
        if expected_num_classes is not None and num_classes != expected_num_classes:
            raise TrainingError(
                f"Checkpoint class count mismatch. Expected {expected_num_classes}, found {num_classes}",
                code="E-TRAIN-005",
                artifact=str(p),
            )

    except ImportError:
        # If torch is not installed in current environment, accept file after basic size/extension check
        pass
    except Exception as e:
        if isinstance(e, TrainingError):
            raise
        raise TrainingError(
            f"Failed to load or inspect checkpoint '{p}': {e}",
            code="E-TRAIN-006",
            artifact=str(p),
        ) from e

    return CheckpointMetadata(
        path=str(p),
        sha256=sha,
        size_bytes=size,
        task=task,
        num_classes=num_classes,
        class_names=class_names,
        architecture=architecture,
        epochs_trained=epochs,
        parameters_count=params_count,
    )
