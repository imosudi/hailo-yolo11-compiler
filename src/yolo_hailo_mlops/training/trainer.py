"""Ultralytics YOLO11 model training orchestrator."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import TrainingError
from yolo_hailo_mlops.training.validator import CheckpointMetadata, validate_checkpoint
from yolo_hailo_mlops.utils.filesystem import ensure_dir, safe_symlink_or_copy


class YOLOTrainer:
    """Encapsulates Ultralytics YOLO11 training execution."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "pytorch")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "pytorch")

    def _set_seeds(self) -> None:
        """Enforce deterministic operation across RNGs."""
        seed = self.config.project.seed
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.config.training.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def run(self) -> CheckpointMetadata:
        """Execute training or fine-tuning and return checkpoint metadata."""
        self._set_seeds()
        source_model = self.config.model.source

        # Check if source_model is an existing pre-trained file that user wants to use directly
        source_path = Path(source_model)
        target_best_pt = self.output_dir / "best.pt"
        canonical_best_pt = self.canonical_dir / "best.pt"

        # If source_model exists and training epochs is 0 (or bypass training requested)
        if source_path.is_file() and getattr(self.config, "skip_train_step", False):
            shutil.copy2(source_path, target_best_pt)
            safe_symlink_or_copy(target_best_pt, canonical_best_pt)
            return validate_checkpoint(target_best_pt, expected_task=self.config.model.task)

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise TrainingError(
                "Ultralytics package is not installed. Install via 'pip install ultralytics'.",
                code="E-TRAIN-007",
                remediation="Run 'pip install -e .[training]' to install training dependencies.",
            ) from e

        train_args: Dict[str, Any] = {
            "data": str(Path(self.config.dataset.yaml).resolve()),
            "epochs": self.config.training.epochs,
            "imgsz": self.config.training.imgsz,
            "batch": self.config.training.batch,
            "device": self.config.training.device if self.config.training.device != "auto" else None,
            "patience": self.config.training.patience,
            "seed": self.config.project.seed,
            "deterministic": self.config.training.deterministic,
            "project": str(self.output_dir),
            "name": "yolo_train",
            "exist_ok": True,
            "verbose": self.config.verbose,
        }

        if self.config.training.optimizer != "auto":
            train_args["optimizer"] = self.config.training.optimizer
        if self.config.training.lr0 is not None:
            train_args["lr0"] = self.config.training.lr0
        if self.config.training.lrf is not None:
            train_args["lrf"] = self.config.training.lrf

        try:
            model = YOLO(source_model)
            results = model.train(**train_args)

            # Locate produced best.pt
            best_candidates = list(self.output_dir.rglob("best.pt"))
            if not best_candidates:
                # If only last.pt exists
                last_candidates = list(self.output_dir.rglob("last.pt"))
                if last_candidates:
                    best_pt = last_candidates[0]
                else:
                    raise TrainingError(
                        "Training completed but no 'best.pt' or 'last.pt' weights found.",
                        code="E-TRAIN-008",
                        artifact=str(self.output_dir),
                    )
            else:
                best_pt = best_candidates[0]

            # Copy to canonical destination
            if best_pt.resolve() != target_best_pt.resolve():
                shutil.copy2(best_pt, target_best_pt)

            safe_symlink_or_copy(target_best_pt, canonical_best_pt)

        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(
                f"YOLO11 training failed: {e}",
                code="E-TRAIN-009",
                artifact=str(self.output_dir),
            ) from e

        return validate_checkpoint(target_best_pt, expected_task=self.config.model.task)
