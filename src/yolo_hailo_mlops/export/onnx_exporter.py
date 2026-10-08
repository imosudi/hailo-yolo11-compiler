"""Phase C: Static FP32 ONNX model export for Hailo compilation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import ExportError
from yolo_hailo_mlops.export.onnx_validator import ONNXValidationReport, validate_onnx_model
from yolo_hailo_mlops.utils.filesystem import ensure_dir, safe_symlink_or_copy


class ONNXExporter:
    """Exports PyTorch YOLO11 checkpoints to static FP32 ONNX graphs."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "onnx")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "onnx")

    def export(self, checkpoint_path: Union[str, Path]) -> ONNXValidationReport:
        """Export checkpoint to canonical FP32 static ONNX model."""
        ckpt = Path(checkpoint_path).resolve()
        if not ckpt.is_file():
            raise ExportError(
                f"Source checkpoint for export does not exist: {ckpt}",
                code="E-EXP-002",
                artifact=str(ckpt),
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ExportError(
                "Ultralytics package is required for ONNX export.",
                code="E-EXP-003",
                remediation="Run 'pip install -e .[training,onnx]' to install export dependencies.",
            ) from e

        # Post-processing decision (Section 11)
        # For Hailo-8L, raw heads without embedded NMS are preferred for Hailo DFC compilation
        use_nms = False
        postprocessing_mode = "hailo_runtime_nms"
        if self.config.export.nms == "embedded":
            use_nms = True
            postprocessing_mode = "embedded_nms"

        export_args: Dict[str, Any] = {
            "format": "onnx",
            "imgsz": self.config.export.imgsz,
            "batch": 1,             # Static batch = 1
            "dynamic": False,       # Static dimensions required
            "simplify": self.config.export.simplify,
            "nms": use_nms,
            "half": False,          # Canonical precision is FP32
        }
        if self.config.export.opset:
            export_args["opset"] = self.config.export.opset

        target_onnx = self.output_dir / "model.onnx"
        canonical_onnx = self.canonical_dir / "model.onnx"

        try:
            model = YOLO(str(ckpt))
            exported_path_str = model.export(**export_args)
            exported_path = Path(exported_path_str).resolve()

            if exported_path.resolve() != target_onnx.resolve():
                shutil.copy2(exported_path, target_onnx)

            safe_symlink_or_copy(target_onnx, canonical_onnx)

        except Exception as e:
            if isinstance(e, ExportError):
                raise
            raise ExportError(
                f"Failed to export checkpoint to ONNX: {e}",
                code="E-EXP-004",
                artifact=str(ckpt),
            ) from e

        # Validate exported ONNX model
        report = validate_onnx_model(target_onnx, expected_shape=(1, 3, self.config.export.imgsz, self.config.export.imgsz))
        return report
