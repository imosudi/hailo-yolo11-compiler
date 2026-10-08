"""Phase F: ONNX to Hailo Archive (HAR) parser abstraction."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import HailoEnvironmentError, HailoParseError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment
from yolo_hailo_mlops.utils.filesystem import atomic_write, ensure_dir, safe_symlink_or_copy
from yolo_hailo_mlops.utils.hashing import compute_sha256
from yolo_hailo_mlops.utils.subprocess import safe_run


class HailoParser:
    """Translates FP32 ONNX graphs to Hailo Archive (HAR) representation."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "hailo")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "hailo")

    def create_model_script(self, script_path: Path) -> Path:
        """Generate standard Hailo Model Script (.alls) for YOLO11."""
        script_content = (
            "# Hailo-8L Model Script for YOLO11\n"
            "# Normalisation layer: scale [0, 255] RGB to [0, 1] normalised float\n"
            "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
            "performance_param(compiler_optimization_level=0)\n"
        )
        return atomic_write(script_path, script_content)

    def parse(self, onnx_path: Union[str, Path]) -> Path:
        """Execute ONNX to HAR parsing."""
        onnx_p = Path(onnx_path).resolve()
        if not onnx_p.is_file():
            raise HailoParseError(
                f"Source ONNX model for parsing not found: {onnx_p}",
                code="E-HAILO-PARSE-002",
                artifact=str(onnx_p),
            )

        env = detect_hailo_environment()
        if not env.has_dfc:
            raise HailoEnvironmentError(
                "Hailo Dataflow Compiler (DFC) is required to parse ONNX into HAR.\n"
                "Hailo DFC was not found in the current Python environment or system PATH.",
                code="E-HAILO-ENV-002",
                remediation="Install the Hailo DFC package or run inside a configured Hailo compilation container.",
            )

        har_target = self.output_dir / "model.har"
        script_path = self.output_dir / "yolo11.alls"
        self.create_model_script(script_path)

        # 1. Python SDK mode
        if env.dfc_mode == "sdk" and self.config.hailo.sdk_mode in ("auto", "sdk"):
            try:
                from hailo_sdk_client import ClientRunner

                net_name = f"yolo11_{self.config.model.variant}"
                runner = ClientRunner(hw_arch=self.config.hailo.target)
                runner.translate_onnx_model(
                    model=str(onnx_p),
                    net_name=net_name,
                    start_node_names=["images"],
                    end_node_names=None,
                )
                runner.load_model_script(str(script_path))
                runner.save_har(str(har_target))
            except Exception as e:
                raise HailoParseError(
                    f"Hailo SDK ONNX parse failed: {e}",
                    code="E-HAILO-PARSE-003",
                    artifact=str(onnx_p),
                ) from e

        # 2. CLI mode
        else:
            hailo_bin = shutil.which("hailo")
            if not hailo_bin:
                raise HailoEnvironmentError("Hailo CLI binary not found in PATH", code="E-HAILO-ENV-003")

            cmd = [
                hailo_bin,
                "parser",
                "onnx",
                str(onnx_p),
                "--hw-arch",
                self.config.hailo.target,
                "--har-path",
                str(har_target),
                "--model-script",
                str(script_path),
            ]
            safe_run(cmd, error_code="E-HAILO-PARSE-004", phase="hailo_parse")

        if not har_target.is_file() or har_target.stat().st_size == 0:
            raise HailoParseError(
                f"HAR artifact generation failed: {har_target}",
                code="E-HAILO-PARSE-005",
                artifact=str(har_target),
            )

        safe_symlink_or_copy(har_target, self.canonical_dir / "model.har")
        return har_target
