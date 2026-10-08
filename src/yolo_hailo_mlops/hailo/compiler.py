"""Phase H: Hailo-8L compilation and HEF binary generation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import HailoCompilationError, HailoEnvironmentError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment
from yolo_hailo_mlops.utils.filesystem import ensure_dir, safe_symlink_or_copy
from yolo_hailo_mlops.utils.subprocess import safe_run


class HailoCompiler:
    """Compiles optimised HAR models to Hailo Executable Format (HEF) binaries."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "hailo")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "hailo")

    def compile(self, quant_har_path: Union[str, Path]) -> Path:
        """Compile quantised HAR specifically for the Hailo-8L target architecture."""
        har_p = Path(quant_har_path).resolve()
        if not har_p.is_file():
            raise HailoCompilationError(
                f"Optimised HAR for compilation does not exist: {har_p}",
                code="E-HAILO-COMP-002",
                artifact=str(har_p),
            )

        env = detect_hailo_environment()
        if not env.has_dfc:
            raise HailoEnvironmentError(
                "Hailo Dataflow Compiler (DFC) is required to compile to HEF.\n"
                "Hailo DFC was not found in the environment.",
                code="E-HAILO-ENV-006",
            )

        target_arch = self.config.hailo.target
        if target_arch != "hailo8l":
            raise HailoCompilationError(
                f"Target accelerator must be 'hailo8l', got '{target_arch}'",
                code="E-HAILO-COMP-003",
            )

        hef_target = self.output_dir / "model.hef"

        # 1. Python SDK mode
        if env.dfc_mode == "sdk" and self.config.hailo.sdk_mode in ("auto", "sdk"):
            try:
                from hailo_sdk_client import ClientRunner

                runner = ClientRunner(har=str(har_p))
                hef = runner.compile()
                with open(hef_target, "wb") as f:
                    f.write(hef)
            except Exception as e:
                raise HailoCompilationError(
                    f"Hailo SDK compilation failed: {e}",
                    code="E-HAILO-COMP-004",
                    artifact=str(har_p),
                ) from e

        # 2. CLI mode
        else:
            hailo_bin = shutil.which("hailo")
            if not hailo_bin:
                raise HailoEnvironmentError("Hailo CLI binary not found in PATH", code="E-HAILO-ENV-007")

            cmd = [
                hailo_bin,
                "compiler",
                str(har_p),
                "--hw-arch",
                target_arch,
                "--output-hef-path",
                str(hef_target),
            ]
            safe_run(cmd, error_code="E-HAILO-COMP-005", phase="hailo_compile")

        if not hef_target.is_file() or hef_target.stat().st_size == 0:
            raise HailoCompilationError(
                f"HEF generation failed or produced empty file: {hef_target}",
                code="E-HAILO-COMP-006",
                artifact=str(hef_target),
            )

        safe_symlink_or_copy(hef_target, self.canonical_dir / "model.hef")
        return hef_target
