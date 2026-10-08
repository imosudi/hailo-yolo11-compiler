"""Phase G: Hailo INT8 Post-Training Quantisation (PTQ) and layer optimisation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import HailoEnvironmentError, HailoOptimisationError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment
from yolo_hailo_mlops.utils.filesystem import ensure_dir, safe_symlink_or_copy
from yolo_hailo_mlops.utils.subprocess import safe_run


class HailoOptimizer:
    """Performs INT8 Post-Training Quantisation on parsed Hailo Archive (HAR)."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "hailo")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "hailo")

    def optimize(
        self,
        har_path: Union[str, Path],
        calib_data_path: Union[str, Path],
    ) -> Path:
        """Run INT8 post-training quantisation against calibration dataset."""
        har_p = Path(har_path).resolve()
        calib_p = Path(calib_data_path).resolve()

        if not har_p.is_file():
            raise HailoOptimisationError(
                f"Source HAR file not found: {har_p}",
                code="E-HAILO-OPT-002",
                artifact=str(har_p),
            )
        if not calib_p.is_file():
            raise HailoOptimisationError(
                f"Calibration dataset file not found: {calib_p}",
                code="E-HAILO-OPT-003",
                artifact=str(calib_p),
            )

        env = detect_hailo_environment()
        if not env.has_dfc:
            raise HailoEnvironmentError(
                "Hailo DFC is required to perform INT8 PTQ.\n"
                "Hailo DFC was not found in the environment.",
                code="E-HAILO-ENV-004",
            )

        quant_har_target = self.output_dir / "model_quantized.har"

        # 1. Python SDK mode
        if env.dfc_mode == "sdk" and self.config.hailo.sdk_mode in ("auto", "sdk"):
            try:
                from hailo_sdk_client import ClientRunner

                runner = ClientRunner(har=str(har_p))
                calib_data = np.load(str(calib_p))

                runner.optimize(calib_data)
                runner.save_har(str(quant_har_target))
            except Exception as e:
                raise HailoOptimisationError(
                    f"Hailo SDK quantisation failed: {e}",
                    code="E-HAILO-OPT-004",
                    artifact=str(har_p),
                ) from e

        # 2. CLI mode
        else:
            hailo_bin = shutil.which("hailo")
            if not hailo_bin:
                raise HailoEnvironmentError("Hailo CLI binary not found in PATH", code="E-HAILO-ENV-005")

            cmd = [
                hailo_bin,
                "optimize",
                str(har_p),
                "--calib-set-path",
                str(calib_p),
                "--output-har-path",
                str(quant_har_target),
            ]
            safe_run(cmd, error_code="E-HAILO-OPT-005", phase="hailo_optimisation")

        if not quant_har_target.is_file() or quant_har_target.stat().st_size == 0:
            raise HailoOptimisationError(
                f"Optimised HAR file generation failed: {quant_har_target}",
                code="E-HAILO-OPT-006",
                artifact=str(quant_har_target),
            )

        safe_symlink_or_copy(quant_har_target, self.canonical_dir / "model_quantized.har")
        return quant_har_target
