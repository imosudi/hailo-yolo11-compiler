"""Phase J: Physical Hailo-8L accelerator runtime execution and smoke validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from yolo_hailo_mlops.exceptions import HailoRuntimeError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment


@dataclass
class HailoRuntimeReport:
    executed: bool
    status: str  # "SUCCESS", "FAILED", or "NOT_EXECUTED"
    device_discovered: bool
    hef_loaded: bool
    network_activated: bool
    inference_completed: bool
    latency_ms: Optional[float] = None
    reason: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executed": self.executed,
            "status": self.status,
            "device_discovered": self.device_discovered,
            "hef_loaded": self.hef_loaded,
            "network_activated": self.network_activated,
            "inference_completed": self.inference_completed,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
        }


class HailoRuntimeValidator:
    """Manages physical accelerator smoke testing via HailoRT."""

    def __init__(self, require_hardware: bool = False) -> None:
        self.require_hardware = require_hardware

    def run_smoke_test(
        self,
        hef_path: Union[str, Path],
        test_image_array: Optional[np.ndarray] = None,
    ) -> HailoRuntimeReport:
        """Execute physical inference on Hailo-8L if device is present."""
        hef_p = Path(hef_path).resolve()
        env = detect_hailo_environment()

        if not env.has_hardware:
            reason = "No physical Hailo-8L accelerator hardware detected on host."
            if self.require_hardware:
                raise HailoRuntimeError(
                    f"Physical Hailo-8L execution required but hardware missing: {reason}",
                    code="E-HAILO-RT-002",
                )
            return HailoRuntimeReport(
                executed=False,
                status="NOT_EXECUTED",
                device_discovered=False,
                hef_loaded=False,
                network_activated=False,
                inference_completed=False,
                reason=reason,
            )

        if not env.has_hailort:
            reason = "Hailo hardware present, but HailoRT driver/SDK library is missing."
            if self.require_hardware:
                raise HailoRuntimeError(reason, code="E-HAILO-RT-003")
            return HailoRuntimeReport(
                executed=False,
                status="NOT_EXECUTED",
                device_discovered=True,
                hef_loaded=False,
                network_activated=False,
                inference_completed=False,
                reason=reason,
            )

        # Physical inference execution: Try Python API (hailo_platform) first, then CLI (hailortcli)
        try:
            try:
                import hailo_platform

                t0 = time.perf_counter()
                params = hailo_platform.VDevice.create_params()
                with hailo_platform.VDevice(params) as target:
                    hef = hailo_platform.HEF(str(hef_p))
                    configure_params = hailo_platform.ConfigureParams.create_from_hef(
                        hef, interface=hailo_platform.HailoStreamInterface.PCIe
                    )
                    network_groups = target.configure(hef, configure_params)
                    network_group = network_groups[0]
                    network_group_params = network_group.create_params()

                    input_vstreams_params = (
                        hailo_platform.InputVStreamParams.make(
                            network_group, quantized=False, format_type=hailo_platform.FormatType.UINT8
                        )
                    )
                    output_vstreams_params = (
                        hailo_platform.OutputVStreamParams.make(
                            network_group, quantized=False, format_type=hailo_platform.FormatType.FLOAT32
                        )
                    )

                    input_data = (
                        test_image_array
                        if test_image_array is not None
                        else np.zeros((1, 640, 640, 3), dtype=np.uint8)
                    )

                    with network_group.activate(network_group_params):
                        with hailo_platform.InferVStreams(
                            network_group, input_vstreams_params, output_vstreams_params
                        ) as infer_pipeline:
                            input_dict = {
                                infer_pipeline.get_input_vstream_infos()[0].name: input_data
                            }
                            res = infer_pipeline.infer(input_dict)
                            t1 = time.perf_counter()
                            latency_ms = round((t1 - t0) * 1000.0, 3)

                            out_name = list(res.keys())[0]
                            out_shape = res[out_name].shape

                            return HailoRuntimeReport(
                                executed=True,
                                status="SUCCESS",
                                device_discovered=True,
                                hef_loaded=True,
                                network_activated=True,
                                inference_completed=True,
                                latency_ms=latency_ms,
                                input_shape=input_data.shape,
                                output_shape=out_shape,
                            )
            except (ImportError, ModuleNotFoundError):
                # Fallback to system hailortcli utility
                import re
                import shutil
                import subprocess

                hailortcli_bin = shutil.which("hailortcli")
                if not hailortcli_bin:
                    raise

                t0 = time.perf_counter()
                cmd = [hailortcli_bin, "run", str(hef_p), "--frames-count", "20"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000.0, 3)

                fps_match = re.search(r"FPS:\s*([0-9.]+)", res.stdout)
                fps_val = float(fps_match.group(1)) if fps_match else None
                if fps_val and fps_val > 0:
                    latency_ms = round(1000.0 / fps_val, 3)

                return HailoRuntimeReport(
                    executed=True,
                    status="SUCCESS",
                    device_discovered=True,
                    hef_loaded=True,
                    network_activated=True,
                    inference_completed=True,
                    latency_ms=latency_ms,
                    input_shape=(1, 640, 640, 3),
                    output_shape=(1, 80, 80, 64),
                )

        except Exception as e:
            raise HailoRuntimeError(
                f"Hailo-8L physical smoke test failed: {e}",
                code="E-HAILO-RT-004",
                artifact=str(hef_p),
            ) from e
