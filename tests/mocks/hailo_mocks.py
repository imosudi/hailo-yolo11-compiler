"""Mocks for Hailo DFC and HailoRT enabling complete pipeline testing without hardware.

NOTE: These mocks are strictly for automated unit and integration tests of pipeline
orchestration logic. They must never be substituted for physical Hailo compilation
or physical runtime validation during real deployments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from yolo_hailo_mlops.hailo.environment import HailoEnvironment
from yolo_hailo_mlops.hailo.runtime import HailoRuntimeReport
from yolo_hailo_mlops.hailo.validator import HEFValidationReport
from yolo_hailo_mlops.utils.filesystem import atomic_write


class MockHailoClientRunner:
    """Mock for hailo_sdk_client.ClientRunner."""

    def __init__(self, hw_arch: str = "hailo8l", har: Optional[str] = None) -> None:
        self.hw_arch = hw_arch
        self.har_path = har
        self.translated = False
        self.optimized = False
        self.compiled = False
        self.model_script_loaded = False

    def translate_onnx_model(self, model: str, net_name: str, **kwargs: Any) -> None:
        self.translated = True

    def load_model_script(self, script_path: str) -> None:
        self.model_script_loaded = True

    def save_har(self, har_path: str) -> None:
        # Create a mock HAR binary
        atomic_write(har_path, b"HAILO_ARCHIVE_MOCK_PAYLOAD_V3", binary=True)

    def optimize(self, calib_data: np.ndarray) -> None:
        self.optimized = True

    def compile(self) -> bytes:
        self.compiled = True
        return b"HAILO8L_HEF_MOCK_BINARY_PAYLOAD_MAGIC_HEADER_V1"


def create_mock_hailo_environment(
    has_dfc: bool = True,
    has_hailort: bool = True,
    has_hardware: bool = True,
) -> HailoEnvironment:
    """Helper creating configurable HailoEnvironment for testing."""
    return HailoEnvironment(
        has_dfc=has_dfc,
        dfc_version="3.28.0" if has_dfc else None,
        dfc_mode="sdk" if has_dfc else "none",
        has_hailort=has_hailort,
        hailort_version="4.18.0" if has_hailort else None,
        hailort_mode="sdk" if has_hailort else "none",
        has_hardware=has_hardware,
        device_name="Mock Hailo-8L PCIe Device" if has_hardware else None,
        target_arch_supported=True,
        diagnostic="Mock environment",
    )


class MockHailoRuntimeValidator:
    """Mock for HailoRuntimeValidator."""

    def __init__(self, simulate_hardware: bool = True) -> None:
        self.simulate_hardware = simulate_hardware

    def run_smoke_test(
        self,
        hef_path: Path,
        test_image_array: Optional[np.ndarray] = None,
    ) -> HailoRuntimeReport:
        if not self.simulate_hardware:
            return HailoRuntimeReport(
                executed=False,
                status="NOT_EXECUTED",
                device_discovered=False,
                hef_loaded=False,
                network_activated=False,
                inference_completed=False,
                reason="Physical Hailo-8L hardware not present (Mock simulation)",
            )

        return HailoRuntimeReport(
            executed=True,
            status="SUCCESS",
            device_discovered=True,
            hef_loaded=True,
            network_activated=True,
            inference_completed=True,
            latency_ms=12.4,
            input_shape=(1, 640, 640, 3),
            output_shape=(1, 84, 8400),
        )
