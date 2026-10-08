"""Hardware-in-the-loop tests for physical Hailo-8L AI HAT+ on Raspberry Pi 5.

These tests are marked with @pytest.mark.hailo and are automatically skipped
if physical Hailo accelerator hardware is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_hailo_mlops.hailo.environment import detect_hailo_environment
from yolo_hailo_mlops.hailo.runtime import HailoRuntimeValidator

env = detect_hailo_environment()
requires_hailo_hw = pytest.mark.skipif(
    not env.has_hardware or not env.has_hailort,
    reason="Physical Hailo-8L accelerator hardware (/dev/hailo0) and HailoRT are required for this test.",
)


@pytest.mark.hailo
@requires_hailo_hw
def test_physical_hailo_device_detection() -> None:
    """Verify that physical Hailo-8L device is detected and accessible."""
    hailo_env = detect_hailo_environment()
    assert hailo_env.has_hardware is True
    assert hailo_env.has_hailort is True
    assert hailo_env.device_name is not None


@pytest.mark.hailo
@requires_hailo_hw
def test_physical_hef_smoke_test(temp_workspace: Path) -> None:
    """Verify physical inference on Hailo-8L hardware when HEF exists."""
    hef_candidate = Path("artifacts/hailo/model.hef")
    if not hef_candidate.is_file():
        pytest.skip("No model.hef found in artifacts/hailo/ to run physical inference.")

    validator = HailoRuntimeValidator(require_hardware=True)
    report = validator.run_smoke_test(hef_candidate)

    assert report.executed is True
    assert report.status == "SUCCESS"
    assert report.inference_completed is True
    assert report.latency_ms is not None
    assert report.latency_ms > 0
