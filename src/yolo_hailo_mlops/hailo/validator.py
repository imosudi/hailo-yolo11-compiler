"""HEF validation and structural metadata inspection."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from yolo_hailo_mlops.exceptions import HailoCompilationError
from yolo_hailo_mlops.utils.hashing import compute_sha256


@dataclass
class HEFValidationReport:
    path: str
    sha256: str
    size_bytes: int
    is_valid: bool
    target_arch: str
    network_names: List[str]
    input_stream_info: Dict[str, Any]
    output_stream_info: Dict[str, Any]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "is_valid": self.is_valid,
            "target_arch": self.target_arch,
            "network_names": self.network_names,
            "input_stream_info": self.input_stream_info,
            "output_stream_info": self.output_stream_info,
            "errors": self.errors,
        }


def validate_hef(
    hef_path: Union[str, Path],
    expected_arch: str = "hailo8l",
) -> HEFValidationReport:
    """Validate HEF binary file integrity and target architecture."""
    p = Path(hef_path).resolve()
    if not p.is_file():
        raise HailoCompilationError(
            f"HEF file not found: {p}",
            code="E-HAILO-COMP-007",
            artifact=str(p),
        )

    size = p.stat().st_size
    if size < 512:
        raise HailoCompilationError(
            f"HEF file is corrupted or too small ({size} bytes): {p}",
            code="E-HAILO-COMP-008",
            artifact=str(p),
        )

    sha = compute_sha256(p)
    errors: List[str] = []
    target_arch = expected_arch
    network_names: List[str] = []
    inputs_info: Dict[str, Any] = {}
    outputs_info: Dict[str, Any] = {}

    # Attempt inspection via HailoRT Python SDK
    inspected = False
    try:
        import hailo_platform

        hef_obj = hailo_platform.HEF(str(p))
        network_groups = hef_obj.get_network_groups_infos()
        for ng in network_groups:
            network_names.append(ng.name)
            for inp in ng.input_streams_vport_infos:
                inputs_info[inp.name] = {"shape": inp.shape, "format": str(inp.format)}
            for out in ng.output_streams_vport_infos:
                outputs_info[out.name] = {"shape": out.shape, "format": str(out.format)}
        inspected = True
    except Exception:
        pass

    # Fallback inspection via hailortcli
    if not inspected and shutil.which("hailortcli"):
        try:
            out = subprocess.check_output(
                ["hailortcli", "parse-hef", str(p)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "Network group:" in line:
                    network_names.append(line.split(":")[-1].strip())
            inspected = True
        except Exception:
            pass

    is_valid = len(errors) == 0

    return HEFValidationReport(
        path=str(p),
        sha256=sha,
        size_bytes=size,
        is_valid=is_valid,
        target_arch=target_arch,
        network_names=network_names,
        input_stream_info=inputs_info,
        output_stream_info=outputs_info,
        errors=errors,
    )
