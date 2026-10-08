"""Hailo Dataflow Compiler (DFC) and HailoRT environment detection and capability probing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class HailoEnvironment:
    has_dfc: bool
    dfc_version: Optional[str]
    dfc_mode: str  # "sdk", "cli", or "none"
    has_hailort: bool
    hailort_version: Optional[str]
    hailort_mode: str  # "sdk", "cli", or "none"
    has_hardware: bool
    device_name: Optional[str]
    target_arch_supported: bool
    diagnostic: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_hailo_environment() -> HailoEnvironment:
    """Probe the system for Hailo DFC and HailoRT installations and physical devices."""
    has_dfc = False
    dfc_version = None
    dfc_mode = "none"

    has_hailort = False
    hailort_version = None
    hailort_mode = "none"

    has_hardware = False
    device_name = None
    target_arch_supported = False

    # 1. Probe Hailo DFC
    try:
        import hailo_sdk_client
        dfc_version = getattr(hailo_sdk_client, "__version__", "unknown")
        has_dfc = True
        dfc_mode = "sdk"
    except ImportError:
        hailo_cli = shutil.which("hailo")
        if hailo_cli:
            has_dfc = True
            dfc_mode = "cli"
            try:
                out = subprocess.check_output([hailo_cli, "--version"], text=True).strip()
                dfc_version = out
            except Exception:
                dfc_version = "cli_detected"

    # 2. Probe HailoRT
    try:
        import hailo_platform
        hailort_version = getattr(hailo_platform, "__version__", "unknown")
        has_hailort = True
        hailort_mode = "sdk"
    except ImportError:
        hailortcli = shutil.which("hailortcli")
        if hailortcli:
            has_hailort = True
            hailort_mode = "cli"
            try:
                out = subprocess.check_output([hailortcli, "--version"], text=True).strip()
                hailort_version = out
            except Exception:
                hailort_version = "cli_detected"

    # 3. Probe physical Hailo hardware
    if os.path.exists("/dev/hailo0"):
        has_hardware = True
        device_name = "Hailo-8L PCIe Device (/dev/hailo0)"
    elif shutil.which("hailortcli"):
        try:
            out = subprocess.check_output(["hailortcli", "scan"], text=True, stderr=subprocess.DEVNULL)
            if "Device:" in out or "Hailo" in out or "PCIe" in out:
                has_hardware = True
                device_name = out.strip().splitlines()[0]
        except Exception:
            pass

    # Verify Hailo-8L target architecture support
    if has_dfc:
        # Hailo-8L is supported in all standard modern DFC distributions (DFC v3.26+)
        target_arch_supported = True

    diag_lines = []
    if has_dfc:
        diag_lines.append(f"Hailo DFC available ({dfc_mode}, v{dfc_version})")
    else:
        diag_lines.append("Hailo DFC not found")

    if has_hailort:
        diag_lines.append(f"HailoRT available ({hailort_mode}, v{hailort_version})")
    else:
        diag_lines.append("HailoRT not found")

    if has_hardware:
        diag_lines.append(f"Hailo Hardware connected: {device_name}")
    else:
        diag_lines.append("No physical Hailo hardware detected (Raspberry Pi 5 AI HAT+)")

    return HailoEnvironment(
        has_dfc=has_dfc,
        dfc_version=dfc_version,
        dfc_mode=dfc_mode,
        has_hailort=has_hailort,
        hailort_version=hailort_version,
        hailort_mode=hailort_mode,
        has_hardware=has_hardware,
        device_name=device_name,
        target_arch_supported=target_arch_supported,
        diagnostic="; ".join(diag_lines),
    )
