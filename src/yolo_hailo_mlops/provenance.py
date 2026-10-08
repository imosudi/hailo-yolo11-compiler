"""System environment inspection, hardware telemetry, git tracking, and metric provenance."""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class TelemetryRecord:
    metric: str
    value: Any
    unit: str
    classification: str  # MEASURED | DERIVED | CONFIGURED | UNAVAILABLE
    source: str
    timestamp: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_provenance(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Capture git commit, branch, and working-tree dirty status."""
    cwd = repo_dir or os.getcwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = len(status_out) > 0
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "status": "available",
        }
    except Exception:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "status": "unavailable",
            "reason": "Not a git repository or git binary unavailable",
        }


def get_installed_packages() -> Dict[str, str]:
    """Return dictionary of installed package names and versions."""
    packages: Dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        packages[dist.metadata["Name"].lower()] = dist.version
    return dict(sorted(packages.items()))


def get_dependency_versions() -> Dict[str, Any]:
    """Inspect core ML and edge AI dependencies explicitly."""
    packages = get_installed_packages()

    def get_pkg_info(pkg_name: str) -> Dict[str, Any]:
        ver = packages.get(pkg_name.lower())
        if ver:
            return {"version": ver, "status": "available"}
        return {"version": None, "status": "unavailable", "reason": f"{pkg_name} not installed"}

    # Special check for hailo
    hailo_dfc = get_pkg_info("hailo-sdk-client")
    if hailo_dfc["status"] == "unavailable":
        # Check if CLI tool is available
        from shutil import which
        if which("hailo"):
            hailo_dfc = {"version": "cli_detected", "status": "available"}

    hailort = get_pkg_info("hailo-platform")
    if hailort["status"] == "unavailable":
        from shutil import which
        if which("hailortcli"):
            hailort = {"version": "cli_detected", "status": "available"}

    return {
        "python": sys.version.split()[0],
        "torch": get_pkg_info("torch"),
        "torchvision": get_pkg_info("torchvision"),
        "ultralytics": get_pkg_info("ultralytics"),
        "onnx": get_pkg_info("onnx"),
        "onnxruntime": get_pkg_info("onnxruntime"),
        "hailo_dfc": hailo_dfc,
        "hailort": hailort,
        "pyyaml": get_pkg_info("pyyaml"),
        "pillow": get_pkg_info("pillow"),
        "numpy": get_pkg_info("numpy"),
    }


def collect_hardware_telemetry() -> List[TelemetryRecord]:
    """Collect system telemetry with strict provenance classifications."""
    records: List[TelemetryRecord] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. CPU
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        records.append(
            TelemetryRecord(
                metric="cpu_utilization",
                value=cpu_pct,
                unit="percent",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="cpu_utilization",
                value=None,
                unit="percent",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
            )
        )

    # 2. Memory
    try:
        mem = psutil.virtual_memory()
        records.append(
            TelemetryRecord(
                metric="memory_total",
                value=mem.total,
                unit="bytes",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
            )
        )
        records.append(
            TelemetryRecord(
                metric="memory_available",
                value=mem.available,
                unit="bytes",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
            )
        )
        records.append(
            TelemetryRecord(
                metric="memory_used_percent",
                value=mem.percent,
                unit="percent",
                classification="DERIVED",
                source="psutil",
                timestamp=now,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="memory_usage",
                value=None,
                unit="bytes",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
            )
        )

    # 3. Storage
    try:
        disk = psutil.disk_usage(os.getcwd())
        records.append(
            TelemetryRecord(
                metric="disk_free",
                value=disk.free,
                unit="bytes",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
            )
        )
        records.append(
            TelemetryRecord(
                metric="disk_used_percent",
                value=disk.percent,
                unit="percent",
                classification="DERIVED",
                source="psutil",
                timestamp=now,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="disk_usage",
                value=None,
                unit="bytes",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
            )
        )

    # 4. SoC Thermal Zones (Linux / Raspberry Pi)
    thermal_read = False
    for tz_path in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/hwmon/hwmon0/temp1_input"):
        if os.path.exists(tz_path):
            try:
                with open(tz_path, "r") as f:
                    milli_c = float(f.read().strip())
                    deg_c = milli_c / 1000.0 if milli_c > 1000 else milli_c
                    records.append(
                        TelemetryRecord(
                            metric="soc_temperature",
                            value=round(deg_c, 2),
                            unit="degC",
                            classification="MEASURED",
                            source=tz_path,
                            timestamp=now,
                        )
                    )
                    thermal_read = True
                    break
            except Exception:
                pass

    if not thermal_read:
        records.append(
            TelemetryRecord(
                metric="soc_temperature",
                value=None,
                unit="degC",
                classification="UNAVAILABLE",
                source="thermal_zone",
                timestamp=now,
                reason="Linux thermal zone sysfs entry not found or unreadable",
            )
        )

    # 5. Raspberry Pi Throttling
    throttled_read = False
    from shutil import which
    if which("vcgencmd"):
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
            records.append(
                TelemetryRecord(
                    metric="rpi_throttled_state",
                    value=out,
                    unit="flags",
                    classification="MEASURED",
                    source="vcgencmd",
                    timestamp=now,
                )
            )
            throttled_read = True
        except Exception:
            pass

    if not throttled_read:
        records.append(
            TelemetryRecord(
                metric="rpi_throttled_state",
                value=None,
                unit="flags",
                classification="UNAVAILABLE",
                source="vcgencmd",
                timestamp=now,
                reason="vcgencmd tool not present on host",
            )
        )

    # 6. Hailo Accelerator Telemetry
    hailo_telemetry_found = False
    if which("hailortcli"):
        try:
            out = subprocess.check_output(
                ["hailortcli", "measure-power"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            records.append(
                TelemetryRecord(
                    metric="hailo_power",
                    value=out,
                    unit="W",
                    classification="MEASURED",
                    source="hailortcli",
                    timestamp=now,
                )
            )
            hailo_telemetry_found = True
        except Exception:
            pass

    if not hailo_telemetry_found:
        records.append(
            TelemetryRecord(
                metric="hailo_power",
                value=None,
                unit="W",
                classification="UNAVAILABLE",
                source="hailortcli",
                timestamp=now,
                reason="Hailo hardware power telemetry not exposed or hardware not connected",
            )
        )

    return records


def get_environment_info() -> Dict[str, Any]:
    """Capture full host platform and runtime environment details."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "dependencies": get_dependency_versions(),
        "git": get_git_provenance(),
    }
