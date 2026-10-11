"""System environment inspection, hardware telemetry, git tracking, and metric provenance."""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Tuple

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

    # 5. Raspberry Pi Throttling & PMIC Power
    throttled_read = False
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

        # RPi 5 Renesas DA9091 PMIC Power Measurement
        pmic_w, sys_w, _ = read_rpi_pmic_power()
        if pmic_w is not None:
            records.append(
                TelemetryRecord(
                    metric="rpi_pmic_power",
                    value=pmic_w,
                    unit="W",
                    classification="MEASURED",
                    source="vcgencmd pmic_read_adc",
                    timestamp=now,
                )
            )
        if sys_w is not None:
            records.append(
                TelemetryRecord(
                    metric="rpi_system_power",
                    value=sys_w,
                    unit="W",
                    classification="DERIVED",
                    source="pmic_model",
                    timestamp=now,
                )
            )

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

    # 6. Hailo Accelerator Power Telemetry
    hailo_w, h_class, h_source, h_reason = measure_hailo_power()
    records.append(
        TelemetryRecord(
            metric="hailo_power",
            value=hailo_w,
            unit="W",
            classification=h_class,
            source=h_source,
            timestamp=now,
            reason=h_reason,
        )
    )

    return records


def read_rpi_pmic_power() -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    """Read Raspberry Pi 5 Renesas DA9091 PMIC rail telemetry via vcgencmd.

    Returns:
        (total_pmic_watts, estimated_system_watts, per_rail_watts)
    """
    if not which("vcgencmd"):
        return None, None, {}
    try:
        raw = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"], text=True, stderr=subprocess.DEVNULL
        )
        currents: Dict[str, float] = {}
        volts: Dict[str, float] = {}
        for line in raw.strip().splitlines():
            line = line.strip()
            m_curr = re.match(r"^([A-Za-z0-9_]+)_A\s+current\(\d+\)=([0-9.]+)A", line)
            if m_curr:
                currents[m_curr.group(1)] = float(m_curr.group(2))
                continue
            m_volt = re.match(r"^([A-Za-z0-9_]+)_V\s+volt\(\d+\)=([0-9.]+)V", line)
            if m_volt:
                volts[m_volt.group(1)] = float(m_volt.group(2))
                continue

        rail_powers: Dict[str, float] = {}
        total_pmic = 0.0
        for rail, i_val in currents.items():
            if rail in volts:
                p = i_val * volts[rail]
                rail_powers[rail] = round(p, 4)
                total_pmic += p

        if total_pmic > 0:
            est_system = (total_pmic * 1.1451) + 0.5879
            return round(total_pmic, 3), round(est_system, 3), rail_powers
    except Exception:
        pass
    return None, None, {}


def measure_hailo_power() -> Tuple[Optional[float], str, str, Optional[str]]:
    """Attempt physical or estimated power measurement from Hailo accelerator.

    Returns:
        (power_in_watts, classification, source, reason)
    """
    hailortcli_bin = which("hailortcli")

    if hailortcli_bin:
        # 1. Try standalone measure-power command
        try:
            proc = subprocess.run(
                [hailortcli_bin, "measure-power", "--duration", "1"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            combined = f"{proc.stdout} {proc.stderr}"
            if proc.returncode == 0:
                m = re.search(
                    r"(?:average|Average|\(average\)|Power|power)[^0-9]*([0-9]+\.[0-9]+)\s*W",
                    proc.stdout,
                )
                if not m:
                    m = re.search(r"([0-9]+\.[0-9]+)\s*W", proc.stdout)
                if m:
                    return round(float(m.group(1)), 3), "MEASURED", "hailortcli measure-power", None

            if (
                "unsupported" in combined.lower()
                or "unsupported_opcode" in combined.lower()
                or "unsupported_device" in combined.lower()
            ):
                return (
                    1.5,
                    "ESTIMATED",
                    "hailo8l_datasheet",
                    "Raspberry Pi AI HAT+ Hailo-8L omits onboard INA power sensor; nominal active power estimated at 1.5W.",
                )
        except Exception:
            pass

        # 2. Check if a compiled HEF exists to test live power measurement under inference
        canonical_hef = Path("artifacts/hailo/model.hef")
        if canonical_hef.is_file():
            try:
                proc = subprocess.run(
                    [
                        hailortcli_bin,
                        "run",
                        str(canonical_hef),
                        "--frames-count",
                        "20",
                        "--measure-power",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                combined = f"{proc.stdout} {proc.stderr}"
                if proc.returncode == 0:
                    m = re.search(
                        r"(?:average|Average|\(average\))\s*=\s*([0-9.]+)\s*W",
                        proc.stdout,
                    )
                    if not m:
                        m = re.search(r"([0-9]+\.[0-9]+)\s*W", proc.stdout)
                    if m:
                        return (
                            round(float(m.group(1)), 3),
                            "MEASURED",
                            "hailortcli run --measure-power",
                            None,
                        )
                if (
                    "unsupported" in combined.lower()
                    or "unsupported_opcode" in combined.lower()
                ):
                    return (
                        1.5,
                        "ESTIMATED",
                        "hailo8l_datasheet",
                        "Raspberry Pi AI HAT+ Hailo-8L omits onboard INA power sensor; nominal active power estimated at 1.5W.",
                    )
            except Exception:
                pass

    # 3. If Hailo hardware device file exists (/dev/hailo0)
    if os.path.exists("/dev/hailo0"):
        return (
            1.5,
            "ESTIMATED",
            "hailo8l_datasheet",
            "Physical Hailo-8L accelerator connected; board omits INA current monitor; nominal power reported.",
        )

    return (
        None,
        "UNAVAILABLE",
        "hailortcli",
        "Hailo hardware power telemetry not exposed or hardware not connected",
    )


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
