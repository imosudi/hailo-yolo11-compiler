"""System environment inspection, hardware telemetry, git tracking, and metric provenance."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
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
    backend: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None  # VALID | UNAVAILABLE
    run_id: Optional[str] = None
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("status") is None:
            d["status"] = (
                "VALID"
                if d.get("value") is not None and d.get("classification") != "UNAVAILABLE"
                else "UNAVAILABLE"
            )
        return d


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


def ensure_system_dist_packages() -> None:
    """Ensure Debian/Raspberry Pi OS system dist-packages is linked and in sys.path."""
    for dist_dir in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
        if os.path.isdir(dist_dir) and dist_dir not in sys.path:
            sys.path.append(dist_dir)

    if sys.prefix != sys.base_prefix:
        try:
            site_pkgs = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            if site_pkgs.is_dir():
                pth_file = site_pkgs / "_system_dist_packages.pth"
                if not pth_file.exists():
                    pth_file.write_text("/usr/lib/python3/dist-packages\n/usr/local/lib/python3/dist-packages\n", encoding="utf-8")
        except Exception:
            pass

        try:
            cfg_path = Path(sys.prefix) / "pyvenv.cfg"
            if cfg_path.is_file():
                txt = cfg_path.read_text(encoding="utf-8")
                if "include-system-site-packages = false" in txt:
                    cfg_path.write_text(
                        txt.replace("include-system-site-packages = false", "include-system-site-packages = true"),
                        encoding="utf-8",
                    )
        except Exception:
            pass


ensure_system_dist_packages()


def get_dependency_versions() -> Dict[str, Any]:
    """Inspect core ML and edge AI dependencies explicitly."""
    ensure_system_dist_packages()
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

    # OpenCV check
    cv2_info = get_pkg_info("opencv-python")
    if cv2_info["status"] == "unavailable":
        try:
            import cv2
            cv2_info = {"version": getattr(cv2, "__version__", "installed"), "status": "available"}
        except ImportError:
            pass

    # Picamera2 / native libcamera check (virtualenv, system dist-packages, or rpicam CLI)
    picam2_info = {"version": None, "status": "unavailable", "reason": "Neither picamera2 nor rpicam-apps installed"}
    try:
        from yolo_hailo_mlops.deployment.sources import try_import_picamera2
        picam2_cls = try_import_picamera2()
        if picam2_cls is not None:
            picam2_info = {"version": "picamera2", "status": "available"}
        elif shutil.which("rpicam-vid") or shutil.which("libcamera-vid"):
            picam2_info = {"version": "rpicam-vid (native libcamera)", "status": "available"}
    except Exception:
        if shutil.which("rpicam-vid") or shutil.which("libcamera-vid"):
            picam2_info = {"version": "rpicam-vid (native libcamera)", "status": "available"}

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
        "opencv": cv2_info,
        "picamera2": picam2_info,
    }


def read_rpi_throttling() -> Tuple[Optional[str], Optional[bool], Optional[bool], Optional[str]]:
    """Parse Raspberry Pi throttling bitmask via vcgencmd.

    Returns:
        (raw_bitmask_str, undervoltage_now, is_throttled_now, error_reason)
    """
    if not which("vcgencmd"):
        return None, None, None, "vcgencmd tool not present on host"
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        m = re.search(r"throttled=(0x[0-9a-fA-F]+|[0-9]+)", out)
        if m:
            raw_str = m.group(1)
            raw_int = int(raw_str, 16) if raw_str.startswith("0x") else int(raw_str)
            undervolt_now = bool(raw_int & 0x1)
            is_throttled_now = bool(raw_int & 0xF)
            return raw_str, undervolt_now, is_throttled_now, None
    except Exception as e:
        return None, None, None, str(e)
    return None, None, None, "Failed to parse vcgencmd get_throttled output"


def detect_hailo_device_presence() -> Tuple[bool, str]:
    """Detect presence of Hailo hardware accelerator."""
    if Path("/dev/hailo0").exists():
        return True, "/dev/hailo0"
    for pci_path in Path("/sys/bus/pci/devices").glob("*"):
        try:
            vendor = (pci_path / "vendor").read_text().strip().lower()
            if vendor in ("0x1e60", "1e60"):
                return True, f"pci:{pci_path.name}"
        except Exception:
            pass
    if which("hailortcli"):
        try:
            res = subprocess.run(["hailortcli", "scan"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and "Hailo" in res.stdout:
                return True, "hailortcli scan"
        except Exception:
            pass
    return False, "sysfs_and_dev_probe"


def read_hailo_temperature() -> Tuple[Optional[float], str, str, Optional[str]]:
    """Read Hailo device temperature if exposed by driver/runtime."""
    for hw_dir in Path("/sys/class/hwmon").glob("hwmon*"):
        name_f = hw_dir / "name"
        if name_f.is_file():
            try:
                name = name_f.read_text().strip().lower()
                if "hailo" in name:
                    temp_f = hw_dir / "temp1_input"
                    if temp_f.is_file():
                        mc = float(temp_f.read_text().strip())
                        return round(mc / 1000.0, 2), "MEASURED", str(temp_f), None
            except Exception:
                pass
    return None, "UNAVAILABLE", "hailo_driver", "Hailo device driver or hardware interface does not expose on-chip temperature sensor"


def read_hailo_power_watts() -> Tuple[Optional[float], str, str, Optional[str]]:
    """Directly measure Hailo power if supported by device interface."""
    hailortcli_bin = which("hailortcli")
    if hailortcli_bin:
        try:
            proc = subprocess.run(
                [hailortcli_bin, "measure-power", "--duration", "1"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if proc.returncode == 0:
                m = re.search(
                    r"(?:average|Average|\(average\)|Power|power)[^0-9]*([0-9]+\.[0-9]+)\s*W",
                    proc.stdout,
                )
                if not m:
                    m = re.search(r"([0-9]+\.[0-9]+)\s*W", proc.stdout)
                if m:
                    return round(float(m.group(1)), 3), "MEASURED", "hailortcli measure-power", None
        except Exception:
            pass
    return None, "UNAVAILABLE", "hailortcli", "Hailo hardware interface does not support onboard power measurement (INA sensor not present on Raspberry Pi AI HAT+)"


def read_lm_sensors_telemetry(
    run_id: Optional[str] = None,
    backend: str = "cpu",
    phase: str = "performance_validation",
    device: Optional[str] = None,
) -> List[TelemetryRecord]:
    """Parse structured JSON from lm-sensors 'sensors -j' if installed."""
    records: List[TelemetryRecord] = []
    sensors_bin = which("sensors")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dev_name = device or f"{platform.system()} {platform.machine()}"

    if not sensors_bin:
        records.append(
            TelemetryRecord(
                metric="lm_sensors_installed",
                value=False,
                unit="boolean",
                classification="MEASURED",
                source="which('sensors')",
                timestamp=now,
                reason="lm-sensors package not installed (run 'sudo apt-get install -y lm-sensors')",
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        return records

    records.append(
        TelemetryRecord(
            metric="lm_sensors_installed",
            value=True,
            unit="boolean",
            classification="MEASURED",
            source=sensors_bin,
            timestamp=now,
            backend=backend,
            phase=phase,
            status="VALID",
            run_id=run_id,
            device=dev_name,
        )
    )

    try:
        proc = subprocess.run([sensors_bin, "-j"], capture_output=True, text=True, timeout=3)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            for chip, sub in data.items():
                chip_clean = re.sub(r"[^a-zA-Z0-9_]", "_", chip.split("-")[0].lower())
                adapter = sub.get("Adapter", "")
                for feat, sub_f in sub.items():
                    if not isinstance(sub_f, dict):
                        continue
                    feat_clean = re.sub(r"[^a-zA-Z0-9_]", "_", feat.lower()).strip("_")
                    for k, v in sub_f.items():
                        if isinstance(v, (int, float)):
                            if k.endswith("_input"):
                                base = k.replace("_input", "")
                                if "temp" in base:
                                    m_name = f"sensors_{chip_clean}_{feat_clean}_temp"
                                    u = "°C"
                                    val = round(float(v), 2)
                                elif "fan" in base:
                                    m_name = f"sensors_{chip_clean}_{feat_clean}_rpm"
                                    u = "RPM"
                                    val = int(v)
                                elif "in" in base:
                                    m_name = f"sensors_{chip_clean}_{feat_clean}_voltage_v"
                                    u = "V"
                                    val = round(float(v), 3)
                                elif "curr" in base:
                                    m_name = f"sensors_{chip_clean}_{feat_clean}_current_a"
                                    u = "A"
                                    val = round(float(v), 3)
                                elif "power" in base:
                                    m_name = f"sensors_{chip_clean}_{feat_clean}_power_w"
                                    u = "W"
                                    val = round(float(v), 3)
                                else:
                                    continue
                                records.append(
                                    TelemetryRecord(
                                        metric=m_name,
                                        value=val,
                                        unit=u,
                                        classification="MEASURED",
                                        source=f"sensors -j ({chip} / {adapter})",
                                        timestamp=now,
                                        backend=backend,
                                        phase=phase,
                                        status="VALID",
                                        run_id=run_id,
                                        device=dev_name,
                                    )
                                )
    except Exception:
        pass

    return records


def collect_hardware_telemetry(
    run_id: Optional[str] = None,
    backend: str = "cpu",
    phase: str = "performance_validation",
    device: Optional[str] = None,
) -> List[TelemetryRecord]:
    """Collect system telemetry including all canonical identifiers with strict provenance."""
    records: List[TelemetryRecord] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dev_name = device or f"{platform.system()} {platform.machine()}"

    # 1. Host CPU Utilisation (Canonical: cpu_utilisation)
    try:
        cpu_pct = round(psutil.cpu_percent(interval=0.1), 2) if psutil else None
        records.append(
            TelemetryRecord(
                metric="cpu_utilisation",
                value=cpu_pct,
                unit="%",
                classification="MEASURED",
                source="psutil.cpu_percent",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID" if cpu_pct is not None else "UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="cpu_utilization",
                value=cpu_pct,
                unit="percent",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID" if cpu_pct is not None else "UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="cpu_utilisation",
                value=None,
                unit="%",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 2. CPU Core Frequencies (Canonical: cpu_freq_cpu*_mhz)
    try:
        core_freqs = psutil.cpu_freq(percpu=True) if psutil else None
        if core_freqs:
            for idx, f_info in enumerate(core_freqs):
                records.append(
                    TelemetryRecord(
                        metric=f"cpu_freq_cpu{idx}_mhz",
                        value=round(f_info.current, 1) if f_info.current else None,
                        unit="MHz",
                        classification="MEASURED",
                        source=f"psutil.cpu_freq(percpu=True)[{idx}]",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID" if f_info.current else "UNAVAILABLE",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
        else:
            cpu_dirs = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
            found_sysfs = False
            for c_dir in cpu_dirs:
                freq_f = c_dir / "cpufreq" / "scaling_cur_freq"
                if freq_f.is_file():
                    try:
                        khz = int(freq_f.read_text().strip())
                        mhz = round(khz / 1000.0, 1)
                        core_idx = c_dir.name.replace("cpu", "")
                        records.append(
                            TelemetryRecord(
                                metric=f"cpu_freq_cpu{core_idx}_mhz",
                                value=mhz,
                                unit="MHz",
                                classification="MEASURED",
                                source=str(freq_f),
                                timestamp=now,
                                backend=backend,
                                phase=phase,
                                status="VALID",
                                run_id=run_id,
                                device=dev_name,
                            )
                        )
                        found_sysfs = True
                    except Exception:
                        pass
            if not found_sysfs:
                records.append(
                    TelemetryRecord(
                        metric="cpu_freq_cpu0_mhz",
                        value=None,
                        unit="MHz",
                        classification="UNAVAILABLE",
                        source="sysfs/psutil",
                        timestamp=now,
                        reason="CPU core frequency interface not accessible",
                        backend=backend,
                        phase=phase,
                        status="UNAVAILABLE",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="cpu_freq_cpu0_mhz",
                value=None,
                unit="MHz",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 3. Host Memory Utilisation & Available (Canonical: memory_utilisation, memory_available_mb)
    try:
        mem = psutil.virtual_memory()
        records.append(
            TelemetryRecord(
                metric="memory_utilisation",
                value=round(mem.percent, 2),
                unit="%",
                classification="DERIVED",
                source="psutil.virtual_memory().percent",
                timestamp=now,
                reason="Calculated as (total - available) / total * 100; available memory excludes reclaimable page cache and buffers.",
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        avail_mb = round(mem.available / (1024.0 * 1024.0), 2)
        records.append(
            TelemetryRecord(
                metric="memory_available_mb",
                value=avail_mb,
                unit="MB",
                classification="MEASURED",
                source="psutil.virtual_memory().available",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="memory_total",
                value=mem.total,
                unit="bytes",
                classification="MEASURED",
                source="psutil",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
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
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="memory_used_percent",
                value=round(mem.percent, 2),
                unit="percent",
                classification="DERIVED",
                source="psutil",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="memory_utilisation",
                value=None,
                unit="%",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="memory_available_mb",
                value=None,
                unit="MB",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 4. Process Memory RSS & Container Cgroup Memory
    try:
        proc_rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        proc_rss_mb = round(proc_rss_bytes / (1024.0 * 1024.0), 2)
        records.append(
            TelemetryRecord(
                metric="process_memory_rss_mb",
                value=proc_rss_mb,
                unit="MB",
                classification="MEASURED",
                source="psutil.Process().memory_info().rss",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="process_memory_rss_mb",
                value=None,
                unit="MB",
                classification="UNAVAILABLE",
                source="psutil.Process()",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # Container cgroup memory
    cgroup_read = False
    for cg_path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        if cg_path.is_file():
            try:
                cg_bytes = int(cg_path.read_text().strip())
                cg_mb = round(cg_bytes / (1024.0 * 1024.0), 2)
                records.append(
                    TelemetryRecord(
                        metric="container_cgroup_memory_mb",
                        value=cg_mb,
                        unit="MB",
                        classification="MEASURED",
                        source=str(cg_path),
                        timestamp=now,
                        reason="Current memory charged to benchmark container cgroup",
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
                cgroup_read = True
                break
            except Exception:
                pass
    if not cgroup_read:
        records.append(
            TelemetryRecord(
                metric="container_cgroup_memory_mb",
                value=None,
                unit="MB",
                classification="UNAVAILABLE",
                source="/sys/fs/cgroup",
                timestamp=now,
                reason="Process running outside container or cgroup memory interface unmounted",
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 5. Storage Metrics
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
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
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
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
    except Exception as e:
        records.append(
            TelemetryRecord(
                metric="disk_free",
                value=None,
                unit="bytes",
                classification="UNAVAILABLE",
                source="psutil",
                timestamp=now,
                reason=str(e),
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 6. Thermal Zones (Canonical: thermal_*_temp)
    thermal_read = False
    for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        t_f = tz / "temp"
        type_f = tz / "type"
        st = type_f.read_text().strip() if type_f.is_file() else "thermal"
        if t_f.is_file():
            try:
                mc = float(t_f.read_text().strip())
                val = round(mc / 1000.0, 2) if mc > 1000 else round(mc, 2)
                records.append(
                    TelemetryRecord(
                        metric=f"thermal_{tz.name}_temp",
                        value=val,
                        unit="°C",
                        classification="MEASURED",
                        source=f"{t_f} ({st})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
                if not thermal_read:
                    records.append(
                        TelemetryRecord(
                            metric="soc_temperature",
                            value=val,
                            unit="degC",
                            classification="MEASURED",
                            source=str(t_f),
                            timestamp=now,
                            backend=backend,
                            phase=phase,
                            status="VALID",
                            run_id=run_id,
                            device=dev_name,
                        )
                    )
                thermal_read = True
            except Exception:
                pass
    if not thermal_read:
        records.append(
            TelemetryRecord(
                metric="thermal_zone0_temp",
                value=None,
                unit="°C",
                classification="UNAVAILABLE",
                source="thermal_zone",
                timestamp=now,
                reason="Linux thermal zone sysfs entry not found or unreadable",
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="soc_temperature",
                value=None,
                unit="degC",
                classification="UNAVAILABLE",
                source="thermal_zone",
                timestamp=now,
                reason="Linux thermal zone sysfs entry not found or unreadable",
                backend=backend,
                phase=phase,
                status="UNAVAILABLE",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 7. Hardware-Monitoring Sensors (Canonical: hwmon_*_temp, hwmon_*_rpm)
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        nm = (hw / "name").read_text().strip() if (hw / "name").is_file() else hw.name
        for ti in sorted(hw.glob("temp*_input")):
            ch = ti.stem.replace("_input", "")
            try:
                mc = float(ti.read_text().strip())
                val = round(mc / 1000.0, 2) if mc > 1000 else round(mc, 2)
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_temp",
                        value=val,
                        unit="°C",
                        classification="MEASURED",
                        source=f"{ti} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass
        for fi in sorted(hw.glob("fan*_input")):
            ch = fi.stem.replace("_input", "")
            try:
                rpm = int(fi.read_text().strip())
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_rpm",
                        value=rpm,
                        unit="RPM",
                        classification="MEASURED",
                        source=f"{fi} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass
        for vi in sorted(hw.glob("in*_input")):
            ch = vi.stem.replace("_input", "")
            try:
                mv = float(vi.read_text().strip())
                val = round(mv / 1000.0, 3)
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_voltage_v",
                        value=val,
                        unit="V",
                        classification="MEASURED",
                        source=f"{vi} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass
        for ci in sorted(hw.glob("curr*_input")):
            ch = ci.stem.replace("_input", "")
            try:
                ma = float(ci.read_text().strip())
                val = round(ma / 1000.0, 3)
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_current_a",
                        value=val,
                        unit="A",
                        classification="MEASURED",
                        source=f"{ci} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass
        for pi in sorted(hw.glob("power*_input")):
            ch = pi.stem.replace("_input", "")
            try:
                uw = float(pi.read_text().strip())
                val = round(uw / 1000000.0, 3)
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_power_w",
                        value=val,
                        unit="W",
                        classification="MEASURED",
                        source=f"{pi} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass
        for tc in sorted(hw.glob("temp*_crit")):
            ch = tc.stem.replace("_crit", "")
            try:
                mc = float(tc.read_text().strip())
                val = round(mc / 1000.0, 2) if mc > 1000 else round(mc, 2)
                records.append(
                    TelemetryRecord(
                        metric=f"hwmon_{nm}_{ch}_temp_crit",
                        value=val,
                        unit="°C",
                        classification="MEASURED",
                        source=f"{tc} ({nm})",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            except Exception:
                pass

    # 8. Raspberry Pi Throttling (Canonical: rpi_throttled_raw, rpi_undervoltage_now, rpi_is_throttled)
    raw_throttled, under_now, throttled_now, err_throttle = read_rpi_throttling()
    if raw_throttled is not None:
        records.append(
            TelemetryRecord(
                metric="rpi_throttled_raw",
                value=raw_throttled,
                unit="hex",
                classification="MEASURED",
                source="vcgencmd get_throttled",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="rpi_undervoltage_now",
                value=under_now,
                unit="boolean",
                classification="DERIVED",
                source="vcgencmd_bit0",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="rpi_is_throttled",
                value=throttled_now,
                unit="boolean",
                classification="DERIVED",
                source="vcgencmd_active_bits_0_3",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
        records.append(
            TelemetryRecord(
                metric="rpi_throttled_state",
                value=f"throttled={raw_throttled}",
                unit="flags",
                classification="MEASURED",
                source="vcgencmd",
                timestamp=now,
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )
    else:
        for m_name, u in [
            ("rpi_throttled_raw", "hex"),
            ("rpi_undervoltage_now", "boolean"),
            ("rpi_is_throttled", "boolean"),
            ("rpi_throttled_state", "flags"),
        ]:
            records.append(
                TelemetryRecord(
                    metric=m_name,
                    value=None,
                    unit=u,
                    classification="UNAVAILABLE",
                    source="vcgencmd",
                    timestamp=now,
                    reason=err_throttle or "vcgencmd tool not present on host",
                    backend=backend,
                    phase=phase,
                    status="UNAVAILABLE",
                    run_id=run_id,
                    device=dev_name,
                )
            )

    # RPi 5 PMIC Power
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
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
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
                backend=backend,
                phase=phase,
                status="VALID",
                run_id=run_id,
                device=dev_name,
            )
        )

    # 9. Hailo Accelerator Identifiers (Canonical: hailo_device_present, hailo_temperature_c, hailo_power_watts)
    dev_present, dev_source = detect_hailo_device_presence()
    records.append(
        TelemetryRecord(
            metric="hailo_device_present",
            value=dev_present,
            unit="boolean",
            classification="MEASURED",
            source=dev_source,
            timestamp=now,
            backend="hailo8l" if dev_present else backend,
            phase=phase,
            status="VALID",
            run_id=run_id,
            device=dev_name,
        )
    )

    h_temp, t_class, t_source, t_reason = read_hailo_temperature()
    records.append(
        TelemetryRecord(
            metric="hailo_temperature_c",
            value=h_temp,
            unit="°C",
            classification=t_class,
            source=t_source,
            timestamp=now,
            reason=t_reason,
            backend="hailo8l",
            phase=phase,
            status="VALID" if h_temp is not None else "UNAVAILABLE",
            run_id=run_id,
            device=dev_name,
        )
    )

    h_pow_w, p_class, p_source, p_reason = read_hailo_power_watts()
    records.append(
        TelemetryRecord(
            metric="hailo_power_watts",
            value=h_pow_w,
            unit="W",
            classification=p_class,
            source=p_source,
            timestamp=now,
            reason=p_reason,
            backend="hailo8l",
            phase=phase,
            status="VALID" if h_pow_w is not None else "UNAVAILABLE",
            run_id=run_id,
            device=dev_name,
        )
    )

    # Broader hailo_power metric
    records.append(
        TelemetryRecord(
            metric="hailo_power",
            value=h_pow_w,
            unit="W",
            classification=p_class,
            source=p_source,
            timestamp=now,
            reason=p_reason,
            backend="hailo8l",
            phase=phase,
            status="VALID" if h_pow_w is not None else "UNAVAILABLE",
            run_id=run_id,
            device=dev_name,
        )
    )

    # 10. LM-Sensors Extended Telemetry (JSON-based sensor telemetry across all chips)
    records.extend(
        read_lm_sensors_telemetry(
            run_id=run_id,
            backend=backend,
            phase=phase,
            device=dev_name,
        )
    )

    # 11. Host System Load Averages
    try:
        load1, load5, load15 = os.getloadavg()
        for idx, (m_tag, l_val) in enumerate([("1m", load1), ("5m", load5), ("15m", load15)]):
            records.append(
                TelemetryRecord(
                    metric=f"load_average_{m_tag}",
                    value=round(l_val, 2),
                    unit="load",
                    classification="MEASURED",
                    source=f"os.getloadavg()[{idx}]",
                    timestamp=now,
                    backend=backend,
                    phase=phase,
                    status="VALID",
                    run_id=run_id,
                    device=dev_name,
                )
            )
    except Exception:
        pass

    # 12. Swap Memory Metrics
    try:
        if psutil:
            sw = psutil.swap_memory()
            records.append(
                TelemetryRecord(
                    metric="swap_utilisation",
                    value=round(sw.percent, 2),
                    unit="%",
                    classification="MEASURED",
                    source="psutil.swap_memory().percent",
                    timestamp=now,
                    backend=backend,
                    phase=phase,
                    status="VALID",
                    run_id=run_id,
                    device=dev_name,
                )
            )
            records.append(
                TelemetryRecord(
                    metric="swap_used_mb",
                    value=round(sw.used / (1024.0 * 1024.0), 2),
                    unit="MB",
                    classification="MEASURED",
                    source="psutil.swap_memory().used",
                    timestamp=now,
                    backend=backend,
                    phase=phase,
                    status="VALID",
                    run_id=run_id,
                    device=dev_name,
                )
            )
    except Exception:
        pass

    # 13. Disk I/O & Network I/O Telemetry
    try:
        if psutil:
            dio = psutil.disk_io_counters()
            if dio:
                records.append(
                    TelemetryRecord(
                        metric="disk_read_mb",
                        value=round(dio.read_bytes / (1024.0 * 1024.0), 2),
                        unit="MB",
                        classification="MEASURED",
                        source="psutil.disk_io_counters().read_bytes",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
                records.append(
                    TelemetryRecord(
                        metric="disk_write_mb",
                        value=round(dio.write_bytes / (1024.0 * 1024.0), 2),
                        unit="MB",
                        classification="MEASURED",
                        source="psutil.disk_io_counters().write_bytes",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            nio = psutil.net_io_counters()
            if nio:
                records.append(
                    TelemetryRecord(
                        metric="network_bytes_sent_mb",
                        value=round(nio.bytes_sent / (1024.0 * 1024.0), 2),
                        unit="MB",
                        classification="MEASURED",
                        source="psutil.net_io_counters().bytes_sent",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
                records.append(
                    TelemetryRecord(
                        metric="network_bytes_recv_mb",
                        value=round(nio.bytes_recv / (1024.0 * 1024.0), 2),
                        unit="MB",
                        classification="MEASURED",
                        source="psutil.net_io_counters().bytes_recv",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
            cstats = psutil.cpu_stats()
            if cstats:
                records.append(
                    TelemetryRecord(
                        metric="cpu_context_switches",
                        value=cstats.ctx_switches,
                        unit="count",
                        classification="MEASURED",
                        source="psutil.cpu_stats().ctx_switches",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
                records.append(
                    TelemetryRecord(
                        metric="cpu_interrupts",
                        value=cstats.interrupts,
                        unit="count",
                        classification="MEASURED",
                        source="psutil.cpu_stats().interrupts",
                        timestamp=now,
                        backend=backend,
                        phase=phase,
                        status="VALID",
                        run_id=run_id,
                        device=dev_name,
                    )
                )
    except Exception:
        pass

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
    """Attempt direct physical power measurement from Hailo accelerator.

    Returns:
        (power_in_watts, classification, source, reason)
    """
    return read_hailo_power_watts()


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
