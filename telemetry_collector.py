#!/usr/bin/env python3
"""Telemetry collector for hailo-yolo11-compiler (Raspberry Pi 5 + Hailo-8L).

Emits JSON Lines using the same record shape as run_manifest.json
(metric, value, unit, classification, source, timestamp, reason, backend,
phase, status, run_id, device) plus monotonic_ns and optional labels.

Rules this collector follows
  * A metric that cannot be read is emitted with value null, status
    UNAVAILABLE and a reason. It never emits a default or guessed value.
  * "MB" means MiB (2**20 bytes), matching memory_available_mb in your manifest.
  * Rates and CPU percentages need a previous sample, so the first cycle
    reports those as UNAVAILABLE ("needs two samples").

Usage
  python3 telemetry_collector.py --phase performance_validation --backend hailo8l \
      --interval 1 --out telemetry.jsonl --pid <benchmark pid>
  python3 telemetry_collector.py --static-only --out -

As a library
  c = TelemetryCollector(backend="hailo8l", pid=bench_pid)
  records = c.static_samples() + c.begin_phase("onnx_cpu")
  ... call c.sample() every interval while the workload runs ...
  records += c.end_phase()
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

try:
    import psutil
except ImportError:  # collectors that need psutil report UNAVAILABLE
    psutil = None

MEASURED, DERIVED, CONFIGURED, UNAVAILABLE = "MEASURED", "DERIVED", "CONFIGURED", "UNAVAILABLE"
MIB = 2**20
HAILO_POWER_REASON = ("Hailo hardware interface does not support onboard power measurement "
                      "(no INA sensor on Raspberry Pi AI HAT+); use external instrumentation")


@dataclass
class S:
    """One metric reading before the record envelope is added."""
    metric: str
    value: object
    unit: str
    cls: str
    source: str
    reason: str | None = None
    labels: dict = field(default_factory=dict)


def unavailable(metric, unit, source, reason):
    return S(metric, None, unit, UNAVAILABLE, source, reason)


# ----------------------------------------------------------------- helpers
def rd(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def num(path, scale=1.0, nd=3):
    try:
        return round(float(rd(path)) * scale, nd)
    except (TypeError, ValueError):
        return None


def sh(cmd, timeout=3.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def slug(s):
    return re.sub(r"\W+", "_", s).strip("_")


def pkg_version(*names):
    for n in names:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return None


def kv_file(path):
    """Parse 'key value' lines (cgroup memory.events, cpu.stat, /proc/vmstat)."""
    d = {}
    for line in (rd(path) or "").splitlines():
        k, _, v = line.partition(" ")
        try:
            d[k] = int(v)
        except ValueError:
            pass
    return d


def proc_status_kb(pid, key):
    for line in (rd(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


# ----------------------------------------------------------------- parsers
THROTTLE_OCCURRED = ((16, "undervoltage"), (17, "frequency_capped"),
                     (18, "throttled"), (19, "soft_temp_limit"))


def parse_throttled(text):
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", text or "")
    return int(m.group(1), 16) if m else None


_PMIC = re.compile(r"^\s*(\w+)_([AV])\s+(?:current|volt)\(\d+\)=([-+0-9.eE]+)[AV]", re.M)


def parse_pmic(text):
    amps, volts = {}, {}
    for rail, kind, val in _PMIC.findall(text or ""):
        (amps if kind == "A" else volts)[rail] = float(val)
    return amps, volts


def parse_psi(text):
    """'some avg10=0.12 ...' lines -> {'some': 0.12, 'full': ...} (avg10 window)."""
    out = {}
    for line in (text or "").splitlines():
        kind, *kv = line.split()
        try:
            out[kind] = float(dict(p.split("=") for p in kv)["avg10"])
        except (KeyError, ValueError):
            pass
    return out


# --------------------------------------------------------------- collector
class TelemetryCollector:
    def __init__(self, run_id=None, backend="cpu", phase="idle", pid=None, path=".",
                 cache_dir=None, detail=False, hailo_probe=False):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        b = str(backend or "cpu").lower()
        if b in ("hailo", "hailo8l", "hailo-8l", "hailo-hat", "npu"):
            self.backend = "hailo-hat"
            self.inference_device = "NPU (Hailo HAT)"
        else:
            self.backend = "cpu"
            self.inference_device = "CPU"
        self.phase = phase
        self.pid = pid or os.getpid()
        self.path, self.cache_dir = path, cache_dir
        self.detail, self.hailo_probe = detail, hailo_probe
        self.device = f"{platform.system()} {platform.machine()}"
        self.persist_errors = 0
        self._errors = self._invalid = 0
        self._prev, self._base = {}, {}
        self._t_last = None
        self._temps = defaultdict(list)
        self._was_throttled = None
        self._throttle_events, self._throttled_s = 0, 0.0
        self._thr_prev = {}
        self._proc = self._self = None
        if psutil:
            self._self = psutil.Process()
            try:
                self._proc = psutil.Process(self.pid)
            except psutil.Error:
                pass
            # prime counters so the first real sample has a window to measure
            psutil.cpu_percent(None)
            psutil.cpu_percent(None, percpu=True)
            psutil.cpu_times_percent(None)
            self._self.cpu_percent(None)

    # ---- state helpers
    def _rate(self, key, value, now):
        prev, self._prev[key] = self._prev.get(key), (now, value)
        if prev is None or now <= prev[0]:
            return None
        return (value - prev[1]) / (now - prev[0])

    def _rs(self, metric, unit, src, key, value, now, scale=1.0, nd=3):
        r = self._rate(key, value, now)
        return S(metric, None if r is None else round(r * scale, nd), unit, DERIVED, src,
                 "needs two samples" if r is None else None)

    def _since(self, key, value):
        return value - self._base.setdefault(key, value)

    # ---- public API
    def static_samples(self):
        return self._records(self._gather([self.s_platform], time.monotonic()))

    def begin_phase(self, phase):
        self.phase = phase
        self._temps.clear()
        return self._records(list(self.s_phase_boundary("before")))

    def end_phase(self):
        return self._records(list(self.s_phase_boundary("after")) + list(self.s_temperature_stats()))

    def sample(self):
        now = time.monotonic()
        out = self._gather([self.c_hwmon, self.c_thermal, self.c_throttle, self.c_vcgencmd,
                            self.c_cpu, self.c_memory, self.c_process, self.c_cgroup,
                            self.c_storage, self.c_network, self.c_hailo], now)
        out += list(self._health(now, out))
        self._t_last = now
        return self._records(out)

    def _gather(self, fns, now):
        out = []
        for fn in fns:
            try:
                out.extend(fn(now))
            except Exception as e:  # one broken source must not stop the others
                self._errors += 1
                out.append(unavailable(f"collector_{fn.__name__[2:]}_failed", "", fn.__name__,
                                       f"{type(e).__name__}: {e}"))
        return out

    def _records(self, samples):
        ts, mono = datetime.now(timezone.utc).isoformat(), time.monotonic_ns()
        recs = []
        for s in samples:
            v = s.value
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            rec = dict(metric=s.metric, value=v, unit=s.unit,
                       classification=s.cls if v is not None else UNAVAILABLE, source=s.source,
                       timestamp=ts, monotonic_ns=mono, reason=s.reason, backend=self.backend,
                       inference_device=self.inference_device,
                       phase=self.phase, status="VALID" if v is not None else "UNAVAILABLE",
                       run_id=self.run_id, device=self.device)
            if s.labels:
                rec["labels"] = s.labels
            if isinstance(v, float) and (s.metric.endswith("_temp") or
                                         s.metric in ("cpu_temperature_c", "hailo_temperature_c")):
                self._temps[s.metric].append((mono / 1e9, v))
            recs.append(rec)
        return recs

    # ---- static platform info
    def s_platform(self, now):
        cfg = lambda m, v, u="string": S(m, v, u, CONFIGURED, "cli")
        yield cfg("benchmark_backend", self.backend, "enum")
        yield cfg("inference_device", self.inference_device, "string")
        mid = rd("/etc/machine-id")
        yield S("host_id", hashlib.sha256(mid.encode()).hexdigest()[:12] if mid else None, "string",
                DERIVED, "sha256(/etc/machine-id)[:12]", None if mid else "no machine-id")
        model = rd("/proc/device-tree/model")
        yield S("host_model", model.strip("\x00") if model else None, "string", MEASURED,
                "/proc/device-tree/model", None if model else "not a device-tree host")
        yield S("cpu_architecture", platform.machine(), "string", MEASURED, "platform.machine")
        yield S("cpu_core_count", os.cpu_count(), "count", MEASURED, "os.cpu_count")
        osr = dict(l.split("=", 1) for l in (rd("/etc/os-release") or "").splitlines() if "=" in l)
        yield S("os_name", osr.get("NAME", "").strip('"') or None, "string", MEASURED, "/etc/os-release")
        yield S("os_version", osr.get("VERSION_ID", "").strip('"') or None, "string", MEASURED, "/etc/os-release")
        yield S("kernel_version", platform.release(), "string", MEASURED, "platform.release")
        yield S("python_version", platform.python_version(), "string", MEASURED, "platform")
        for metric, names in (("ultralytics_version", ("ultralytics",)),
                              ("onnxruntime_version", ("onnxruntime", "onnxruntime-gpu"))):
            v = pkg_version(*names)
            yield S(metric, v, "string", MEASURED, "importlib.metadata", None if v else "package not installed")
        v = pkg_version("hailo_dataflow_compiler", "hailo-dataflow-compiler")
        if v is None:
            m = re.search(r"(\d+\.\d+\.\d+)", sh(["hailo", "--version"]) or "")
            v = m.group(1) if m else None
        yield S("hailo_dfc_version", v, "string", MEASURED, "importlib.metadata / hailo --version",
                None if v else "DFC not found")
        m = re.search(r"(\d+\.\d+\.\d+)", sh(["hailortcli", "--version"]) or "")
        yield S("hailo_runtime_version", m.group(1) if m else None, "string", MEASURED, "hailortcli --version",
                None if m else "hailortcli not found")
        drv = rd("/sys/module/hailo_pci/version") or (sh(["modinfo", "-F", "version", "hailo_pci"]) or "").strip() or None
        yield S("hailo_driver_version", drv, "string", MEASURED, "hailo_pci module", None if drv else "driver not loaded")
        info = self._probe() if self.hailo_probe else {"error": "HailoRT probe disabled (use --hailo-probe)"}
        yield S("hailo_firmware_version", info.get("firmware"), "string", MEASURED, "HailoRT Python API", info.get("error"))
        yield S("hailo_device_architecture", info.get("arch"), "string", MEASURED, "HailoRT Python API", info.get("error"))
        if psutil:
            yield S("memory_total_mb", round(psutil.virtual_memory().total / MIB, 2), "MB", MEASURED, "psutil")
            yield S("memory_swap_total_mb", round(psutil.swap_memory().total / MIB, 2), "MB", MEASURED, "psutil")

    def s_phase_boundary(self, when):
        try:
            du = shutil.disk_usage(self.path)
            yield S(f"disk_free_{when}_phase_bytes", du.free, "bytes", MEASURED, f"shutil.disk_usage({self.path})")
            if when == "before":
                yield S("filesystem_capacity_bytes", du.total, "bytes", MEASURED, f"shutil.disk_usage({self.path})")
        except OSError as e:
            yield unavailable(f"disk_free_{when}_phase_bytes", "bytes", self.path, str(e))

    def s_temperature_stats(self):
        for metric, pts in sorted(self._temps.items()):
            vals = [v for _, v in pts]
            lab, src = {"sensor": metric}, "derived from samples in this phase"
            yield S("temperature_min_c", min(vals), "°C", DERIVED, src, labels=lab)
            yield S("temperature_mean_c", round(sum(vals) / len(vals), 2), "°C", DERIVED, src, labels=lab)
            yield S("temperature_max_c", max(vals), "°C", DERIVED, src, labels=lab)
            rise, mins = vals[-1] - vals[0], (pts[-1][0] - pts[0][0]) / 60
            yield S("temperature_rise_c", round(rise, 2), "°C", DERIVED, "last minus first sample in phase", labels=lab)
            yield S("temperature_rise_rate_c_per_min", round(rise / mins, 3) if mins > 0 else None,
                    "°C/min", DERIVED, src, None if mins > 0 else "phase shorter than one sample", lab)

    # ---- thermal / fan / hwmon (the same sysfs data lm-sensors reads)
    def c_hwmon(self, now):
        for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            name = rd(f"{hw}/name") or Path(hw).name
            s = slug(name)
            for p in sorted(glob.glob(f"{hw}/temp*_input")):
                yield S(f"hwmon_{s}_{Path(p).name[:-6]}_temp", num(p, 1e-3), "°C", MEASURED, f"{p} ({name})")
            for p in sorted(glob.glob(f"{hw}/fan*_input")):
                v = num(p, 1, 0)
                yield S(f"hwmon_{s}_{Path(p).name[:-6]}_rpm", None if v is None else int(v), "RPM", MEASURED, f"{p} ({name})")
            for p in sorted(glob.glob(f"{hw}/pwm[0-9]")):
                n, raw = Path(p).name, num(p, 1, 0)
                yield S(f"hwmon_{s}_{n}_raw", raw, "0-255", MEASURED, f"{p} ({name})")
                yield S(f"hwmon_{s}_{n}_percent", None if raw is None else round(raw / 255 * 100, 1), "%",
                        DERIVED, "pwm raw / 255", labels={"note": "sensors -j may differ by about 0.1"})
                yield S(f"hwmon_{s}_{n}_enable", num(f"{p}_enable", 1, 0), "mode", MEASURED, f"{p}_enable")
            for p in sorted(glob.glob(f"{hw}/in[0-9]*_input")):
                yield S(f"hwmon_{s}_{Path(p).name[:-6]}_v", num(p, 1e-3), "V", MEASURED, f"{p} ({name})",
                        "raw ADC channel; meaning is chip specific")
            for p in sorted(glob.glob(f"{hw}/*_alarm") + glob.glob(f"{hw}/*_fault")):
                v = num(p, 1, 0)
                yield S(f"hwmon_{s}_{Path(p).name}", None if v is None else bool(v), "boolean", MEASURED, f"{p} ({name})")

    def c_thermal(self, now):
        cpu = None
        for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            typ, v = rd(f"{z}/type") or "unknown", num(f"{z}/temp", 1e-3)
            yield S(f"thermal_{Path(z).name}_temp", v, "°C", MEASURED, f"{z}/temp ({typ})")
            if cpu is None and v is not None and typ in ("cpu-thermal", "cpu_thermal", "soc-thermal"):
                cpu = (v, z)
        yield (S("cpu_temperature_c", cpu[0], "°C", MEASURED, f"{cpu[1]}/temp") if cpu else
               unavailable("cpu_temperature_c", "°C", "/sys/class/thermal", "no CPU thermal zone found"))
        for c in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
            lab = {"type": rd(f"{c}/type") or "unknown"}
            for k in ("cur_state", "max_state"):
                yield S(f"{Path(c).name}_{k}", num(f"{c}/{k}", 1, 0), "state", MEASURED, f"{c}/{k}", labels=lab)

    # ---- Raspberry Pi firmware: throttling, clocks, PMIC
    def c_throttle(self, now):
        src = "vcgencmd get_throttled"
        raw = parse_throttled(sh(["vcgencmd", "get_throttled"]))
        if raw is None:
            for m, u in (("rpi_throttled_raw", "hex"), ("rpi_undervoltage_now", "boolean"),
                         ("rpi_is_throttled", "boolean"), ("rpi_frequency_capped", "boolean")):
                yield unavailable(m, u, src, "vcgencmd missing or failed")
            return
        bit = lambda b: bool(raw >> b & 1)
        is_thr = bool(raw & 0xF)
        yield S("rpi_throttled_raw", f"0x{raw:x}", "hex", MEASURED, src)
        yield S("rpi_undervoltage_now", bit(0), "boolean", DERIVED, src + " bit 0")
        yield S("rpi_frequency_capped", bit(1), "boolean", DERIVED, src + " bit 1")
        yield S("rpi_throttled_now", bit(2), "boolean", DERIVED, src + " bit 2")
        yield S("rpi_soft_temp_limit_now", bit(3), "boolean", DERIVED, src + " bit 3")
        yield S("rpi_is_throttled", is_thr, "boolean", DERIVED, src + " bits 0-3")
        warnings = []
        for b, name in THROTTLE_OCCURRED:
            yield S(f"rpi_{name}_occurred", bit(b), "boolean", DERIVED, f"{src} bit {b} (sticky since boot)")
            if bit(b):
                warnings.append(f"{name}_occurred")
            if bit(b - 16):
                warnings.append(f"{name}_now")
        if self._was_throttled and self._t_last is not None:
            self._throttled_s += now - self._t_last
        if self._was_throttled is False and is_thr:
            self._throttle_events += 1
        self._was_throttled = is_thr
        yield S("rpi_throttling_events", self._throttle_events, "count", DERIVED, "transitions seen by this collector")
        yield S("rpi_throttled_duration_s", round(self._throttled_s, 3), "s", DERIVED,
                "sample-interval resolution")
        yield S("rpi_hardware_warnings", warnings, "list", DERIVED, src)

    def c_vcgencmd(self, now):
        m = re.search(r"temp=([\d.]+)", sh(["vcgencmd", "measure_temp"]) or "")
        yield S("rpi_soc_temp_c", float(m.group(1)) if m else None, "°C", MEASURED, "vcgencmd measure_temp",
                None if m else "vcgencmd failed")
        for dom in ("arm", "core"):
            m = re.search(r"=(\d+)", sh(["vcgencmd", "measure_clock", dom]) or "")
            yield S(f"rpi_{dom}_clock_hz", int(m.group(1)) if m else None, "Hz", MEASURED,
                    f"vcgencmd measure_clock {dom}", None if m else "vcgencmd failed")
        m = re.search(r"volt=([\d.]+)V", sh(["vcgencmd", "measure_volts", "core"]) or "")
        yield S("rpi_core_volts_v", float(m.group(1)) if m else None, "V", MEASURED,
                "vcgencmd measure_volts core", None if m else "not supported on this firmware")
        amps, volts = parse_pmic(sh(["vcgencmd", "pmic_read_adc"]))
        src = "vcgencmd pmic_read_adc"
        rails = [r for r in amps if r in volts and r != "EXT5V"]
        if rails:
            w = sum(amps[r] * volts[r] for r in rails)
            yield S("rpi_pmic_power", round(w, 3), "W", DERIVED, src,
                    "sum(I*V) over PMIC rails, excluding EXT5V; not wall power")
        else:
            yield unavailable("rpi_pmic_power", "W", src, "pmic_read_adc unavailable or unparsable")
        yield S("rpi_ext5v_v", volts.get("EXT5V"), "V", MEASURED, src, None if "EXT5V" in volts else "rail not reported")
        if self.detail:
            for r, a in amps.items():
                yield S(f"rpi_pmic_{slug(r)}_a", a, "A", MEASURED, src)
            for r, v in volts.items():
                yield S(f"rpi_pmic_{slug(r)}_v", v, "V", MEASURED, src)

    # ---- CPU
    def c_cpu(self, now):
        if psutil:
            src = "psutil (window since previous sample)"
            yield S("cpu_utilisation", psutil.cpu_percent(None), "%", MEASURED, src)
            for i, v in enumerate(psutil.cpu_percent(None, percpu=True)):
                yield S(f"cpu_utilisation_cpu{i}_percent", v, "%", MEASURED, src)
            t = psutil.cpu_times_percent(None)
            for k in ("user", "system", "idle", "iowait", "irq", "softirq"):
                yield S(f"cpu_{k}_utilisation", getattr(t, k, None), "%", MEASURED, src,
                        None if hasattr(t, k) else "not exposed")
            st = psutil.cpu_stats()
            yield self._rs("cpu_context_switches_per_s", "switches/s", "psutil.cpu_stats", "ctx", st.ctx_switches, now, nd=1)
            yield self._rs("cpu_interrupts_per_s", "interrupts/s", "psutil.cpu_stats", "irq", st.interrupts, now, nd=1)
        else:
            yield unavailable("cpu_utilisation", "%", "psutil", "psutil not installed")
        freqs = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")
        for p in sorted(freqs, key=lambda x: int(re.search(r"cpu(\d+)/", x).group(1))):
            m = re.search(r"cpu(\d+)/", p)
            cpu_idx = m.group(1) if m else "0"
            yield S(f"cpu_freq_cpu{cpu_idx}_mhz", num(p, 1e-3, 1), "MHz", MEASURED, p)
        if not freqs:
            yield unavailable("cpu_freq_cpu*_mhz", "MHz", "/sys/devices/system/cpu", "cpufreq not exposed")
        gov = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        yield S("cpu_frequency_governor", gov, "string", MEASURED, "cpufreq scaling_governor", None if gov else "not exposed")
        for name, v in zip(("1m", "5m", "15m"), os.getloadavg()):
            yield S(f"cpu_load_{name}", round(v, 2), "load", MEASURED, "os.getloadavg")
        st = {k: int(v) for k, _, v in (l.partition(" ") for l in (rd("/proc/stat") or "").splitlines())
              if k in ("procs_running", "procs_blocked")}
        yield S("cpu_runnable_processes", st.get("procs_running"), "count", MEASURED, "/proc/stat")
        yield S("cpu_blocked_processes", st.get("procs_blocked"), "count", MEASURED, "/proc/stat")
        yield S("cpu_process_count", sum(d.isdigit() for d in os.listdir("/proc")), "count", MEASURED, "/proc")
        try:
            threads = int(rd("/proc/loadavg").split()[3].split("/")[1])
        except (AttributeError, IndexError, ValueError):
            threads = None
        yield S("cpu_thread_count", threads, "count", MEASURED, "/proc/loadavg (kernel scheduling entities)")
        psi = parse_psi(rd("/proc/pressure/cpu"))
        for k in ("some", "full"):
            yield S(f"cpu_pressure_{k}_percent", psi.get(k), "%", MEASURED, "/proc/pressure/cpu avg10",
                    None if k in psi else "PSI not available on this kernel")

    # ---- memory
    def c_memory(self, now):
        if not psutil:
            yield unavailable("memory_utilisation", "%", "psutil", "psutil not installed")
            return
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        mb = lambda b: None if b is None else round(b / MIB, 2)
        yield S("memory_utilisation", vm.percent, "%", DERIVED, "psutil.virtual_memory().percent",
                "(total - available) / total * 100")
        for name, val in (("used", vm.used), ("available", vm.available), ("free", vm.free),
                          ("buffers", getattr(vm, "buffers", None)), ("cache", getattr(vm, "cached", None)),
                          ("shared", getattr(vm, "shared", None))):
            yield S(f"memory_{name}_mb", mb(val), "MB", MEASURED, "psutil.virtual_memory")
        yield S("memory_swap_used_mb", mb(sw.used), "MB", MEASURED, "psutil.swap_memory")
        yield self._rs("memory_swap_in_mb_s", "MB/s", "psutil.swap_memory", "sin", sw.sin, now, 1 / MIB)
        yield self._rs("memory_swap_out_mb_s", "MB/s", "psutil.swap_memory", "sout", sw.sout, now, 1 / MIB)
        vs = kv_file("/proc/vmstat")
        if "pgfault" in vs:
            yield self._rs("memory_page_faults_per_s", "faults/s", "/proc/vmstat", "pgf", vs["pgfault"], now, nd=1)
        else:
            yield unavailable("memory_page_faults_per_s", "faults/s", "/proc/vmstat", "pgfault not exposed")
        yield S("memory_oom_events", self._since("oom", vs["oom_kill"]) if "oom_kill" in vs else None, "count",
                DERIVED, "/proc/vmstat oom_kill since collector start", None if "oom_kill" in vs else "not exposed")
        psi = parse_psi(rd("/proc/pressure/memory"))
        for k in ("some", "full"):
            yield S(f"memory_pressure_{k}_percent", psi.get(k), "%", MEASURED, "/proc/pressure/memory avg10",
                    None if k in psi else "PSI not available on this kernel")

    def c_process(self, now):
        if not self._proc:
            yield unavailable("process_memory_rss_mb", "MB", f"pid {self.pid}", "psutil missing or pid not found")
            return
        p, lab = self._proc, {"pid": self.pid}
        with p.oneshot():
            mi, ct = p.memory_info(), p.cpu_times()
        yield S("process_memory_rss_mb", round(mi.rss / MIB, 2), "MB", MEASURED, "psutil.Process.memory_info", labels=lab)
        yield S("process_memory_vms_mb", round(mi.vms / MIB, 2), "MB", MEASURED, "psutil.Process.memory_info", labels=lab)
        hwm = proc_status_kb(self.pid, "VmHWM")
        yield S("process_memory_peak_rss_mb", None if hwm is None else round(hwm / 1024, 2), "MB", MEASURED,
                f"/proc/{self.pid}/status VmHWM", labels=lab)
        yield S("process_memory_growth_mb", round(self._since("rss0", mi.rss) / MIB, 2), "MB", DERIVED,
                "rss minus first observed rss", labels=lab)
        yield S("cpu_process_time_s", round(ct.user + ct.system, 3), "s", MEASURED, "psutil.Process.cpu_times", labels=lab)
        try:
            yield S("cpu_affinity", sorted(p.cpu_affinity()), "core ids", MEASURED, "psutil.Process.cpu_affinity", labels=lab)
        except (AttributeError, psutil.Error):
            yield unavailable("cpu_affinity", "core ids", "psutil", "not supported")
        if self.detail:
            for th in p.threads():
                tl, tt = {"pid": self.pid, "tid": th.id}, th.user_time + th.system_time
                r = self._rate(f"thr{th.id}", tt, now)
                yield S("cpu_thread_id", th.id, "integer", MEASURED, "psutil.Process.threads", labels=tl)
                yield S("cpu_thread_time_s", round(tt, 3), "s", MEASURED, "psutil.Process.threads", labels=tl)
                yield S("cpu_thread_utilisation_percent", None if r is None else round(r * 100, 1), "%", DERIVED,
                        "delta thread cpu time / delta wall time", "needs two samples" if r is None else None, tl)

    def c_cgroup(self, now):
        names = ("container_cgroup_memory_mb", "container_cgroup_memory_peak_mb",
                 "container_cgroup_memory_limit_mb", "container_cgroup_memory_events", "cpu_quota_throttled_s")
        units = ("MB", "MB", "MB", "object", "s")
        if not (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
            for n, u in zip(names, units):
                yield unavailable(n, u, "/sys/fs/cgroup", "monitored process is not in a container")
            return
        line = next((l for l in (rd(f"/proc/{self.pid}/cgroup") or "").splitlines() if l.startswith("0::")), "0::/")
        d = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        if not d.is_dir():
            d = Path("/sys/fs/cgroup")
        mb = lambda f: num(d / f, 1 / MIB, 2)
        yield S(names[0], mb("memory.current"), "MB", MEASURED, f"{d}/memory.current")
        yield S(names[1], mb("memory.peak"), "MB", MEASURED, f"{d}/memory.peak", "needs kernel 6.5+ for memory.peak")
        lim = rd(d / "memory.max")
        yield S(names[2], None if lim in (None, "max") else round(int(lim) / MIB, 2), "MB", MEASURED,
                f"{d}/memory.max", "no limit set" if lim == "max" else None)
        ev = kv_file(d / "memory.events")
        yield S(names[3], ev or None, "object", MEASURED, f"{d}/memory.events")
        thr = kv_file(d / "cpu.stat").get("throttled_usec")
        yield S(names[4], None if thr is None else round(thr / 1e6, 3), "s", MEASURED, f"{d}/cpu.stat",
                None if thr is not None else "no CPU quota or cpu controller disabled")

    # ---- storage / network
    def c_storage(self, now):
        try:
            du, sv = shutil.disk_usage(self.path), os.statvfs(self.path)
            yield S("filesystem_free_bytes", du.free, "bytes", MEASURED, f"shutil.disk_usage({self.path})")
            yield S("filesystem_available_inodes", sv.f_favail, "count", MEASURED, f"os.statvfs({self.path})")
        except OSError as e:
            yield unavailable("filesystem_free_bytes", "bytes", self.path, str(e))
        io = psutil.disk_io_counters() if psutil else None
        if io:
            src = "psutil.disk_io_counters (all disks)"
            yield self._rs("storage_read_bytes_per_s", "bytes/s", src, "rb", io.read_bytes, now, nd=0)
            yield self._rs("storage_write_bytes_per_s", "bytes/s", src, "wb", io.write_bytes, now, nd=0)
            yield self._rs("storage_read_ops_per_s", "ops/s", src, "ro", io.read_count, now, nd=1)
            yield self._rs("storage_write_ops_per_s", "ops/s", src, "wo", io.write_count, now, nd=1)
            ops, busy = io.read_count + io.write_count, getattr(io, "read_time", 0) + getattr(io, "write_time", 0)
            po, pb = self._prev.get("lat", (None, (ops, busy)))[1], (ops, busy)
            self._prev["lat"] = (now, pb)
            dops = pb[0] - po[0] if po else 0
            yield S("storage_io_latency_ms", round((pb[1] - po[1]) / dops, 3) if po and dops > 0 else None, "ms", DERIVED,
                    "delta (read_time+write_time) / delta ops", None if po and dops > 0 else "needs two samples with I/O")
        else:
            yield unavailable("storage_read_bytes_per_s", "bytes/s", "psutil", "disk counters unavailable")
        if psutil:
            yield S("storage_io_wait_percent", getattr(psutil.cpu_times_percent(None), "iowait", None), "%", DERIVED,
                    "cpu iowait")
        psi = parse_psi(rd("/proc/pressure/io"))
        yield S("storage_io_pressure_percent", psi.get("some"), "%", MEASURED, "/proc/pressure/io avg10",
                None if "some" in psi else "PSI not available on this kernel")
        if self.cache_dir:
            tot = sum(f.stat().st_size for f in Path(self.cache_dir).rglob("*") if f.is_file())
            yield S("cache_directory_size_bytes", tot, "bytes", MEASURED, f"scan of {self.cache_dir}")

    def c_network(self, now):
        sync = (sh(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]) or "").strip()
        yield S("clock_sync_status", sync == "yes" if sync else None, "boolean", MEASURED, "timedatectl",
                None if sync else "timedatectl unavailable")
        if not psutil:
            yield unavailable("network_bytes_sent", "bytes", "psutil", "psutil not installed")
            return
        nics = {k: v for k, v in psutil.net_io_counters(pernic=True).items() if k != "lo"}
        tot = lambda f: sum(getattr(v, f) for v in nics.values())
        src = "psutil.net_io_counters (excluding lo)"
        for name, f in (("bytes_sent", "bytes_sent"), ("bytes_received", "bytes_recv"),
                        ("packets_sent", "packets_sent"), ("packets_received", "packets_recv")):
            yield S(f"network_{name}", self._since(name, tot(f)), "bytes" if "bytes" in name else "count", DERIVED,
                    src + " since collector start")
        yield self._rs("network_throughput_sent_bytes_per_s", "bytes/s", src, "tx", tot("bytes_sent"), now, nd=0)
        yield self._rs("network_throughput_received_bytes_per_s", "bytes/s", src, "rx", tot("bytes_recv"), now, nd=0)
        yield S("network_dropped_packets", self._since("drop", tot("dropin") + tot("dropout")), "count", DERIVED, src)
        yield S("network_interface_errors", self._since("nerr", tot("errin") + tot("errout")), "count", DERIVED, src)
        up = [(n, s) for n, s in sorted(psutil.net_if_stats().items()) if n != "lo" and s.isup]
        yield S("network_interface_status", f"{up[0][0]}:up" if up else "none up", "string", MEASURED, "psutil.net_if_stats")
        yield S("network_link_speed_mbps", up[0][1].speed or None if up else None, "Mbit/s", MEASURED,
                "psutil.net_if_stats", None if up and up[0][1].speed else "speed not reported")

    # ---- Hailo
    def _probe(self):
        """Opt-in. Opens the device through the HailoRT Python API; unverified on your version."""
        try:
            try:
                from hailo_platform import Device
            except ImportError:
                from hailo_platform import PcieDevice as Device
        except ImportError:
            return {"error": "hailo_platform (HailoRT Python API) not importable in this venv"}
        info = {}
        try:
            dev = Device()
        except Exception as e:
            return {"error": f"cannot open device (busy or API mismatch): {type(e).__name__}: {e}"}
        try:
            ident = dev.control.identify()
            info["firmware"] = str(getattr(ident, "firmware_version", None))
            info["arch"] = str(getattr(ident, "device_architecture", None))
        except Exception as e:
            info["error"] = f"identify failed: {type(e).__name__}: {e}"
        try:
            t = dev.control.get_chip_temperature()
            temps = [getattr(t, k, None) for k in ("ts0_temperature", "ts1_temperature")]
            info["temps"] = [x for x in temps if x is not None]
        except Exception as e:
            info.setdefault("error", f"get_chip_temperature failed: {type(e).__name__}: {e}")
        finally:
            getattr(dev, "release", lambda: None)()
        return info

    def c_hailo(self, now):
        present = Path("/dev/hailo0").exists()
        yield S("hailo_device_present", present, "boolean", MEASURED, "/dev/hailo0")
        yield S("hailo_driver_loaded", Path("/sys/module/hailo_pci").exists(), "boolean", MEASURED, "/sys/module/hailo_pci")
        info = (self._probe() if self.hailo_probe and present else
                {"error": "HailoRT probe disabled (use --hailo-probe)" if not self.hailo_probe else "no device"})
        temps = info.get("temps")
        yield S("hailo_temperature_c", max(temps) if temps else None, "°C", MEASURED, "HailoRT get_chip_temperature",
                None if temps else info.get("error", "temperature not exposed"))
        if temps:
            for i, t in enumerate(temps):
                yield S(f"hailo_ts{i}_temperature_c", t, "°C", MEASURED, "HailoRT get_chip_temperature")
        yield S("hailo_device_initialised", ("firmware" in info) if self.hailo_probe and present else None, "boolean",
                MEASURED, "HailoRT identify", None if self.hailo_probe and present else "needs --hailo-probe")
        for m in ("hailo_power_watts", "hailo_device_power_watts"):
            yield unavailable(m, "W", "HailoRT", HAILO_POWER_REASON)

    # ---- telemetry health
    def _health(self, now, out):
        self._invalid += sum(isinstance(s.value, float) and not math.isfinite(s.value) for s in out)
        yield S("telemetry_collection_interval_ms", None if self._t_last is None else round((now - self._t_last) * 1e3, 1),
                "ms", MEASURED, "monotonic clock", "first sample" if self._t_last is None else None)
        yield S("telemetry_collection_errors", self._errors, "count", MEASURED, "collector")
        yield S("telemetry_missing_samples", sum(s.value is None for s in out), "count", DERIVED, "this cycle")
        yield S("telemetry_invalid_samples", self._invalid, "count", DERIVED, "non-finite values since start")
        yield S("telemetry_persistence_errors", self.persist_errors, "count", MEASURED, "writer")
        if self._self:
            yield S("telemetry_collector_cpu_percent", self._self.cpu_percent(None), "%", MEASURED, "psutil")
            yield S("telemetry_collector_memory_mb", round(self._self.memory_info().rss / MIB, 2), "MB", MEASURED, "psutil")


# --------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = until Ctrl-C)")
    ap.add_argument("--phase", default="idle")
    ap.add_argument("--backend", default=None, choices=("cpu", "hailo", "hailo8l", "hailo-hat", "npu"),
                    help="inference execution backend ('cpu' or 'hailo-hat')")
    ap.add_argument("--target", "--accelerator", dest="target", default=None,
                    choices=("cpu", "hailo", "hailo8l", "hailo-hat", "npu"),
                    help="target accelerator ('cpu' or 'hailo-hat')")
    ap.add_argument("--cpu", action="store_true", help="indicate CPU is running current inference")
    ap.add_argument("--hailo", "--npu", "--hailo-hat", dest="hailo", action="store_true",
                    help="indicate Raspberry Pi AI HAT+ Hailo NPU is running current inference")
    ap.add_argument("--run-id")
    ap.add_argument("--out", default="telemetry.jsonl", help="JSONL path, or - for stdout")
    ap.add_argument("--pid", type=int, help="process to monitor (default: the collector itself)")
    ap.add_argument("--path", default=".", help="filesystem to report on")
    ap.add_argument("--cache-dir", help="directory whose size to report")
    ap.add_argument("--detail", action="store_true", help="per-thread and per-PMIC-rail metrics")
    ap.add_argument("--hailo-probe", action="store_true",
                    help="open the Hailo device via the Python API for firmware/temperature (may fail if busy)")
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args(argv)

    chosen_backend = "cpu"
    if a.cpu:
        chosen_backend = "cpu"
    elif a.hailo:
        chosen_backend = "hailo-hat"
    elif a.target:
        chosen_backend = a.target
    elif a.backend:
        chosen_backend = a.backend
    else:
        chosen_backend = "cpu"

    c = TelemetryCollector(a.run_id, chosen_backend, a.phase, a.pid, a.path, a.cache_dir, a.detail, a.hailo_probe)
    out = sys.stdout if a.out == "-" else open(a.out, "a", buffering=1)

    def emit(recs):
        for r in recs:
            try:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            except OSError:
                c.persist_errors += 1

    emit(c.static_samples())
    if not a.static_only:
        emit(c.begin_phase(a.phase))
        end = time.monotonic() + a.duration if a.duration else None
        try:
            while end is None or time.monotonic() < end:
                time.sleep(a.interval)
                emit(c.sample())
        except KeyboardInterrupt:
            pass
        emit(c.end_phase())
    if out is not sys.stdout:
        out.close()
        print(f"run_id {c.run_id} written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
