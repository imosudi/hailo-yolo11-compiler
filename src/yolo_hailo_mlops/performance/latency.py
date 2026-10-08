"""Inference latency measurement and percentile computation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from yolo_hailo_mlops.hailo.environment import detect_hailo_environment


@dataclass
class LatencyBenchmarkResult:
    target: str  # "onnx_cpu" or "hailo8l"
    warmup_iterations: int
    measurement_iterations: int
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_latency_percentiles(
    latencies: List[float],
    target: str,
    warmup: int,
) -> LatencyBenchmarkResult:
    """Calculate statistical distribution for a series of timed executions (in ms)."""
    arr = np.array(latencies, dtype=np.float64)
    return LatencyBenchmarkResult(
        target=target,
        warmup_iterations=warmup,
        measurement_iterations=len(latencies),
        mean_ms=round(float(np.mean(arr)), 3),
        median_ms=round(float(np.median(arr)), 3),
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        p99_ms=round(float(np.percentile(arr, 99)), 3),
        min_ms=round(float(np.min(arr)), 3),
        max_ms=round(float(np.max(arr)), 3),
        std_ms=round(float(np.std(arr)), 3),
    )


def benchmark_onnx_latency(
    onnx_path: Union[str, Path],
    warmup_iterations: int = 20,
    measurement_iterations: int = 100,
    imgsz: int = 640,
) -> Optional[LatencyBenchmarkResult]:
    """Measure ONNX Runtime FP32 CPU latency over configurable iterations."""
    p = Path(onnx_path).resolve()
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

        # Warmup
        for _ in range(warmup_iterations):
            _ = sess.run(None, {inp_name: dummy})

        # Measurement
        latencies: List[float] = []
        for _ in range(measurement_iterations):
            t0 = time.perf_counter()
            _ = sess.run(None, {inp_name: dummy})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        return compute_latency_percentiles(latencies, target="onnx_cpu", warmup=warmup_iterations)

    except Exception:
        return None


def benchmark_hailo_latency(
    hef_path: Union[str, Path],
    warmup_iterations: int = 20,
    measurement_iterations: int = 100,
) -> Optional[LatencyBenchmarkResult]:
    """Measure physical Hailo-8L accelerator latency on Raspberry Pi 5."""
    env = detect_hailo_environment()
    if not env.has_hardware or not env.has_hailort:
        return None

    try:
        import hailo_platform

        latencies: List[float] = []
        params = hailo_platform.VDevice.create_params()
        with hailo_platform.VDevice(params) as target:
            hef = hailo_platform.HEF(str(hef_path))
            configure_params = hailo_platform.ConfigureParams.create_from_hef(
                hef, interface=hailo_platform.HailoStreamInterface.PCIe
            )
            network_groups = target.configure(hef, configure_params)
            network_group = network_groups[0]
            network_group_params = network_group.create_params()

            input_vstreams_params = hailo_platform.InputVStreamParams.make(
                network_group, quantized=False, format_type=hailo_platform.FormatType.UINT8
            )
            output_vstreams_params = hailo_platform.OutputVStreamParams.make(
                network_group, quantized=False, format_type=hailo_platform.FormatType.FLOAT32
            )

            dummy_in = np.zeros((1, 640, 640, 3), dtype=np.uint8)

            with network_group.activate(network_group_params):
                with hailo_platform.InferVStreams(
                    network_group, input_vstreams_params, output_vstreams_params
                ) as pipeline:
                    inp_name = pipeline.get_input_vstream_infos()[0].name
                    in_dict = {inp_name: dummy_in}

                    # Warmup
                    for _ in range(warmup_iterations):
                        _ = pipeline.infer(in_dict)

                    # Measurement
                    for _ in range(measurement_iterations):
                        t0 = time.perf_counter()
                        _ = pipeline.infer(in_dict)
                        t1 = time.perf_counter()
                        latencies.append((t1 - t0) * 1000.0)

        return compute_latency_percentiles(latencies, target="hailo8l", warmup=warmup_iterations)

    except Exception:
        return None
