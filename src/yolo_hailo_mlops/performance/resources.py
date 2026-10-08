"""Unified performance reporting combining latency, throughput, and system telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from yolo_hailo_mlops.performance.latency import LatencyBenchmarkResult
from yolo_hailo_mlops.performance.throughput import ThroughputBenchmarkResult
from yolo_hailo_mlops.provenance import TelemetryRecord, collect_hardware_telemetry


@dataclass
class PerformanceReport:
    onnx_latency: Optional[LatencyBenchmarkResult]
    onnx_throughput: Optional[ThroughputBenchmarkResult]
    hailo_latency: Optional[LatencyBenchmarkResult]
    hailo_throughput: Optional[ThroughputBenchmarkResult]
    telemetry: List[TelemetryRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onnx_latency": self.onnx_latency.to_dict() if self.onnx_latency else None,
            "onnx_throughput": self.onnx_throughput.to_dict() if self.onnx_throughput else None,
            "hailo_latency": self.hailo_latency.to_dict() if self.hailo_latency else None,
            "hailo_throughput": self.hailo_throughput.to_dict() if self.hailo_throughput else None,
            "telemetry": [t.to_dict() for t in self.telemetry],
        }


def collect_performance_report(
    onnx_latency: Optional[LatencyBenchmarkResult] = None,
    onnx_throughput: Optional[ThroughputBenchmarkResult] = None,
    hailo_latency: Optional[LatencyBenchmarkResult] = None,
    hailo_throughput: Optional[ThroughputBenchmarkResult] = None,
) -> PerformanceReport:
    """Capture system telemetry snapshot alongside benchmark metrics."""
    telemetry = collect_hardware_telemetry()
    return PerformanceReport(
        onnx_latency=onnx_latency,
        onnx_throughput=onnx_throughput,
        hailo_latency=hailo_latency,
        hailo_throughput=hailo_throughput,
        telemetry=telemetry,
    )
