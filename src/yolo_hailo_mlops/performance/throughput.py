"""Throughput (Frames Per Second) benchmarking and pipeline calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from yolo_hailo_mlops.performance.latency import LatencyBenchmarkResult


@dataclass
class ThroughputBenchmarkResult:
    target: str
    fps: float
    mean_frame_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_throughput_from_latency(
    latency_result: Optional[LatencyBenchmarkResult],
) -> Optional[ThroughputBenchmarkResult]:
    """Derive throughput (FPS) from measured mean latency."""
    if latency_result is None or latency_result.mean_ms <= 0:
        return None

    fps = round(1000.0 / latency_result.mean_ms, 2)
    return ThroughputBenchmarkResult(
        target=latency_result.target,
        fps=fps,
        mean_frame_time_ms=latency_result.mean_ms,
    )
