"""Evaluation metrics data structures and calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AccuracyMetrics:
    model_type: str  # "pytorch_fp32", "onnx_fp32", or "hailo_int8"
    map50: float
    map50_95: float
    precision: float
    recall: float
    per_class_ap50: Dict[str, float] = field(default_factory=dict)
    detection_count: int = 0
    samples_evaluated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NumericalComparisonReport:
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    shapes_match: bool
    status: str  # "PASSED" or "FAILED"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccuracyGateResult:
    passed: bool
    status: str  # "PASSED" or "FAILED"
    map50_drop: float
    map50_threshold: float
    map5095_drop: float
    map5095_threshold: float
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComprehensiveEvaluationReport:
    pytorch_metrics: Optional[AccuracyMetrics]
    onnx_metrics: Optional[AccuracyMetrics]
    hailo_metrics: Optional[AccuracyMetrics]
    numerical_comparison: Optional[NumericalComparisonReport]
    accuracy_gate: AccuracyGateResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pytorch": self.pytorch_metrics.to_dict() if self.pytorch_metrics else None,
            "onnx": self.onnx_metrics.to_dict() if self.onnx_metrics else None,
            "hailo": self.hailo_metrics.to_dict() if self.hailo_metrics else None,
            "numerical_comparison": self.numerical_comparison.to_dict() if self.numerical_comparison else None,
            "accuracy_gate": self.accuracy_gate.to_dict(),
        }
