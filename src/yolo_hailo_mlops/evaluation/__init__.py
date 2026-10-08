"""Quantitative accuracy and numerical evaluation subsystem."""

from yolo_hailo_mlops.evaluation.comparison import build_evaluation_report, evaluate_accuracy_gates
from yolo_hailo_mlops.evaluation.hailo import evaluate_hailo_model
from yolo_hailo_mlops.evaluation.metrics import (
    AccuracyGateResult,
    AccuracyMetrics,
    ComprehensiveEvaluationReport,
    NumericalComparisonReport,
)
from yolo_hailo_mlops.evaluation.onnx import compare_pytorch_onnx_numerical, evaluate_onnx_model
from yolo_hailo_mlops.evaluation.pytorch import evaluate_pytorch_model

__all__ = [
    "AccuracyMetrics",
    "NumericalComparisonReport",
    "AccuracyGateResult",
    "ComprehensiveEvaluationReport",
    "evaluate_pytorch_model",
    "evaluate_onnx_model",
    "compare_pytorch_onnx_numerical",
    "evaluate_hailo_model",
    "evaluate_accuracy_gates",
    "build_evaluation_report",
]
