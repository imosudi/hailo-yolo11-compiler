"""Accuracy comparison across models and configurable degradation gates."""

from __future__ import annotations

from typing import Optional

from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.evaluation.metrics import (
    AccuracyGateResult,
    AccuracyMetrics,
    ComprehensiveEvaluationReport,
    NumericalComparisonReport,
)
from yolo_hailo_mlops.exceptions import AccuracyValidationError


def evaluate_accuracy_gates(
    baseline: AccuracyMetrics,
    candidate: AccuracyMetrics,
    config: AppConfig,
) -> AccuracyGateResult:
    """Compare candidate metrics against baseline and enforce drop thresholds."""
    max_drop_50 = config.validation.accuracy.map50_max_drop
    max_drop_5095 = config.validation.accuracy.map5095_max_drop

    drop_50 = round(baseline.map50 - candidate.map50, 4)
    drop_5095 = round(baseline.map50_95 - candidate.map50_95, 4)

    passed = True
    reasons = []

    if drop_50 > max_drop_50:
        passed = False
        reasons.append(
            f"mAP50 degradation ({drop_50:.4f}) exceeds threshold ({max_drop_50:.4f})"
        )

    if drop_5095 > max_drop_5095:
        passed = False
        reasons.append(
            f"mAP50-95 degradation ({drop_5095:.4f}) exceeds threshold ({max_drop_5095:.4f})"
        )

    status = "PASSED" if passed else "FAILED"
    reason_str = "; ".join(reasons) if reasons else "Accuracy within permissible bounds."

    return AccuracyGateResult(
        passed=passed,
        status=status,
        map50_drop=drop_50,
        map50_threshold=max_drop_50,
        map5095_drop=drop_5095,
        map5095_threshold=max_drop_5095,
        reason=reason_str,
    )


def build_evaluation_report(
    pytorch_metrics: Optional[AccuracyMetrics],
    onnx_metrics: Optional[AccuracyMetrics],
    hailo_metrics: Optional[AccuracyMetrics],
    numerical_comparison: Optional[NumericalComparisonReport],
    config: AppConfig,
) -> ComprehensiveEvaluationReport:
    """Build unified comparative evaluation report."""
    # Enforce accuracy gates between baseline and downstream models
    if pytorch_metrics and onnx_metrics:
        gate_result = evaluate_accuracy_gates(pytorch_metrics, onnx_metrics, config)
    elif pytorch_metrics and hailo_metrics:
        gate_result = evaluate_accuracy_gates(pytorch_metrics, hailo_metrics, config)
    else:
        # If no downstream model evaluated
        gate_result = AccuracyGateResult(
            passed=True,
            status="PASSED",
            map50_drop=0.0,
            map50_threshold=config.validation.accuracy.map50_max_drop,
            map5095_drop=0.0,
            map5095_threshold=config.validation.accuracy.map5095_max_drop,
            reason="Baseline only evaluated; no downstream degradation detected.",
        )

    report = ComprehensiveEvaluationReport(
        pytorch_metrics=pytorch_metrics,
        onnx_metrics=onnx_metrics,
        hailo_metrics=hailo_metrics,
        numerical_comparison=numerical_comparison,
        accuracy_gate=gate_result,
    )

    if not gate_result.passed and config.validation.enabled:
        raise AccuracyValidationError(
            message=f"Accuracy validation gate failed: {gate_result.reason}",
            code="E-ACC-005",
        )

    return report
