"""yolo_hailo_mlops - Reproducible YOLO11 -> ONNX -> Hailo-8L MLOps pipeline."""

__version__ = "1.0.0"
__author__ = "Mosudi I. O."

from yolo_hailo_mlops.exceptions import (
    AccuracyValidationError,
    ArtifactIntegrityError,
    CalibrationError,
    ConfigurationError,
    DatasetValidationError,
    ExportError,
    HailoCompilationError,
    HailoEnvironmentError,
    HailoOptimisationError,
    HailoParseError,
    HailoRuntimeError,
    ONNXValidationError,
    TrainingError,
    YoloHailoError,
)
from yolo_hailo_mlops.state import ExecutionMode, PhaseName, PhaseState, PipelineStatus, StateMachine

__all__ = [
    "__version__",
    "__author__",
    "YoloHailoError",
    "ConfigurationError",
    "DatasetValidationError",
    "TrainingError",
    "ExportError",
    "ONNXValidationError",
    "CalibrationError",
    "HailoEnvironmentError",
    "HailoParseError",
    "HailoOptimisationError",
    "HailoCompilationError",
    "HailoRuntimeError",
    "AccuracyValidationError",
    "ArtifactIntegrityError",
    "PhaseState",
    "PipelineStatus",
    "ExecutionMode",
    "PhaseName",
    "StateMachine",
]
