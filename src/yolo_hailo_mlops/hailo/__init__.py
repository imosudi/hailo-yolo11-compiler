"""Hailo Dataflow Compiler (DFC) and HailoRT execution abstraction subsystem."""

from yolo_hailo_mlops.hailo.environment import HailoEnvironment, detect_hailo_environment

__all__ = [
    "HailoEnvironment",
    "detect_hailo_environment",
    "HailoParser",
    "HailoOptimizer",
    "HailoCompiler",
    "validate_hef",
    "HEFValidationReport",
    "HailoRuntimeValidator",
    "HailoRuntimeReport",
]


def __getattr__(name: str):
    """Lazily import compiler, optimiser, and runtime modules on demand."""
    if name == "HailoParser":
        from yolo_hailo_mlops.hailo.parser import HailoParser
        return HailoParser
    if name == "HailoOptimizer":
        from yolo_hailo_mlops.hailo.optimizer import HailoOptimizer
        return HailoOptimizer
    if name == "HailoCompiler":
        from yolo_hailo_mlops.hailo.compiler import HailoCompiler
        return HailoCompiler
    if name in ("validate_hef", "HEFValidationReport"):
        from yolo_hailo_mlops.hailo.validator import HEFValidationReport, validate_hef
        return HEFValidationReport if name == "HEFValidationReport" else validate_hef
    if name in ("HailoRuntimeValidator", "HailoRuntimeReport"):
        from yolo_hailo_mlops.hailo.runtime import HailoRuntimeReport, HailoRuntimeValidator
        return HailoRuntimeReport if name == "HailoRuntimeReport" else HailoRuntimeValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

