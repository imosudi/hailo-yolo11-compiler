"""Typed exception taxonomy with structured error codes, remediation, and provenance."""

from __future__ import annotations

from typing import Optional


class YoloHailoError(Exception):
    """Base exception for all pipeline errors with structured diagnostics."""

    def __init__(
        self,
        message: str,
        code: str,
        phase: str,
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.phase = phase
        self.remediation = remediation
        self.artifact = artifact

    def format_diagnostic(self) -> str:
        """Render a standardized diagnostic block for terminal and log files."""
        lines = [
            f"[{self.code}] Phase: {self.phase}",
            f"Cause: {self.message}",
        ]
        if self.artifact:
            lines.append(f"Affected Artifact: {self.artifact}")
        if self.remediation:
            lines.append("Remediation:")
            for rem_line in self.remediation.strip().splitlines():
                lines.append(f"  {rem_line}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_diagnostic()


class ConfigurationError(YoloHailoError):
    """Raised when configuration validation, file loading, or precedence resolution fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-CFG-001",
        phase: str = "configuration",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Inspect config/config.yaml or CLI overrides and correct the invalid parameter.",
            artifact=artifact,
        )


class DatasetValidationError(YoloHailoError):
    """Raised when dataset pre-flight validation detects corruption, leakage, or missing labels."""

    def __init__(
        self,
        message: str,
        code: str = "E-DATA-001",
        phase: str = "dataset_validation",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Verify the dataset YAML structure, image readability, and bounding-box label formats.",
            artifact=artifact,
        )


class TrainingError(YoloHailoError):
    """Raised when model training or checkpoint extraction fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-TRAIN-001",
        phase: str = "training",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Ensure PyTorch and Ultralytics are installed with sufficient GPU/CPU memory.",
            artifact=artifact,
        )


class ExportError(YoloHailoError):
    """Raised when ONNX graph export fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-EXP-001",
        phase: str = "onnx_export",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Verify model checkpoint integrity and static dimension parameters (1x3x640x640).",
            artifact=artifact,
        )


class ONNXValidationError(YoloHailoError):
    """Raised when exported ONNX violates shape contracts or contains NaN/Inf values."""

    def __init__(
        self,
        message: str,
        code: str = "E-ONNX-001",
        phase: str = "onnx_validation",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Re-export model ensuring static batch=1 and fixed 640x640 spatial dimensions.",
            artifact=artifact,
        )


class CalibrationError(YoloHailoError):
    """Raised when calibration dataset sampling or preprocessing fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-CAL-001",
        phase: str = "calibration",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Ensure the source training images exist, are uncorrupted, and count is in [100, 200].",
            artifact=artifact,
        )


class HailoEnvironmentError(YoloHailoError):
    """Raised when Hailo DFC or HailoRT environment is missing or incompatible."""

    def __init__(
        self,
        message: str,
        code: str = "E-HAILO-ENV-001",
        phase: str = "hailo_environment",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Install Hailo Dataflow Compiler (DFC) or HailoRT in your environment.",
            artifact=artifact,
        )


class HailoParseError(YoloHailoError):
    """Raised when translating ONNX to Hailo Archive (HAR) fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-HAILO-PARSE-001",
        phase: str = "hailo_parse",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Check ONNX operator compatibility with target architecture 'hailo8l'.",
            artifact=artifact,
        )


class HailoOptimisationError(YoloHailoError):
    """Raised during INT8 post-training quantisation or layer optimisation."""

    def __init__(
        self,
        message: str,
        code: str = "E-HAILO-OPT-001",
        phase: str = "hailo_optimisation",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Verify calibration dataset preprocessing (RGB, 640x640, normalised ranges).",
            artifact=artifact,
        )


class HailoCompilationError(YoloHailoError):
    """Raised when compiling optimised HAR to Hailo Executable Format (HEF) fails."""

    def __init__(
        self,
        message: str,
        code: str = "E-HAILO-COMP-001",
        phase: str = "hailo_compile",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Ensure model capacity and layer routing fit the Hailo-8L resource limits.",
            artifact=artifact,
        )


class HailoRuntimeError(YoloHailoError):
    """Raised during physical HailoRT inference on Raspberry Pi 5."""

    def __init__(
        self,
        message: str,
        code: str = "E-HAILO-RT-001",
        phase: str = "hailo_runtime",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Verify Raspberry Pi 5 PCIe HAT+ connection, dmesg logs, and /dev/hailo0 permissions.",
            artifact=artifact,
        )


class AccuracyValidationError(YoloHailoError):
    """Raised when model evaluation fails configured accuracy degradation thresholds."""

    def __init__(
        self,
        message: str,
        code: str = "E-ACC-001",
        phase: str = "accuracy_validation",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Increase calibration image count, verify representative distribution, or adjust gate thresholds.",
            artifact=artifact,
        )


class ArtifactIntegrityError(YoloHailoError):
    """Raised when SHA-256 validation or artifact structural checks fail."""

    def __init__(
        self,
        message: str,
        code: str = "E-ART-001",
        phase: str = "artifact_integrity",
        remediation: Optional[str] = None,
        artifact: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            phase=phase,
            remediation=remediation or "Re-run the producer phase with --force to regenerate the corrupted artifact.",
            artifact=artifact,
        )
