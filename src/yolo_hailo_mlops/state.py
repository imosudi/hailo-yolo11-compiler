"""Pipeline state machine, phase tracking, and execution boundaries."""

from __future__ import annotations

import datetime
import enum
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class PhaseState(str, enum.Enum):
    """Execution status for an individual phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_EXECUTED = "NOT_EXECUTED"
    CANCELLED = "CANCELLED"


class PipelineStatus(str, enum.Enum):
    """End-to-end pipeline execution status."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NOT_EXECUTED = "NOT_EXECUTED"
    CANCELLED = "CANCELLED"


class ExecutionMode(str, enum.Enum):
    """Explicit boundary defining allowed execution capabilities."""
    PORTABLE = "portable"           # Workstation/Host: validation, train, export, calibrate
    HAILO_COMPILE = "hailo_compile" # Host with DFC: parse, INT8 PTQ, compile HEF
    HAILO_RUNTIME = "hailo_runtime" # Raspberry Pi 5 + Hailo-8L: physical inference & telemetry
    AUTO = "auto"                   # Automatically detect host capabilities


class PhaseName(str, enum.Enum):
    """Standardized phase identifiers."""
    DOCTOR = "doctor"
    DATASET_VALIDATION = "dataset_validation"
    TRAINING = "training"
    ONNX_EXPORT = "onnx_export"
    ONNX_VALIDATION = "onnx_validation"
    NUMERICAL_VALIDATION = "numerical_validation"
    CALIBRATION = "calibration"
    HAILO_PARSE = "hailo_parse"
    HAILO_OPTIMISATION = "hailo_optimisation"
    HAILO_COMPILE = "hailo_compile"
    HAILO_VALIDATION = "hailo_validation"
    ACCURACY_VALIDATION = "accuracy_validation"
    PERFORMANCE_VALIDATION = "performance_validation"
    REPORTING = "reporting"


@dataclass
class PhaseRecord:
    """Historical and state tracking record for an individual phase."""
    name: str
    state: PhaseState = PhaseState.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    reason: Optional[str] = None
    artifacts_produced: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _start_monotonic: Optional[float] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_start_monotonic", None)
        data["state"] = self.state.value
        return data


class StateMachine:
    """Explicit deterministic state machine managing the pipeline life-cycle."""

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.AUTO) -> None:
        self.execution_mode = execution_mode
        self.phases: Dict[str, PhaseRecord] = {}
        for phase in PhaseName:
            self.phases[phase.value] = PhaseRecord(name=phase.value)
        self.started_at: str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self._start_monotonic: float = time.monotonic()
        self.cancellation_reason: Optional[str] = None

    def start_phase(self, phase_name: str | PhaseName) -> PhaseRecord:
        """Mark a phase as running."""
        name = phase_name.value if isinstance(phase_name, PhaseName) else phase_name
        record = self.phases.setdefault(name, PhaseRecord(name=name))
        record.state = PhaseState.RUNNING
        record.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record._start_monotonic = time.monotonic()
        return record

    def complete_phase(
        self,
        phase_name: str | PhaseName,
        artifacts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhaseRecord:
        """Mark a phase as successfully completed."""
        name = phase_name.value if isinstance(phase_name, PhaseName) else phase_name
        record = self.phases[name]
        record.state = PhaseState.SUCCESS
        record.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if record._start_monotonic is not None:
            record.duration_seconds = round(time.monotonic() - record._start_monotonic, 3)
        if artifacts:
            record.artifacts_produced.extend(artifacts)
        if metadata:
            record.metadata.update(metadata)
        return record

    def fail_phase(
        self,
        phase_name: str | PhaseName,
        error_code: str,
        error_message: str,
        reason: Optional[str] = None,
    ) -> PhaseRecord:
        """Mark a phase as failed with diagnostic details."""
        name = phase_name.value if isinstance(phase_name, PhaseName) else phase_name
        record = self.phases[name]
        record.state = PhaseState.FAILED
        record.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if record._start_monotonic is not None:
            record.duration_seconds = round(time.monotonic() - record._start_monotonic, 3)
        record.error_code = error_code
        record.error_message = error_message
        record.reason = reason or error_message
        return record

    def skip_phase(self, phase_name: str | PhaseName, reason: str) -> PhaseRecord:
        """Mark a phase as skipped."""
        name = phase_name.value if isinstance(phase_name, PhaseName) else phase_name
        record = self.phases[name]
        record.state = PhaseState.SKIPPED
        record.reason = reason
        return record

    def mark_not_executed(self, phase_name: str | PhaseName, reason: str) -> PhaseRecord:
        """Mark a phase as NOT_EXECUTED due to environment or boundary constraints."""
        name = phase_name.value if isinstance(phase_name, PhaseName) else phase_name
        record = self.phases[name]
        record.state = PhaseState.NOT_EXECUTED
        record.reason = reason
        return record

    def cancel_pipeline(self, reason: str) -> None:
        """Handle graceful user cancellation (SIGINT/SIGTERM)."""
        self.cancellation_reason = reason
        self.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for record in self.phases.values():
            if record.state in (PhaseState.PENDING, PhaseState.RUNNING):
                record.state = PhaseState.CANCELLED
                record.reason = reason

    def finish(self) -> PipelineStatus:
        """Conclude the state machine and compute overall pipeline status."""
        self.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return self.compute_pipeline_status()

    def compute_pipeline_status(self) -> PipelineStatus:
        """Derive overall pipeline status based on executed phases and boundaries."""
        states = [record.state for record in self.phases.values()]

        if any(s == PhaseState.CANCELLED for s in states):
            return PipelineStatus.CANCELLED

        if any(s == PhaseState.FAILED for s in states):
            return PipelineStatus.FAILED

        executed_successes = [
            name for name, record in self.phases.items()
            if record.state == PhaseState.SUCCESS
        ]

        if not executed_successes:
            return PipelineStatus.NOT_EXECUTED

        # End-to-end validation requires successful hailo compile and runtime validation
        hailo_compile_success = (
            self.phases.get(PhaseName.HAILO_COMPILE.value, PhaseRecord(name="")).state
            == PhaseState.SUCCESS
        )
        hailo_runtime_success = (
            self.phases.get(PhaseName.HAILO_VALIDATION.value, PhaseRecord(name="")).state
            == PhaseState.SUCCESS
        )

        if self.execution_mode == ExecutionMode.PORTABLE:
            # In portable mode, full success on portable phases qualifies as SUCCESS for portable mode
            return PipelineStatus.SUCCESS

        if hailo_compile_success and hailo_runtime_success:
            return PipelineStatus.SUCCESS

        # If some phases ran successfully but Hailo compilation/runtime was skipped or not executed
        if any(
            record.state in (PhaseState.NOT_EXECUTED, PhaseState.SKIPPED)
            for record in self.phases.values()
            if record.name in (PhaseName.HAILO_COMPILE.value, PhaseName.HAILO_VALIDATION.value)
        ):
            return PipelineStatus.PARTIAL

        return PipelineStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine status to structured dictionary."""
        return {
            "execution_mode": self.execution_mode.value,
            "pipeline_status": self.compute_pipeline_status().value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(time.monotonic() - self._start_monotonic, 3),
            "cancellation_reason": self.cancellation_reason,
            "phases": {name: record.to_dict() for name, record in self.phases.items()},
        }
