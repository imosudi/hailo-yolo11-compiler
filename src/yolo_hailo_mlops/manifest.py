"""Experiment run manifest serialisation and lifecycle tracking."""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from yolo_hailo_mlops.checksums import ArtifactRecord
from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.provenance import TelemetryRecord, get_environment_info
from yolo_hailo_mlops.utils.filesystem import atomic_write


@dataclass
class RunManifest:
    """Master experiment ledger documenting the end-to-end execution chain."""
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    execution_mode: str = "auto"
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    phases: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    telemetry: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, run_id: str, config: AppConfig) -> RunManifest:
        """Initialise a new experiment manifest."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return cls(
            run_id=run_id,
            status="running",
            started_at=now,
            execution_mode=config.execution_mode.value,
            config=config.to_dict(),
            environment=get_environment_info(),
        )

    def record_artifact(self, name: str, record: ArtifactRecord) -> None:
        """Register an artifact in the manifest."""
        self.artifacts[name] = record.to_dict()

    def record_metric(self, name: str, value: Any) -> None:
        """Record an evaluation or execution metric."""
        self.metrics[name] = value

    def record_telemetry(self, records: List[TelemetryRecord]) -> None:
        """Record telemetry with provenance classification."""
        self.telemetry = [r.to_dict() for r in records]

    def record_error(self, code: str, phase: str, message: str, remediation: Optional[str] = None) -> None:
        """Record a pipeline error."""
        self.errors.append({
            "code": code,
            "phase": phase,
            "message": message,
            "remediation": remediation,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serialisable dictionary."""
        return asdict(self)

    def save(self, output_path: Union[str, Path]) -> Path:
        """Atomically persist manifest as JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        return atomic_write(output_path, json_str)


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a run_manifest.json file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
