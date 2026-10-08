"""Artifact integrity verification, SHA-256 recording, and tamper detection."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

from yolo_hailo_mlops.exceptions import ArtifactIntegrityError
from yolo_hailo_mlops.utils.hashing import compute_sha256


@dataclass
class ArtifactRecord:
    path: str
    type: str
    size_bytes: int
    sha256: str
    created_at: str
    producer_phase: str
    status: str = "valid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_artifact_record(
    path: Union[str, Path],
    artifact_type: str,
    producer_phase: str,
) -> ArtifactRecord:
    """Inspect and generate an immutable record with SHA-256 for an artifact."""
    p = Path(path).resolve()
    if not p.exists():
        raise ArtifactIntegrityError(
            message=f"Artifact does not exist: {p}",
            code="E-ART-002",
            phase=producer_phase,
            artifact=str(p),
        )

    size = p.stat().st_size
    sha = compute_sha256(p)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return ArtifactRecord(
        path=str(p),
        type=artifact_type,
        size_bytes=size,
        sha256=sha,
        created_at=now,
        producer_phase=producer_phase,
        status="valid",
    )


def verify_artifact_record(record: ArtifactRecord) -> bool:
    """Verify that an artifact's physical file matches its recorded SHA-256."""
    p = Path(record.path)
    if not p.is_file():
        raise ArtifactIntegrityError(
            message=f"Recorded artifact is missing: {record.path}",
            code="E-ART-003",
            phase=record.producer_phase,
            artifact=record.path,
        )

    current_sha = compute_sha256(p)
    if current_sha.lower() != record.sha256.lower():
        raise ArtifactIntegrityError(
            message=(
                f"Artifact SHA-256 mismatch for {record.path}.\n"
                f"Expected: {record.sha256}\n"
                f"Actual:   {current_sha}"
            ),
            code="E-ART-004",
            phase=record.producer_phase,
            artifact=record.path,
        )

    return True
