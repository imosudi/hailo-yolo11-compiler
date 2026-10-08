"""Unit tests for pipeline state machine and run manifest tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_hailo_mlops.checksums import create_artifact_record, verify_artifact_record
from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.exceptions import ArtifactIntegrityError
from yolo_hailo_mlops.manifest import RunManifest
from yolo_hailo_mlops.state import ExecutionMode, PhaseName, PhaseState, PipelineStatus, StateMachine


def test_state_machine_lifecycle() -> None:
    """Verify phase transitions and status calculation."""
    sm = StateMachine(execution_mode=ExecutionMode.PORTABLE)

    # Start phase
    sm.start_phase(PhaseName.DATASET_VALIDATION)
    assert sm.phases[PhaseName.DATASET_VALIDATION.value].state == PhaseState.RUNNING

    # Complete phase
    sm.complete_phase(PhaseName.DATASET_VALIDATION, artifacts=["data/dataset.yaml"])
    assert sm.phases[PhaseName.DATASET_VALIDATION.value].state == PhaseState.SUCCESS
    assert sm.phases[PhaseName.DATASET_VALIDATION.value].duration_seconds is not None

    # Skip phase
    sm.skip_phase(PhaseName.TRAINING, reason="Existing checkpoint provided")
    assert sm.phases[PhaseName.TRAINING.value].state == PhaseState.SKIPPED

    # In portable mode, successful portable phases yield SUCCESS
    status = sm.finish()
    assert status == PipelineStatus.SUCCESS


def test_state_machine_failure_propagation() -> None:
    """A failed phase results in overall pipeline failure."""
    sm = StateMachine(execution_mode=ExecutionMode.AUTO)
    sm.start_phase(PhaseName.HAILO_PARSE)
    sm.fail_phase(PhaseName.HAILO_PARSE, error_code="E-HAILO-PARSE-001", error_message="Parse error")

    assert sm.compute_pipeline_status() == PipelineStatus.FAILED


def test_artifact_record_and_verification(temp_workspace: Path) -> None:
    """Artifact record records SHA-256 and detects tampering."""
    art_file = temp_workspace / "model.onnx"
    art_file.write_bytes(b"VALID_MODEL_BYTES_12345")

    rec = create_artifact_record(art_file, artifact_type="onnx_model", producer_phase="export")
    assert rec.size_bytes == len(b"VALID_MODEL_BYTES_12345")
    assert rec.status == "valid"
    assert verify_artifact_record(rec) is True

    # Tamper with file
    art_file.write_bytes(b"TAMPERED_BYTES_ABC")
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_record(rec)


def test_manifest_serialization(sample_config: AppConfig, temp_workspace: Path) -> None:
    """Manifest records lifecycle events and serializes to valid JSON."""
    manifest = RunManifest.create(run_id="test_run_123", config=sample_config)
    manifest.record_metric("mAP50", 0.85)

    manifest_path = temp_workspace / "manifest.json"
    manifest.save(manifest_path)

    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["run_id"] == "test_run_123"
    assert data["metrics"]["mAP50"] == 0.85
    assert data["config"]["project"]["name"] == "test-yolo-hailo"
