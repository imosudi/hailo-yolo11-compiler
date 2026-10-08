"""Integration tests for end-to-end pipeline runner orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.mocks.hailo_mocks import create_mock_hailo_environment
from yolo_hailo_mlops.cli import PipelineRunner
from yolo_hailo_mlops.config import AppConfig


@pytest.mark.integration
def test_pipeline_dry_run_orchestration(sample_config: AppConfig) -> None:
    """Verify that PipelineRunner dry-run runs cleanly without generating output files."""
    runner = PipelineRunner(sample_config)
    exit_code = runner.run_dry_run()
    assert exit_code == 0


@pytest.mark.integration
def test_pipeline_doctor_orchestration(sample_config: AppConfig) -> None:
    """Verify that PipelineRunner doctor runs cleanly."""
    runner = PipelineRunner(sample_config)
    exit_code = runner.run_doctor()
    assert exit_code == 0


@pytest.mark.integration
def test_pipeline_calibration_phase(sample_config: AppConfig, temp_workspace: Path) -> None:
    """Verify standalone calibration execution generates artifacts and updates manifest."""
    runner = PipelineRunner(sample_config)
    npy_path, man_path = runner.run_calibrate()

    assert npy_path.is_file()
    assert man_path.is_file()
    assert "calib_data_npy" in runner.manifest.artifacts
    assert "calibration_manifest" in runner.manifest.artifacts


@pytest.mark.integration
def test_report_generation(sample_config: AppConfig, temp_workspace: Path) -> None:
    """Verify that generate_reports creates run_manifest.json, run_report.json, and run_report.md."""
    runner = PipelineRunner(sample_config)
    runner.sm.finish()
    runner.generate_reports()

    run_dir = runner.run_dir
    manifest_p = run_dir / "run_manifest.json"
    report_json_p = run_dir / "run_report.json"
    report_md_p = run_dir / "run_report.md"

    assert manifest_p.is_file()
    assert report_json_p.is_file()
    assert report_md_p.is_file()

    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == runner.run_id

    md_content = report_md_p.read_text(encoding="utf-8")
    assert "# Hailo-8L YOLO11 Pipeline Validation Report" in md_content
    assert runner.run_id in md_content
