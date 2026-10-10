"""Command-Line Interface (CLI) orchestration and pipeline execution engine."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yolo_hailo_mlops.checksums import create_artifact_record
from yolo_hailo_mlops.config import AppConfig, load_config
from yolo_hailo_mlops.exceptions import ConfigurationError, YoloHailoError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment
from yolo_hailo_mlops.logging import PipelineLogger, get_logger
from yolo_hailo_mlops.manifest import RunManifest
from yolo_hailo_mlops.provenance import get_dependency_versions, get_environment_info
from yolo_hailo_mlops.state import ExecutionMode, PhaseName, PhaseState, PipelineStatus, StateMachine
from yolo_hailo_mlops.utils.filesystem import atomic_write, ensure_dir, safe_symlink_or_copy


def generate_run_id() -> str:
    """Generate deterministic timestamped run identifier: YYYYMMDD-HHMMSS-<hash>."""
    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d-%H%M%S")
    rnd = hashlib.sha256(f"{time.time()}_{os.getpid()}".encode()).hexdigest()[:6]
    return f"{ts}-{rnd}"


class PipelineRunner:
    """End-to-end pipeline execution controller."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.run_id = config.run_id or generate_run_id()
        self.config.run_id = self.run_id

        # Setup paths
        self.root_artifacts = Path(self.config.artifacts.root).resolve()
        self.run_dir = ensure_dir(self.root_artifacts / "runs" / self.run_id)

        # Logger
        self.logger = get_logger(run_id=self.run_id, log_dir=self.run_dir, verbose=self.config.verbose)

        # State machine & Manifest
        self.sm = StateMachine(execution_mode=self.config.execution_mode)
        self.manifest = RunManifest.create(run_id=self.run_id, config=self.config)

        # Setup signal handlers for graceful cancellation
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.warning(f"Received {sig_name}. Gracefully cancelling pipeline...", phase="cancellation")
        self.sm.cancel_pipeline(reason=f"Interrupted by {sig_name}")
        self.manifest.status = "cancelled"
        self.manifest.phases = self.sm.to_dict()["phases"]
        self.manifest.save(self.run_dir / "run_manifest.json")
        sys.exit(130)

    # --------------------------------------------------------------------------
    # Subcommands
    # --------------------------------------------------------------------------

    def run_doctor(self) -> int:
        """Execute environment diagnostics."""
        print("================================================================================")
        print(" hailo-yolo11-compiler Environment Doctor")
        print("================================================================================")

        env_info = get_environment_info()
        deps = env_info["dependencies"]
        hailo_env = detect_hailo_environment()

        status_map = {
            "available": "PASS",
            "unavailable": "NOT AVAILABLE",
        }

        print(f"\n[HOST PLATFORM]")
        print(f"  OS:                 {env_info['os']} {env_info['os_release']} ({env_info['architecture']})")
        print(f"  Python:             {env_info['python_version'].split()[0]} ({env_info['python_executable']})")

        print(f"\n[CORE ML DEPENDENCIES]")
        for pkg, label in [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("ultralytics", "Ultralytics"),
            ("onnx", "ONNX"),
            ("onnxruntime", "ONNX Runtime"),
            ("pyyaml", "PyYAML"),
            ("pillow", "Pillow"),
            ("numpy", "NumPy"),
        ]:
            info = deps.get(pkg, {})
            st = status_map.get(info.get("status"), "FAIL")
            ver = info.get("version") or info.get("reason", "missing")
            print(f"  {label:<18} [{st:<13}] {ver}")

        print(f"\n[HAILO ACCELERATOR & COMPILER]")
        dfc_st = "PASS" if hailo_env.has_dfc else "NOT AVAILABLE"
        print(f"  Hailo DFC:          [{dfc_st:<13}] {hailo_env.dfc_mode} (v{hailo_env.dfc_version or 'N/A'})")

        hrt_st = "PASS" if hailo_env.has_hailort else "NOT AVAILABLE"
        print(f"  HailoRT:            [{hrt_st:<13}] {hailo_env.hailort_mode} (v{hailo_env.hailort_version or 'N/A'})")

        hw_st = "PASS" if hailo_env.has_hardware else "NOT AVAILABLE"
        print(f"  Hailo-8L Hardware:  [{hw_st:<13}] {hailo_env.device_name or 'Not connected (Raspberry Pi 5 AI HAT+)'}")

        # Dataset & Model check
        print(f"\n[PROJECT CONFIGURATION]")
        model_exists = Path(self.config.model.source).exists()
        model_st = "PASS" if model_exists else "WARN"
        print(f"  Model Source:       [{model_st:<13}] {self.config.model.source}")

        ds_exists = Path(self.config.dataset.yaml).exists()
        ds_st = "PASS" if ds_exists else "WARN"
        print(f"  Dataset YAML:       [{ds_st:<13}] {self.config.dataset.yaml}")

        print("\n================================================================================")
        print(" SUMMARY DIAGNOSTIC:")
        print(f" {hailo_env.diagnostic}")

        missing_core = [
            label
            for pkg, label in [
                ("numpy", "NumPy"),
                ("pyyaml", "PyYAML"),
                ("pillow", "Pillow"),
            ]
            if deps.get(pkg, {}).get("status") != "available"
        ]
        if missing_core:
            print(f"\n WARNING: Foundation packages missing: {', '.join(missing_core)}.")
            print(" To install foundation dependencies, run:")
            print("   pip install -e \".[dev]\"")
            print(" Or run the automated bootstrapper:")
            print("   ./scripts/bootstrap.sh")
        print("================================================================================")
        return 0

    def run_dry_run(self) -> int:
        """Show resolved execution plan without modifying production artifacts."""
        hailo_env = detect_hailo_environment()

        print("================================================================================")
        print(" hailo-yolo11-compiler Execution Plan (DRY RUN)")
        print("================================================================================")
        print(f"Run ID:             {self.run_id}")
        print(f"Execution Mode:     {self.config.execution_mode.value.upper()}")
        print(f"Target Accelerator: {self.config.hailo.target} (Raspberry Pi 5 AI HAT+)")
        print(f"Model / Task:       {self.config.model.source} ({self.config.model.task})")
        print(f"Dataset YAML:       {self.config.dataset.yaml}")
        print(f"Calibration Count:  {self.config.dataset.calibration.count} (Seed: {self.config.dataset.calibration.seed})")

        print("\nPlanned Stages:")
        stages = [
            ("1. Dataset Pre-flight Validation", "Portable", "Validates YOLO syntax, labels, leakage, dimensions"),
            ("2. YOLO11 Training / Checkpoint",  "Portable", "Produces/validates best.pt"),
            ("3. FP32 ONNX Export",              "Portable", "Exports 1x3x640x640 static FP32 model.onnx"),
            ("4. ONNX Validation & Smoke Test",  "Portable", "Validates ONNX graph and runs smoke inference"),
            ("5. Calibration Set Generation",   "Portable", "Samples & letterboxes 200 images, writes calib_data.npy"),
            ("6. ONNX -> HAR Translation",       "Hailo DFC", "Parses ONNX and creates Hailo Archive with .alls script"),
            ("7. INT8 Post-Training Quantisation","Hailo DFC", "Quantises HAR using calibration dataset"),
            ("8. Hailo-8L Compilation",          "Hailo DFC", "Compiles optimised HAR to model.hef"),
            ("9. Accuracy Validation Gates",     "Portable/Hailo", "Evaluates mAP50 and mAP50-95 degradation"),
            ("10. Physical HailoRT Smoke Test",  "Hailo Runtime", "Runs inference on Raspberry Pi 5 Hailo-8L AI HAT+"),
        ]
        for name, boundary, desc in stages:
            print(f"  {name:<36} [{boundary:<14}] {desc}")

        print("\nHardware & Software Capability Assessment:")
        if hailo_env.has_dfc:
            print("  Hailo DFC:          READY for stages 6-8.")
        else:
            print("  Hailo DFC:          NOT DETECTED (Stages 6-8 will be marked NOT_EXECUTED in portable mode).")

        if hailo_env.has_hardware:
            print("  Hailo-8L Hardware:  CONNECTED (Stage 10 ready).")
        else:
            print("  Hailo-8L Hardware:  NOT DETECTED (Stage 10 will report HARDWARE_VALIDATION=NOT_EXECUTED).")

        print("================================================================================")
        return 0

    def run_dataset(self) -> int:
        """Execute Phase A (Dataset validation)."""
        try:
            from yolo_hailo_mlops.dataset.validator import DatasetValidator
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for dataset validation: {e}. Run 'pip install -e \".[dev]\"' or './scripts/bootstrap.sh'.",
                code="E-DATA-001",
                phase="dataset_validation",
                remediation="Install dataset validation dependencies with: pip install -e \".[dev]\"",
            ) from e

        self.logger.info("Starting Dataset Pre-flight Validation...", phase="dataset_validation")
        self.sm.start_phase(PhaseName.DATASET_VALIDATION)
        validator = DatasetValidator(self.config.dataset.yaml)
        report = validator.validate(fail_fast=True)
        self.sm.complete_phase(PhaseName.DATASET_VALIDATION, metadata=report.to_dict())
        self.logger.info("Dataset validation passed successfully.", phase="dataset_validation")
        total_images = report.train_stats.num_images + report.val_stats.num_images + (report.test_stats.num_images if report.test_stats else 0)
        total_labels = report.train_stats.num_labels + report.val_stats.num_labels + (report.test_stats.num_labels if report.test_stats else 0)
        print("\n" + "=" * 80)
        print(" DATASET PRE-FLIGHT VALIDATION: SUCCESS")
        print(f" Images: {total_images:,} | Labels: {total_labels:,} | Classes: {report.num_classes}")
        print("=" * 80)
        return 0

    def run_train(self) -> Path:
        """Execute Phase A (Dataset validation) and Phase B (Training)."""
        try:
            from yolo_hailo_mlops.dataset.validator import DatasetValidator
            from yolo_hailo_mlops.training.trainer import YOLOTrainer
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for training: {e}. Run 'pip install -e \".[training]\"' or './scripts/bootstrap.sh'.",
                code="E-TRAIN-007",
                phase="training",
                remediation="Install training dependencies with: pip install -e \".[training]\"",
            ) from e

        # Phase A: Dataset validation
        self.logger.info("Starting Dataset Pre-flight Validation...", phase="dataset_validation")
        self.sm.start_phase(PhaseName.DATASET_VALIDATION)
        validator = DatasetValidator(self.config.dataset.yaml)
        report = validator.validate(fail_fast=True)
        self.sm.complete_phase(PhaseName.DATASET_VALIDATION, metadata=report.to_dict())
        self.logger.info("Dataset validation passed successfully.", phase="dataset_validation")

        # Phase B: Training
        self.logger.info("Starting YOLO11 training / checkpoint validation...", phase="training")
        self.sm.start_phase(PhaseName.TRAINING)
        trainer = YOLOTrainer(self.config, self.run_dir)
        ckpt_meta = trainer.run()
        best_pt = Path(ckpt_meta.path)

        rec = create_artifact_record(best_pt, artifact_type="pytorch_checkpoint", producer_phase="training")
        self.manifest.record_artifact("best_pt", rec)
        self.sm.complete_phase(PhaseName.TRAINING, artifacts=[str(best_pt)], metadata=ckpt_meta.to_dict())
        self.logger.info(f"Training completed. Checkpoint saved: {best_pt}", phase="training")
        return best_pt

    def run_export(self, checkpoint_path: Optional[Path] = None) -> Path:
        """Execute Phase C (FP32 ONNX Export) and validation."""
        try:
            from yolo_hailo_mlops.export.onnx_exporter import ONNXExporter
            from yolo_hailo_mlops.evaluation.onnx import compare_pytorch_onnx_numerical
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for ONNX export: {e}. Run 'pip install -e \".[onnx]\"' or './scripts/bootstrap.sh'.",
                code="E-EXP-004",
                phase="onnx_export",
                remediation="Install ONNX dependencies with: pip install -e \".[onnx]\"",
            ) from e

        ckpt = checkpoint_path or (self.run_dir / "pytorch" / "best.pt")
        if not ckpt.is_file():
            ckpt = Path(self.root_artifacts / "pytorch" / "best.pt")
        if not ckpt.is_file():
            ckpt = Path(self.config.model.source)

        self.logger.info("Starting static FP32 ONNX export...", phase="onnx_export")
        self.sm.start_phase(PhaseName.ONNX_EXPORT)
        exporter = ONNXExporter(self.config, self.run_dir)
        onnx_report = exporter.export(ckpt)
        onnx_path = Path(onnx_report.path)

        rec = create_artifact_record(onnx_path, artifact_type="onnx_model", producer_phase="onnx_export")
        self.manifest.record_artifact("model_onnx", rec)
        self.sm.complete_phase(PhaseName.ONNX_EXPORT, artifacts=[str(onnx_path)], metadata=onnx_report.to_dict())

        # Numerical comparison
        self.logger.info("Running PyTorch vs ONNX numerical validation...", phase="numerical_validation")
        self.sm.start_phase(PhaseName.NUMERICAL_VALIDATION)
        num_report = compare_pytorch_onnx_numerical(ckpt, onnx_path, imgsz=self.config.export.imgsz)
        self.sm.complete_phase(PhaseName.NUMERICAL_VALIDATION, metadata=num_report.to_dict())
        self.logger.info(f"Numerical validation status: {num_report.status}", phase="numerical_validation")

        return onnx_path

    def run_calibrate(self) -> Tuple[Path, Path]:
        """Execute Phase D (Deterministic Calibration Dataset generation)."""
        try:
            from yolo_hailo_mlops.calibration.selector import CalibrationSelector
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for calibration: {e}. Run 'pip install -e \".[dev]\"' or './scripts/bootstrap.sh'.",
                code="E-CAL-003",
                phase="calibration",
                remediation="Install calibration dependencies with: pip install -e \".[dev]\"",
            ) from e

        self.logger.info("Starting deterministic calibration sampling...", phase="calibration")
        self.sm.start_phase(PhaseName.CALIBRATION)
        selector = CalibrationSelector(self.config, self.run_dir)
        npy_path, manifest_path = selector.generate()

        rec_npy = create_artifact_record(npy_path, artifact_type="calibration_dataset", producer_phase="calibration")
        rec_man = create_artifact_record(manifest_path, artifact_type="calibration_manifest", producer_phase="calibration")
        self.manifest.record_artifact("calib_data_npy", rec_npy)
        self.manifest.record_artifact("calibration_manifest", rec_man)

        self.sm.complete_phase(
            PhaseName.CALIBRATION,
            artifacts=[str(npy_path), str(manifest_path)],
            metadata={"sample_count": self.config.dataset.calibration.count},
        )
        self.logger.info(f"Calibration dataset ready: {npy_path}", phase="calibration")
        return npy_path, manifest_path

    def run_compile(
        self,
        onnx_path: Optional[Path] = None,
        calib_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Execute Phases F, G, H (Hailo parse, INT8 PTQ, Hailo-8L compilation)."""
        env = detect_hailo_environment()
        if not env.has_dfc:
            reason = "Hailo DFC not available in this environment."
            self.logger.warning(f"Skipping Hailo compilation: {reason}", phase="hailo_compile")
            self.sm.mark_not_executed(PhaseName.HAILO_PARSE, reason)
            self.sm.mark_not_executed(PhaseName.HAILO_OPTIMISATION, reason)
            self.sm.mark_not_executed(PhaseName.HAILO_COMPILE, reason)
            return None

        try:
            from yolo_hailo_mlops.hailo.compiler import HailoCompiler
            from yolo_hailo_mlops.hailo.optimizer import HailoOptimizer
            from yolo_hailo_mlops.hailo.parser import HailoParser
            from yolo_hailo_mlops.hailo.validator import validate_hef
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for Hailo compilation: {e}. Run 'pip install -e \".[dev]\"' or './scripts/bootstrap.sh'.",
                code="E-COMP-001",
                phase="hailo_compile",
                remediation="Install compilation dependencies with: pip install -e \".[dev]\"",
            ) from e

        # 1. Parse
        onnx_p = onnx_path or (self.run_dir / "onnx" / "model.onnx")
        if not onnx_p.is_file():
            onnx_p = Path(self.root_artifacts / "onnx" / "model.onnx")

        self.logger.info("Starting Hailo ONNX -> HAR parsing...", phase="hailo_parse")
        self.sm.start_phase(PhaseName.HAILO_PARSE)
        parser = HailoParser(self.config, self.run_dir)
        har_path = parser.parse(onnx_p)
        rec_har = create_artifact_record(har_path, artifact_type="hailo_archive", producer_phase="hailo_parse")
        self.manifest.record_artifact("model_har", rec_har)
        self.sm.complete_phase(PhaseName.HAILO_PARSE, artifacts=[str(har_path)])

        # 2. Optimise
        calib_p = calib_path or (self.run_dir / "calibration" / "calib_data.npy")
        if not calib_p.is_file():
            calib_p = Path(self.root_artifacts / "calibration" / "calib_data.npy")

        self.logger.info("Starting Hailo INT8 Post-Training Quantisation...", phase="hailo_optimisation")
        self.sm.start_phase(PhaseName.HAILO_OPTIMISATION)
        optimizer = HailoOptimizer(self.config, self.run_dir)
        quant_har = optimizer.optimize(har_path, calib_p)
        rec_qhar = create_artifact_record(quant_har, artifact_type="quantized_har", producer_phase="hailo_optimisation")
        self.manifest.record_artifact("model_quantized_har", rec_qhar)
        self.sm.complete_phase(PhaseName.HAILO_OPTIMISATION, artifacts=[str(quant_har)])

        # 3. Compile
        self.logger.info("Compiling for Hailo-8L target architecture...", phase="hailo_compile")
        self.sm.start_phase(PhaseName.HAILO_COMPILE)
        compiler = HailoCompiler(self.config, self.run_dir)
        hef_path = compiler.compile(quant_har)

        # Validate HEF
        hef_report = validate_hef(hef_path, expected_arch=self.config.hailo.target)
        rec_hef = create_artifact_record(hef_path, artifact_type="hailo_hef", producer_phase="hailo_compile")
        self.manifest.record_artifact("model_hef", rec_hef)
        self.sm.complete_phase(PhaseName.HAILO_COMPILE, artifacts=[str(hef_path)], metadata=hef_report.to_dict())
        self.logger.info(f"Hailo-8L HEF compilation successful: {hef_path}", phase="hailo_compile")

        return hef_path

    def run_validate(
        self,
        checkpoint_path: Optional[Path] = None,
        onnx_path: Optional[Path] = None,
        hef_path: Optional[Path] = None,
    ) -> None:
        """Execute Phase I (Quantitative Accuracy) and Phase J (Runtime smoke test)."""
        try:
            from yolo_hailo_mlops.evaluation.comparison import build_evaluation_report
            from yolo_hailo_mlops.evaluation.onnx import evaluate_onnx_model
            from yolo_hailo_mlops.evaluation.pytorch import evaluate_pytorch_model
            from yolo_hailo_mlops.hailo.runtime import HailoRuntimeValidator
            from yolo_hailo_mlops.performance.latency import benchmark_onnx_latency
            from yolo_hailo_mlops.performance.resources import collect_performance_report
            from yolo_hailo_mlops.performance.throughput import calculate_throughput_from_latency
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Missing dependency for validation: {e}. Run 'pip install -e \".[training,onnx]\"' or './scripts/bootstrap.sh'.",
                code="E-ACC-002",
                phase="accuracy_validation",
                remediation="Install validation dependencies with: pip install -e \".[training,onnx]\"",
            ) from e

        ckpt = checkpoint_path or (self.run_dir / "pytorch" / "best.pt")
        if not ckpt.is_file():
            ckpt = Path(self.root_artifacts / "pytorch" / "best.pt")

        onnx_p = onnx_path or (self.run_dir / "onnx" / "model.onnx")
        if not onnx_p.is_file():
            onnx_p = Path(self.root_artifacts / "onnx" / "model.onnx")

        hef_p = hef_path or (self.run_dir / "hailo" / "model.hef")
        if not hef_p.is_file():
            hef_p = Path(self.root_artifacts / "hailo" / "model.hef")

        # 1. Accuracy validation
        self.logger.info("Evaluating model accuracy & degradation gates...", phase="accuracy_validation")
        self.sm.start_phase(PhaseName.ACCURACY_VALIDATION)

        pyt_metrics = None
        if ckpt.is_file():
            try:
                pyt_metrics = evaluate_pytorch_model(ckpt, self.config)
            except Exception as e:
                self.logger.warning(f"PyTorch evaluation skipped: {e}", phase="accuracy_validation")

        onnx_metrics = None
        if onnx_p.is_file():
            onnx_metrics = evaluate_onnx_model(onnx_p, self.config, baseline_metrics=pyt_metrics)

        eval_report = build_evaluation_report(
            pytorch_metrics=pyt_metrics,
            onnx_metrics=onnx_metrics,
            hailo_metrics=None,
            numerical_comparison=None,
            config=self.config,
        )
        self.sm.complete_phase(PhaseName.ACCURACY_VALIDATION, metadata=eval_report.to_dict())
        self.manifest.record_metric("accuracy", eval_report.to_dict())

        # 2. Performance benchmark
        self.logger.info("Running performance and latency benchmarks...", phase="performance_validation")
        self.sm.start_phase(PhaseName.PERFORMANCE_VALIDATION)
        onnx_lat = None
        if onnx_p.is_file():
            onnx_lat = benchmark_onnx_latency(
                onnx_p,
                warmup_iterations=self.config.validation.performance.warmup_iterations,
                measurement_iterations=self.config.validation.performance.measurement_iterations,
                imgsz=self.config.export.imgsz,
            )
        onnx_fps = calculate_throughput_from_latency(onnx_lat)
        perf_report = collect_performance_report(onnx_latency=onnx_lat, onnx_throughput=onnx_fps)
        self.manifest.record_telemetry(perf_report.telemetry)
        self.manifest.record_metric("performance", perf_report.to_dict())
        self.sm.complete_phase(PhaseName.PERFORMANCE_VALIDATION, metadata=perf_report.to_dict())

        # 3. Hailo Runtime Smoke Test
        self.logger.info("Running Hailo runtime smoke test...", phase="hailo_validation")
        self.sm.start_phase(PhaseName.HAILO_VALIDATION)
        runtime_validator = HailoRuntimeValidator(require_hardware=self.config.hailo.require_hardware)
        if hef_p.is_file():
            rt_report = runtime_validator.run_smoke_test(hef_p)
            if rt_report.status == "SUCCESS":
                self.sm.complete_phase(PhaseName.HAILO_VALIDATION, metadata=rt_report.to_dict())
            elif rt_report.status == "NOT_EXECUTED":
                self.sm.mark_not_executed(PhaseName.HAILO_VALIDATION, reason=rt_report.reason or "Hardware absent")
            else:
                self.sm.fail_phase(PhaseName.HAILO_VALIDATION, "E-HAILO-RT-001", rt_report.reason or "Failed")
        else:
            self.sm.mark_not_executed(PhaseName.HAILO_VALIDATION, reason="No HEF artifact available to test")

    def run_all(self) -> int:
        """Run complete end-to-end dependency-aware pipeline."""
        self.logger.info(f"Starting complete pipeline run (Run ID: {self.run_id})")

        # Step 1: Train
        best_pt = self.run_train()

        # Step 2: Export
        onnx_path = self.run_export(best_pt)

        # Step 3: Calibrate
        calib_npy, calib_man = self.run_calibrate()

        # Step 4: Compile
        hef_path = self.run_compile(onnx_path, calib_npy)

        # Step 5: Validate
        self.run_validate(best_pt, onnx_path, hef_path)

        # Conclude and Report
        status = self.sm.finish()
        self.manifest.status = status.value
        self.manifest.phases = self.sm.to_dict()["phases"]
        self.manifest.completed_at = self.sm.completed_at
        self.manifest.duration_seconds = self.sm.to_dict()["duration_seconds"]

        # Save latest pointer
        latest_link = self.root_artifacts / "latest"
        safe_symlink_or_copy(self.run_dir, latest_link)

        # Generate final reports
        self.generate_reports()

        print(f"\n================================================================================")
        print(f" PIPELINE STATUS: {status.value}")
        print(f" Run Directory:   {self.run_dir}")
        print(f" Manifest:        {self.run_dir / 'run_manifest.json'}")
        print(f" Report:          {self.run_dir / 'run_report.md'}")
        print(f"================================================================================\n")

        return 0 if status in (PipelineStatus.SUCCESS, PipelineStatus.PARTIAL) else 1

    def generate_reports(self) -> None:
        """Generate machine-readable JSON report and human-readable Markdown validation report."""
        manifest_path = self.run_dir / "run_manifest.json"
        self.manifest.save(manifest_path)

        report_json_path = self.run_dir / "run_report.json"
        atomic_write(report_json_path, json.dumps(self.manifest.to_dict(), indent=2, default=str))

        # Markdown Report (Section 42)
        status_val = self.manifest.status
        md_lines = [
            "# Hailo-8L YOLO11 Pipeline Validation Report",
            "",
            f"**Experiment Run ID:** `{self.run_id}`  ",
            f"**Execution Mode:** `{self.config.execution_mode.value.upper()}`  ",
            f"**Pipeline Status:** **`{status_val}`**  ",
            f"**Timestamp:** `{self.manifest.started_at}`  ",
            f"**Duration:** `{self.manifest.duration_seconds}s`  ",
            "",
            "---",
            "",
            "## 1. Execution Summary",
            "",
            "| Phase | Status | Duration (s) | Error / Reason |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for p_name, p_data in self.manifest.phases.items():
            st = p_data.get("state", "UNKNOWN")
            dur = p_data.get("duration_seconds", "-")
            err = p_data.get("reason") or p_data.get("error_message") or "-"
            md_lines.append(f"| `{p_name}` | **{st}** | {dur} | {err} |")

        md_lines.extend([
            "",
            "## 2. Model & Checkpoints",
            "",
            f"- **Base Model:** `{self.config.model.source}`",
            f"- **Task:** `{self.config.model.task}`",
            f"- **Variant:** `{self.config.model.variant}`",
            f"- **Export Precision:** `{self.config.export.precision}` (Static: `1x3x{self.config.export.imgsz}x{self.config.export.imgsz}`)",
            "",
            "## 3. Accuracy Gates & Evaluation",
            "",
        ])

        acc = self.manifest.metrics.get("accuracy", {})
        gate = acc.get("accuracy_gate", {})
        md_lines.extend([
            f"- **Gate Status:** `{gate.get('status', 'N/A')}`",
            f"- **mAP50 Drop:** `{gate.get('map50_drop', 'N/A')}` (Max Allowed: `{gate.get('map50_threshold', 'N/A')}`)",
            f"- **mAP50-95 Drop:** `{gate.get('map5095_drop', 'N/A')}` (Max Allowed: `{gate.get('map5095_threshold', 'N/A')}`)",
            "",
            "## 4. Hardware Telemetry & Telemetry Provenance",
            "",
            "| Metric | Value | Unit | Classification | Source |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for t in self.manifest.telemetry:
            val_str = str(t.get("value")) if t.get("value") is not None else "N/A"
            md_lines.append(
                f"| `{t.get('metric')}` | {val_str} | {t.get('unit')} | `{t.get('classification')}` | `{t.get('source')}` |"
            )

        md_lines.extend([
            "",
            "## 5. Artifact Provenance & SHA-256",
            "",
            "| Artifact Name | Type | Size (bytes) | SHA-256 Hash |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for art_name, art_rec in self.manifest.artifacts.items():
            md_lines.append(
                f"| `{art_name}` | {art_rec.get('type')} | {art_rec.get('size_bytes'):,} | `{art_rec.get('sha256')[:16]}...` |"
            )

        md_lines.extend([
            "",
            "---",
            f"**PIPELINE STATUS: {status_val}**",
            "",
        ])

        report_md_path = self.run_dir / "run_report.md"
        atomic_write(report_md_path, "\n".join(md_lines))


def main() -> None:
    """CLI entry point parsing arguments and directing execution flow."""
    parser = argparse.ArgumentParser(
        prog="train_and_compile",
        description="Reproducible YOLO11 -> ONNX -> Hailo-8L MLOps Compiler & Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("command", choices=["doctor", "dataset", "train", "export", "calibrate", "compile", "validate", "all"],
                        help="Pipeline phase to execute")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Explicit run identifier")
    parser.add_argument("--model", type=str, default=None,
                        help="Source YOLO11 weights or variant (e.g. yolo11n.pt)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to YOLO dataset YAML file")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch compute device ('auto', 'cpu', '0')")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image resolution (square, default 640)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Training mini-batch size")
    parser.add_argument("--seed", type=int, default=None,
                        help="Deterministic random seed")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force regeneration of existing artifacts")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume pipeline from existing artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Enable verbose debug logging")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Show execution plan without modifying artifacts")
    parser.add_argument("--execution-mode", choices=["auto", "portable", "hailo_compile", "hailo_runtime"],
                        default="auto", help="Boundary execution mode")

    args = parser.parse_args()

    cli_overrides = {
        "model": args.model,
        "dataset": args.dataset,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "run_id": args.run_id,
        "force": args.force,
        "resume": args.resume,
        "verbose": args.verbose,
        "dry_run": args.dry_run,
        "execution_mode": args.execution_mode,
    }

    try:
        cfg = load_config(config_path=args.config, cli_overrides=cli_overrides)
        runner = PipelineRunner(config=cfg)

        if args.dry_run:
            sys.exit(runner.run_dry_run())

        cmd = args.command
        if cmd == "doctor":
            sys.exit(runner.run_doctor())
        elif cmd == "dataset":
            sys.exit(runner.run_dataset())
        elif cmd == "train":
            runner.run_train()
            runner.generate_reports()
        elif cmd == "export":
            runner.run_export()
            runner.generate_reports()
        elif cmd == "calibrate":
            runner.run_calibrate()
            runner.generate_reports()
        elif cmd == "compile":
            runner.run_compile()
            runner.generate_reports()
        elif cmd == "validate":
            runner.run_validate()
            runner.generate_reports()
        elif cmd == "all":
            sys.exit(runner.run_all())

    except YoloHailoError as e:
        print(f"\n{e.format_diagnostic()}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
