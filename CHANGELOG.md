# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-08

### Added
- Complete end-to-end MLOps pipeline for Ultralytics YOLO11 -> ONNX FP32 -> Hailo-8L HEF compilation targeting Raspberry Pi 5.
- Unified CLI orchestration entry point: `train_and_compile.py` supporting subcommands: `doctor`, `train`, `export`, `calibrate`, `compile`, `validate`, and `all`.
- Three-tier execution boundaries: `portable`, `hailo_compile`, and `hailo_runtime`.
- Strict pre-flight dataset validation engine checking YOLO formatting, labels, dimensions, duplicate hashing, and train/val split leakage.
- Deterministic calibration dataset selector and preprocessor with manifest and statistics generation.
- Full ONNX graph validation, static shape contract (`1x3x640x640`), dynamic batching checks, and ONNX Runtime smoke testing.
- Numerical validation engine comparing PyTorch FP32 against ONNX Runtime FP32.
- Hailo Dataflow Compiler (DFC) abstraction layer supporting ONNX -> HAR translation, INT8 Post-Training Quantization (PTQ), and Hailo-8L compilation.
- HailoRT runtime validation and smoke testing on physical Hailo-8L AI HAT+ on Raspberry Pi 5.
- Automated quantitative accuracy evaluation (mAP50, mAP50-95, per-class metrics) with configurable degradation gates (`map50_max_drop`, `map5095_max_drop`).
- Hardware telemetry monitoring (CPU, memory, storage, thermal zones, Raspberry Pi throttling, Hailo accelerator metrics) with strict provenance classification (`MEASURED`, `DERIVED`, `CONFIGURED`, `UNAVAILABLE`).
- Experiment manifest generation (`run_manifest.json`) and Markdown validation report (`run_report.md`).
- Rootless Podman and Docker Compose support (`Containerfile`, `compose.yaml`).
- Comprehensive pytest test suite with unit, integration, mock, and hardware test suites.
- Full architectural and workflow documentation in `docs/`.
