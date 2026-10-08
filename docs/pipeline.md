# Pipeline Execution Guide

## Overview

The `train_and_compile.py` CLI provides both atomic subcommand execution for individual development workflows and complete end-to-end pipeline execution with `all`.

## Execution Commands

### 1. Doctor (`doctor`)
Runs comprehensive pre-flight diagnostics across software packages, Python modules, Hailo DFC tools, HailoRT drivers, and hardware device nodes.

```bash
python train_and_compile.py doctor
```

### 2. Dataset Validation & Training (`train`)
Executes Phase A (dataset pre-flight validation) and Phase B (YOLO11 training or checkpoint validation).

```bash
python train_and_compile.py train --config config/config.yaml --epochs 50 --batch 16
```
- Validates YOLO label formats, coordinate ranges, class boundaries, duplicate images, and data leakage.
- Preserves checkpoint to `artifacts/runs/<run_id>/pytorch/best.pt` and canonical `artifacts/pytorch/best.pt`.

### 3. Static FP32 ONNX Export (`export`)
Converts `best.pt` to a fixed-dimensional ONNX graph (`1x3x640x640`).

```bash
python train_and_compile.py export
```
- Disables dynamic batching and dynamic spatial axes (mandatory for Hailo-8L).
- Applies `onnx-simplifier`.
- Runs ONNX Runtime CPU smoke inference to verify absence of NaN or Inf values.
- Executes numerical validation against the PyTorch baseline.

### 4. Calibration Dataset Generation (`calibrate`)
Extracts a deterministic calibration dataset for Hailo INT8 Post-Training Quantization (PTQ).

```bash
python train_and_compile.py calibrate --seed 42
```
- Deterministically samples 200 uncorrupted images from the training split.
- Letterboxes images to `640x640` with pad value 114.
- Generates `calib_data.npy`, `calibration_manifest.json`, and `calibration_statistics.json`.

### 5. Hailo Compilation (`compile`)
Executes the Hailo compilation chain on hosts with Hailo Dataflow Compiler (DFC) installed.

```bash
python train_and_compile.py compile
```
- Translates ONNX to Hailo Archive (HAR) via `hailo_sdk_client.ClientRunner` or `hailo parser onnx`.
- Generates `yolo11.alls` specifying RGB normalization: `[0.0, 0.0, 0.0]` to `[255.0, 255.0, 255.0]`.
- Runs INT8 PTQ using `calib_data.npy`.
- Compiles optimized HAR to Hailo Executable Format (`model.hef`) targeting `hailo8l`.

### 6. Validation & Benchmarks (`validate`)
Evaluates accuracy metrics, accuracy degradation gates, and runs latency benchmarks.

```bash
python train_and_compile.py validate
```
- Computes mAP50 and mAP50-95.
- Enforces degradation gates: fails validation if drop exceeds `map50_max_drop` (default 2%).
- Measures inference latency (mean, median, p50, p95, p99) and throughput (FPS).
- Executes HailoRT smoke test if physical Hailo-8L accelerator is detected.

### 7. End-to-End Execution (`all`)
Runs the complete dependency-aware pipeline:

```bash
python train_and_compile.py all
```
- Preserves experiment provenance in `artifacts/runs/<run_id>/run_manifest.json`.
- Outputs human-readable validation report `run_report.md`.
- Updates `artifacts/latest` pointer.
