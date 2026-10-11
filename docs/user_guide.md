# Complete User Guide & CLI Reference

This manual provides an exhaustive operational walkthrough for the **`hailo-yolo11-compiler`** MLOps automation pipeline. It covers practical user workflows, command-line usage, pipeline resumption patterns, and operational flags.

---

## Table of Contents

1. [Pipeline Overview & Principles](#1-pipeline-overview--principles)
2. [Environment Preparation](#2-environment-preparation)
3. [Step-by-Step Workflows](#3-step-by-step-workflows)
   - [Workflow A: Complete End-to-End Pipeline (`all`)](#workflow-a-complete-end-to-end-pipeline-all)
   - [Workflow B: Dataset Validation Only (`dataset`)](#workflow-b-dataset-validation-only-dataset)
   - [Workflow C: Model Training Only (`train`)](#workflow-c-model-training-only-train)
   - [Workflow D: Static FP32 ONNX Export (`export`)](#workflow-d-static-fp32-onnx-export-export)
   - [Workflow E: Calibration Dataset Generation (`calibrate`)](#workflow-e-calibration-dataset-generation-calibrate)
   - [Workflow F: Hailo DFC Compilation (`compile`)](#workflow-f-hailo-dfc-compilation-compile)
   - [Workflow G: Validation & Edge Benchmarking (`validate`)](#workflow-g-validation--edge-benchmarking-validate)
4. [CLI Command-Line Reference](#4-cli-command-line-reference)
5. [Operational Modes & Execution Boundaries](#5-operational-modes--execution-boundaries)
6. [Resuming Workflows with Existing Artefacts](#6-resuming-workflows-with-existing-artefacts)
7. [Dry-Run Planning & Inspection](#7-dry-run-planning--inspection)
8. [Forced Regeneration of Artefacts](#8-forced-regeneration-of-artefacts)
9. [Interpreting Generated Reports & Manifests](#9-interpreting-generated-reports--manifests)
10. [Live Model Deployment (`deploy`)](#10-live-model-deployment-deploy)
11. [CPU Multithreading & Operations Engine (`operations.py`)](#11-cpu-multithreading--operations-engine-operationspy)
12. [Licensing, Academic Provenance & Citation](#12-licensing-academic-provenance--citation)

---

## 1. Pipeline Overview & Principles

The pipeline orchestrates ten discrete phases through an explicit state machine:

```text
[Phase A] Dataset Validation  ──► Validates annotation ranges, leakage, and corrupt images
         │
[Phase B] YOLO11 Training     ──► Trains / fine-tunes checkpoint (best.pt)
         │
[Phase C] ONNX Export         ──► Emits 1×3×640×640 static FP32 graph (model.onnx)
         │
[Phase C2] ONNX Validation    ──► Verifies static shapes, opset, and smoke inference
         │
[Phase D] Calibration         ──► Letterboxes 200 RGB samples into calib_data.npy
         │
[Phase F] ONNX → HAR          ──► Hailo parser translates ONNX and injects yolo11.alls
         │
[Phase G] INT8 PTQ            ──► Hailo optimiser computes layer-wise scale factors
         │
[Phase H] Hailo-8L Compile    ──► Routes layers, schedules dataflow, emits model.hef
         │
[Phase J] Hardware Smoke Test ──► Validates physical execution on /dev/hailo0
         │
[Phase I] Accuracy Gates      ──► Evaluates ΔmAP50 and ΔmAP50-95 degradation
         │
[Phase K] Performance Bench   ──► Records p50, p95, p99 latency percentiles and FPS
```

### Engineering Tenets

- **No Silent Fallbacks**: If Hailo hardware is missing, physical tests are classified as `NOT_EXECUTED`. The system never emulates NPUs on CPU/GPU or fabricates telemetry zeros.
- **Defensible Provenance**: Every generated file is SHA-256 hashed and recorded in `run_manifest.json`.
- **Atomic Operations**: All files are staged via temporary files (`.tmp_*`) and committed atomically using `os.fsync()` and `Path.replace()`.

---

## 2. Environment Preparation & Setup

The compiler follows a dual-environment operational model:
- **Compilation Host (x86_64)**: Runs **Python 3.10** (`venv-dfc3`) for Hailo DFC v3.34.0, YOLO11 training, ONNX export, calibration, and HEF compilation.
- **Edge Accelerator (Raspberry Pi 5 / aarch64)**: Runs native **Python 3.12** with HailoRT (`python3-hailort`), `picamera2`, `libcamera`, and live deployment.

### Host Workstation Setup (x86_64 — Python 3.10)

```bash
# 1. Create Python 3.10 virtual environment
python3.10 -m venv venv-dfc3
source venv-dfc3/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install Hailo DFC v3.34.0 wheel (obtained from Hailo Developer Zone)
pip install ./hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

# 3. Install pipeline dependencies
pip install -e ".[dev,training,onnx]"

# 4. Verify system environment readiness
python train_and_compile.py doctor
```

### Raspberry Pi 5 Edge Setup (aarch64 — Python 3.12)

```bash
# 1. Install system HailoRT and camera libraries
sudo apt update && sudo apt install -y python3-hailort python3-picamera2 libcamera-apps

# 2. Create Python 3.12 virtual environment with system packages
python3 -m venv --system-site-packages venv-dfc3
source venv-dfc3/bin/activate

# 3. Install runtime dependencies
pip install -e ".[onnx]"

# 4. Verify physical accelerator and camera detection
python train_and_compile.py doctor
```

---

## 3. Step-by-Step Workflows

### Workflow A: Complete End-to-End Pipeline (`all`)

Executes the entire lifecycle from dataset validation through edge performance benchmarking:

```bash
python train_and_compile.py all --config config/config.yaml
```

**What occurs:**
1. Validates `data/dataset.yaml` and training/validation image-label pairs.
2. Fine-tunes YOLO11 (or verifies existing base weights) and emits `best.pt`.
3. Exports a static $1 \times 3 \times 640 \times 640$ FP32 ONNX graph.
4. Validates the ONNX graph with ONNX Runtime CPU.
5. Deterministically samples and letterboxes 200 images into `calib_data.npy`.
6. Invokes Hailo DFC parser to translate ONNX to HAR with `yolo11.alls`.
7. Performs INT8 Post-Training Quantisation using `calib_data.npy`.
8. Compiles the quantised HAR into `model.hef` targeting `hailo8l`.
9. Conducts HailoRT physical smoke test on `/dev/hailo0` (if present).
10. Computes mAP50 and mAP50-95 across PyTorch, ONNX, and Hailo, enforcing quality gates.
11. Benchmarks inference latency percentiles ($p_{50}, p_{95}, p_{99}$) and throughput (FPS).
12. Produces `run_manifest.json` and human-readable `run_report.md`.

---

### Workflow B: Dataset Validation Only (`dataset`)

Checks dataset integrity without starting training or compiler phases:

```bash
# If initializing from bundled sample data:
cp -r data_example data

# Run dataset validator
python train_and_compile.py dataset --dataset data/dataset.yaml
```

**Common Outputs:**
- Verifies label coordinates are strictly within $[0.0, 1.0]$.
- Detects whether any images in `images/val` are duplicate hashes of images in `images/train` (train/val leakage).
- Ensures sufficient uncorrupted images are available for subsequent INT8 calibration.

---

### Workflow C: Model Training Only (`train`)

Runs pre-flight validation and trains the YOLO11 model:

```bash
python train_and_compile.py train \
  --config config/config.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

**Output Artefacts:**
- Checkpoint: `artifacts/runs/<run_id>/pytorch/best.pt`
- Canonical Pointer: `artifacts/pytorch/best.pt`
- Training metadata and metrics in `run_manifest.json`.

---

### Workflow D: Static FP32 ONNX Export (`export`)

Converts a trained checkpoint into a strict static FP32 ONNX model:

```bash
python train_and_compile.py export \
  --config config/config.yaml \
  --model artifacts/pytorch/best.pt
```

**Key Validations:**
- Verifies batch size is fixed to $1$.
- Verifies spatial dimensions are fixed to $640 \times 640$.
- Rejects dynamic axes (incompatible with Hailo-8L).
- Runs an ONNX Runtime smoke test to verify numerical sanity.

**Output Artefacts:**
- `artifacts/runs/<run_id>/onnx/model.onnx`
- `artifacts/onnx/model.onnx`

---

### Workflow E: Calibration Dataset Generation (`calibrate`)

Extracts and letterboxes a deterministic calibration dataset for Hailo INT8 PTQ:

```bash
python train_and_compile.py calibrate \
  --config config/config.yaml \
  --count 200 \
  --seed 42
```

**Output Artefacts:**
- Preprocessed tensor: `artifacts/runs/<run_id>/calibration/calib_data.npy`
- Sample images: `artifacts/runs/<run_id>/calibration/images/`
- Audit ledger: `artifacts/runs/<run_id>/calibration/calibration_manifest.json`
- Distribution statistics: `artifacts/runs/<run_id>/calibration/calibration_statistics.json`

---

### Workflow F: Hailo DFC Compilation (`compile`)

Executes Hailo Dataflow Compiler stages on an x86_64 host:

```bash
python train_and_compile.py compile \
  --config config/config.yaml \
  --model artifacts/onnx/model.onnx
```

**Compiler Stages Executed:**
1. ONNX → HAR parsing (`hailo parser onnx`).
2. Model script injection (`yolo11.alls` specifying input normalisation).
3. INT8 PTQ optimisation using `calib_data.npy`.
4. Layer mapping, dataflow routing, and scheduling targeting `hailo8l`.
5. HEF generation: `artifacts/runs/<run_id>/hailo/model.hef`.

---

### Workflow G: Validation & Edge Benchmarking (`validate`)

Runs accuracy comparison gates and physical hardware benchmarks:

```bash
# Execute evaluation and benchmark
python train_and_compile.py validate --config config/config.yaml

# Strictly require physical Hailo hardware (fails if /dev/hailo0 is absent)
python train_and_compile.py validate --config config/config.yaml --require-hardware
```

**Degradation Gates:**
- Compares FP32 PyTorch baseline against INT8 Hailo HEF.
- If $\Delta\text{mAP50} > 0.02$ or $\Delta\text{mAP50-95} > 0.02$, validation reports `FAILED` with code `E-ACC-005`.

---

## 4. CLI Command-Line Reference

The main entry point is `train_and_compile.py`.

```text
usage: train_and_compile.py [-h] [--config CONFIG] [--run-id RUN_ID]
                            [--model MODEL] [--dataset DATASET]
                            [--device DEVICE] [--epochs EPOCHS]
                            [--imgsz IMGSZ] [--batch BATCH] [--seed SEED]
                            [--dry-run] [--force]
                            [--execution-mode {auto,portable,hailo_compile,hailo_runtime}]
                            [--require-hardware]
                            {doctor,dataset,train,export,calibrate,compile,validate,all}
```

### Subcommands

| Subcommand | Description |
| :--- | :--- |
| `doctor` | Inspect host environment, installed packages, Hailo DFC, and hardware devices. |
| `dataset` | Execute Phase A dataset pre-flight validation. |
| `train` | Execute Phase A validation and Phase B YOLO11 training. |
| `export` | Execute Phase C static FP32 ONNX export and Phase C2 validation. |
| `calibrate` | Execute Phase D deterministic INT8 calibration dataset generation. |
| `compile` | Execute Phase F (parser), Phase G (optimiser), and Phase H (HEF compilation). |
| `validate` | Execute Phase I (accuracy gates) and Phase K (latency and telemetry benchmarks). |
| `all` | Execute the complete, dependency-aware end-to-end pipeline. |

### Global Flags

| Option | Shorthand | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `-c` | `str` | `config/config.yaml` | Path to master YAML configuration file. |
| `--run-id` | | `str` | Auto-generated | Explicit run identifier (`YYYYMMDD-HHMMSS-<hash>`). |
| `--model` | | `str` | Configured | Model source weights (`.pt`), variant name, or ONNX path. |
| `--dataset` | | `str` | Configured | Path to YOLO `dataset.yaml`. |
| `--device` | | `str` | `auto` | Training compute device (`auto`, `cpu`, `0`, `0,1`). |
| `--epochs` | | `int` | `100` | Number of training epochs. |
| `--imgsz` | | `int` | `640` | Square image resolution (must be 640 for canonical Hailo contract). |
| `--batch` | | `int` | `16` | Training batch size. |
| `--seed` | | `int` | `42` | Master random seed. |
| `--dry-run` | | `flag` | `false` | Display execution plan and pre-flight assessments without modifying storage. |
| `--force` | | `flag` | `false` | Force re-execution and overwrite existing stage artefacts. |
| `--execution-mode` | | `choice`| `auto` | Enforce execution boundary (`auto`, `portable`, `hailo_compile`, `hailo_runtime`). |
| `--require-hardware`| | `flag` | `false` | Fail with error if physical Hailo-8L accelerator (`/dev/hailo0`) is absent. |

---

## 5. Operational Modes & Execution Boundaries

You can constrain the pipeline to a specific architectural tier using `--execution-mode`:

### 1. `portable` Mode
Use on development laptops, workstations without Hailo DFC, or CI runners:
```bash
python train_and_compile.py --execution-mode portable all
```
*Behaviour*: Executes dataset validation, training, ONNX export, and calibration. Hailo compilation and hardware validation phases are skipped cleanly and marked `NOT_EXECUTED`.

### 2. `hailo_compile` Mode
Use on Linux x86_64 compilation servers equipped with Hailo Dataflow Compiler (DFC):
```bash
python train_and_compile.py --execution-mode hailo_compile compile
```
*Behaviour*: Requires Hailo DFC. Converts ONNX to HAR, performs INT8 PTQ, and compiles `model.hef`.

### 3. `hailo_runtime` Mode
Use directly on the **Raspberry Pi 5 with Hailo-8L AI HAT+**:
```bash
python train_and_compile.py --execution-mode hailo_runtime validate --require-hardware
```
*Behaviour*: Probes `/dev/hailo0`, executes HailoRT physical inference, captures sensor telemetry, and measures latency percentiles.

---

## 6. Resuming Workflows with Existing Artefacts

You do not need to re-train or re-export if valid artefacts already exist. Point the CLI directly to your existing files:

### Resuming from an Existing PyTorch Checkpoint
```bash
python train_and_compile.py all --model /path/to/custom_best.pt
```
The pipeline validates the `.pt` file, skips training, and proceeds to ONNX export.

### Resuming from an Existing ONNX Model
```bash
python train_and_compile.py compile --model /path/to/model.onnx
```
The pipeline verifies the static $1 \times 3 \times 640 \times 640$ ONNX contract and proceeds to calibration and Hailo DFC compilation.

### Resuming from an Existing HEF Binary on Raspberry Pi 5
In `config/config.yaml`:
```yaml
hailo:
  hef_path: "artifacts/hailo/model.hef"
```
Then run:
```bash
python train_and_compile.py validate
```
The pipeline immediately executes HailoRT hardware validation and benchmarks the HEF on physical hardware.

---

## 7. Dry-Run Planning & Inspection

To inspect the execution plan without modifying files or running compute-intensive operations:

```bash
python train_and_compile.py --dry-run all
```

**Example Output:**
```text
================================================================================
 hailo-yolo11-compiler Execution Plan (DRY RUN)
================================================================================
Run ID:             20261008-223015-88f2ab
Execution Mode:     AUTO
Target Accelerator: hailo8l (Raspberry Pi 5 AI HAT+)
Model / Task:       yolo11n.pt (detect)
Dataset YAML:       data/dataset.yaml
Calibration Count:  200 (Seed: 42)

Planned Stages:
  1. Dataset Pre-flight Validation     [Portable      ] Validates YOLO syntax, labels, leakage
  2. YOLO11 Training / Checkpoint      [Portable      ] Produces/validates best.pt
  3. FP32 ONNX Export                  [Portable      ] Exports 1x3x640x640 static FP32 model.onnx
  4. ONNX Validation & Smoke Test      [Portable      ] Validates ONNX graph and runs smoke inference
  5. Calibration Set Generation        [Portable      ] Samples & letterboxes 200 images, writes calib_data.npy
  6. ONNX -> HAR Translation           [Hailo DFC     ] Parses ONNX and creates Hailo Archive with .alls script
  7. INT8 Post-Training Quantisation   [Hailo DFC     ] Quantises HAR using calibration dataset
  8. Hailo-8L Compilation              [Hailo DFC     ] Compiles optimised HAR to model.hef
  9. Accuracy Validation Gates         [Portable/Hailo] Evaluates mAP50 and mAP50-95 degradation
  10. Physical HailoRT Smoke Test      [Hailo Runtime ] Runs inference on Raspberry Pi 5 Hailo-8L AI HAT+

Hardware & Software Capability Assessment:
  Hailo DFC:          AVAILABLE (DFC v3.28.0 detected)
  Hailo-8L Hardware:  NOT DETECTED (Stage 10 will report HARDWARE_VALIDATION=NOT_EXECUTED)
================================================================================
```

---

## 8. Forced Regeneration of Artefacts

By default, the pipeline uses existing valid artefacts if they match cryptographic checksums. To force complete re-execution from scratch:

```bash
python train_and_compile.py --force all
```

This bypasses cached stages, re-trains the model, re-extracts calibration samples, and triggers a full Hailo compilation pass.

---

## 9. Interpreting Generated Reports & Manifests

Every pipeline execution generates two comprehensive audit files in `artifacts/runs/<run_id>/`:

### 1. `run_manifest.json` (Machine-Readable Ledger)
Contains full cryptographic details, environment information, and metric provenance:
```json
{
  "run_id": "20261008-201422-a31f",
  "status": "SUCCESS",
  "execution_mode": "portable",
  "environment": {
    "os": "Linux 6.6.20+rpt-rpi-2712",
    "python_version": "3.11.2",
    "git": { "commit": "7a7d2b5", "dirty": false }
  },
  "artifacts": {
    "model_hef": {
      "path": "/workspace/artifacts/runs/.../hailo/model.hef",
      "sha256": "8f3b...19c2",
      "size_bytes": 4519820,
      "status": "valid"
    }
  },
  "metrics": {
    "pytorch_map50": 0.812,
    "hailo_map50": 0.804,
    "delta_map50": 0.008,
    "accuracy_gate_passed": true
  }
}
```

### 2. `run_report.md` (Human-Readable Markdown Report)
Contains formatted execution summaries, mAP comparison tables, latency percentiles, and hardware telemetry graphs ready for inclusion in engineering documentation.

---

## 10. Live Model Deployment (`deploy`)

The `deploy` phase provides unified real-time streaming inference across edge cameras, pre-recorded video files, and image directories.

### Live Camera Ingestion

The deployment engine supports multiple camera backends configured via `--camera-type {auto, opencv, picamera2, libcamera}`:
- **Raspberry Pi AI Camera (Sony IMX500)** & **Camera Module 3**: Natively ingested using `picamera2` or the direct hardware pipeline stream (`rpicam-vid` / `libcamera-vid`).
- **USB Webcams**: Captured via V4L2 `cv2.VideoCapture` with automatic enumeration and proactive filtering of Raspberry Pi internal ISP, CFE, and PiSP memory nodes.

```bash
# Live deployment on physical Hailo-8L NPU
python train_and_compile.py deploy --source 0 --hailo

# Native Raspberry Pi AI Camera (Sony IMX500) via Picamera2
python train_and_compile.py deploy --source 0 --camera-type picamera2 --hailo

# Fallback or comparative multi-threaded CPU execution
python train_and_compile.py deploy --source 0 --cpu --threads 4
```

### Continuous Hardware Telemetry Streaming

The pipeline can record live edge hardware metrics (accelerator power, CPU temperatures, thermal zones, core frequencies, and inference latency) into structured JSONL:
```bash
python train_and_compile.py deploy --source 0 --hailo --telemetry --telemetry-path runs/live_telemetry.jsonl
```
Every recorded data point includes an active inference device label (`Device: NPU (Hailo HAT)` vs. `Device: CPU`) to enable rigorous edge performance evaluation.

---

## 11. CPU Multithreading & Operations Engine (`operations.py`)

To maximize pipeline performance during host-bound operations, the compiler includes a dedicated operations optimization engine ([operations.py](../operations.py)):

### Standalone Status & Benchmark Tools

```bash
# Display detected hardware capacity, thread pools, and active BLAS/MKL bindings
python operations.py --status

# Run multi-threaded CPU matrix operation benchmark
python operations.py --benchmark
```

### Multithreaded Subsystems
- **ONNX Runtime**: Automatically configured with `ExecutionMode.ORT_PARALLEL`, intra-op and inter-op thread pools.
- **PyTorch**: Inter-op and intra-op thread counts tuned to host physical core capacity.
- **OpenCV**: Vectorized optimizations enabled (`cv2.setUseOptimized(True)`) with thread pool matching available cores.
- **Data Loaders & Validation**: Parallel multi-threaded calibration sampling and dataset integrity checks.
- **CLI Parameter**: Specify `--threads <N>` or `--cpu-threads <N>` on any pipeline command to override automatic thread pool sizing.

---

## 12. Licensing, Academic Provenance & Citation

### Licensing & Legal Terms

This research software and associated architectural artefacts are published under the open-source **BSD 3-Clause Licence**. Refer to the [LICENSE](../LICENSE) file for comprehensive legal terms.

```text
Copyright (c) 2026, Isiaka Mosudi, FHTW.
All rights reserved.
```

Developed within the academic framework of Master's thesis requirements. Department of Electronic Engineering and Entrepreneurship, University of Applied Sciences Technikum Wien (FH Technikum Wien), Vienna, Austria.

### Academic Citation

If you utilise this compiler, MLOps orchestration framework, or benchmarking architecture in academic research, please cite:

```bibtex
@software{hailo_yolo11_compiler_2026,
  author    = {Isiaka, Mosudi},
  title     = {hailo-yolo11-compiler: Reproducible YOLO11 to Hailo-8L Edge AI MLOps Compilation and Evaluation Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/imosudi/hailo-yolo11-compiler},
  license   = {BSD-3-Clause}
}
```
