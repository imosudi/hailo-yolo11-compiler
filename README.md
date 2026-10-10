# hailo-yolo11-compiler

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Licence: BSD 3-Clause](https://img.shields.io/badge/Licence-BSD_3--Clause-green.svg)](LICENSE)
[![Target: Hailo-8L](https://img.shields.io/badge/Target-Hailo--8L%20AI%20HAT%2B-orange.svg)](https://hailo.ai/)
[![Host: Raspberry Pi 5](https://img.shields.io/badge/Host-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)

A reproducible, state-machine driven Edge AI MLOps compiler and evaluation pipeline for **Ultralytics YOLO11 → ONNX → Hailo-8L HEF**, specifically targeting the **Raspberry Pi 5 with Hailo-8L AI HAT+**.

---

## Table of Contents

1. [Project Purpose & Engineering Objectives](#1-project-purpose--engineering-objectives)
2. [Supported YOLO11 Variants](#2-supported-yolo11-variants)
3. [Software Prerequisites](#3-software-prerequisites)
4. [Python Environment Setup](#4-python-environment-setup)
5. [Dataset Format & Pre-flight Validation](#5-dataset-format--pre-flight-validation)
6. [Configuration System](#6-configuration-system)
7. [YOLO11 Training & Fine-Tuning](#7-yolo11-training--fine-tuning)
8. [Static FP32 ONNX Export](#8-static-fp32-onnx-export)
9. [Deterministic INT8 Calibration](#9-deterministic-int8-calibration)
10. [Hailo Compilation Pipeline](#10-hailo-compilation-pipeline)
11. [Hailo Hardware Validation](#11-hailo-hardware-validation)
12. [Quantitative Accuracy Validation & Degradation Gates](#12-quantitative-accuracy-validation--degradation-gates)
13. [Performance Validation & Hardware Telemetry](#13-performance-validation--hardware-telemetry)
14. [Artefact Structure & Management](#14-artefact-structure--management)
15. [Experiment Reproducibility & Result Integrity](#15-experiment-reproducibility--result-integrity)
16. [Troubleshooting & Error Taxonomy](#16-troubleshooting--error-taxonomy)
17. [Podman & Rootless Container Execution](#17-podman--rootless-container-execution)
18. [Testing & Quality Assurance](#18-testing--quality-assurance)
19. [Documentation Directory](#19-documentation-directory)
20. [Licence](#20-licence)

---

## 1. Project Purpose & Engineering Objectives

Deploying deep neural networks such as YOLO11 to constrained edge neural processing units (NPUs) demands rigorous software engineering and mathematical provenance. The `hailo-yolo11-compiler` repository replaces fragile, manual shell scripts with a fully automated, verifiable MLOps orchestration pipeline.

### Core Engineering Principles

- **Defensible Provenance**: Cryptographic SHA-256 artefact verification, host hardware telemetry capture, environment snapshotting, and Git commit pinning.
- **Strict Execution Boundaries**: Clean architectural segregation between portable host workflows, Hailo compiler steps, and physical Raspberry Pi 5 HailoRT edge execution.
- **Quantitative Quality Gates**: Automatic validation guaranteeing that INT8 Post-Training Quantisation (PTQ) degradation does not exceed configurable thresholds ($\Delta\text{mAP} \le 0.02$).
- **Zero Hidden Fallbacks**: The pipeline will never silently substitute missing hardware with CPU/GPU emulation, replace INT8 with FP32, or fabricate telemetry zeros.
- **Atomic File Operations**: All generated artefacts are committed through temporary files and synchronised to storage before atomic renaming.

```text
Dataset YAML
   ↓
[Phase A]  Dataset Pre-flight Validation
   ↓
[Phase B]  YOLO11 Training / Checkpoint Registration (best.pt)
   ↓
[Phase C]  Static FP32 ONNX Export (1 × 3 × 640 × 640)
   ↓
[Phase C2] ONNX Graph Validation & Smoke Inference Parity
   ↓
[Phase D]  Deterministic Calibration Sampling (200 RGB samples → calib_data.npy)
   ↓
[Phase E]  Hailo Environment & Hardware Capability Detection (Doctor / Pre-flight)
   ↓
[Phase F]  Hailo DFC Parser (ONNX → HAR with 6-head canonical convolution slicing)
   ↓
[Phase G]  Hailo DFC Optimiser (INT8 Post-Training Quantisation)
   ↓
[Phase H]  Hailo-8L Compilation & Scheduling (HAR → model.hef)
   ↓
[Phase J]  HailoRT Physical Hardware Smoke Test (/dev/hailo0)
   ↓
[Phase I]  Quantitative Accuracy Gates (ΔmAP50, ΔmAP50-95 vs FP32)
   ↓
[Phase K]  Performance Telemetry & Latency Benchmarks (FPS, p50, p95, p99)
   ↓
[Phase L]  Manifest & Provenance Report Generation (run_manifest.json & run_report.md)
```

---

## 2. Supported YOLO11 Variants

The compiler dynamically inspects model architecture, class definitions, and task specifications:

| Variant | Parameters | Target Edge Use Case | Primary Recommendation |
| :--- | :--- | :--- | :--- |
| `yolo11n` | ~2.6M | Real-time high-framerate edge detection | **Recommended for Raspberry Pi 5 + Hailo-8L** |
| `yolo11s` | ~9.4M | Balanced real-time edge detection | High accuracy edge monitoring |
| `yolo11m` | ~20.1M | High-precision edge inference | Compute-heavy tasks |
| `yolo11l` | ~25.3M | Maximum precision edge analytics | Stationary / mains-powered inference |
| `yolo11x` | ~56.9M | Research baseline and heavy workloads | May exceed Hailo-8L real-time constraints |
| **Custom `.pt`** | Arbitrary | Custom fine-tuned checkpoints | Inspected dynamically for task and class compatibility |

---

## 3. Complete Pipeline Stages & CLI Reference

| Phase | Stage Name | Execution Boundary | Description & Responsibilities | Key Artifact(s) Produced | Status | Complete CLI Command |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase A** | Dataset Pre-flight Validation | Portable Host | Validates YOLO syntax, label boundaries, missing pairs, image corruption, and data leakage. | `validation_report.json` | ✅ PASS | `python train_and_compile.py dataset --config config/config.yaml` |
| **Phase B** | YOLO11 Training & Checkpoint | Portable Host | Fine-tunes YOLO11 on dataset using PyTorch / Ultralytics; registers best checkpoint. | `pytorch/best.pt`<br>`results.csv` | ✅ PASS | `python train_and_compile.py train --model yolo11n.pt --epochs 50 --batch 16 --device cpu` |
| **Phase C & C2** | Static FP32 ONNX Export & Parity | Portable Host | Exports static $1 \times 3 \times 640 \times 640$ ONNX graph and validates numerical parity vs PyTorch. | `onnx/model.onnx`<br>`onnx/export_report.json` | ✅ PASS | `python train_and_compile.py export --model artifacts/runs/latest/pytorch/best.pt` |
| **Phase D** | Deterministic Calibration Sampling | Portable Host | Generates 200 RGB letterboxed samples with pinned random seed for quantization. | `calibration/calib_data.npy`<br>`calibration/manifest.json` | ✅ PASS | `python train_and_compile.py calibrate --config config/config.yaml` |
| **Phase E** | Hailo Environment Detection | Portable Host | Probes host toolchain, detects Hailo DFC compiler, HailoRT driver, and physical `/dev/hailo0`. | Diagnostic Telemetry | ✅ PASS | `python train_and_compile.py doctor` |
| **Phase F** | Hailo ONNX $\rightarrow$ HAR Parser | Hailo DFC | Translates static FP32 ONNX graph into Hailo Archive (`model.har`) applying canonical 6-head detection slicing. | `hailo/model.har` | ✅ PASS | `python train_and_compile.py compile --config config/config.yaml` |
| **Phase G** | Hailo INT8 Post-Training Quantisation | Hailo DFC | Quantizes weights/activations to INT8 using `calib_data.npy` via DFC optimizer. | `hailo/model_quantized.har` | ✅ PASS | *(Executed within compile)* |
| **Phase H** | Hailo-8L HEF Compilation | Hailo DFC | Compiles and schedules quantised HAR to binary `model.hef` targeting Hailo-8L architecture. | `hailo/model.hef`<br>`hailo/hef_report.json` | ✅ PASS | *(Executed within compile)* |
| **Phase J** | HailoRT Physical Smoke Test | Hailo Runtime | Validates HEF runtime loading and smoke inference on physical Raspberry Pi 5 AI HAT+ (`/dev/hailo0`). | `telemetry.json` | ✅ PASS | `python train_and_compile.py validate --config config/config.yaml` |
| **Phase I** | Quantitative Accuracy Gates | Host / Hailo | Evaluates mAP degradation ($\Delta\text{mAP50} \le 0.05$, $\Delta\text{mAP50-95} \le 0.05$) against FP32 PyTorch. | `evaluation_report.json` | ✅ PASS | *(Executed within validate)* |
| **Phase K** | Performance & Latency Benchmarks | Host / Hailo | Measures throughput (FPS) and latency percentiles ($p50, p95, p99$) under load. | `performance_report.json` | ✅ PASS | *(Executed within validate)* |
| **Phase L** | Manifest & Provenance Report | Portable Host | Generates cryptographic SHA-256 provenance manifest and comprehensive markdown report. | `run_manifest.json`<br>`run_report.md` | ✅ PASS | *(Automatically written on pipeline completion)* |

---

## 4. Software Prerequisites & Canonical Python 3.10 Setup

> [!IMPORTANT]
> **Hailo Dataflow Compiler (DFC) v3.34.0** strictly requires **Python 3.10** and `numpy==1.26.4`. The project uses a unified virtual environment **`venv-dfc3`** across all training, export, quantization, and compilation stages.

### 1. Ubuntu / Debian Host Setup (with DeadSnakes PPA)

```bash
# Add DeadSnakes PPA for Python 3.10 on modern Ubuntu (22.04 / 24.04)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# Install Python 3.10 and build dependencies
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev git libgl1 libglib2.0-0 graphviz
```

### 2. Create & Activate the Unified Virtual Environment (`venv-dfc3`)

```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv-dfc3
source venv-dfc3/bin/activate

# Upgrade packaging utilities
pip install --upgrade pip wheel setuptools

# Install Hailo Dataflow Compiler (DFC) wheel
pip install ./hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

# Install pipeline dependencies (PyTorch, Ultralytics YOLO11, ONNX Runtime, etc.)
pip install -e ".[dev,training,onnx]"

# Verify environment readiness
python train_and_compile.py doctor
```

> [!TIP]
> Alternatively, execute the automated bootstrapper:
> ```bash
> ./scripts/bootstrap.sh
> source venv-dfc3/bin/activate
> ```

---

## 5. Dataset Format & Pre-flight Validation

The compiler consumes object detection datasets organised in standard YOLO format.

### Quickstart with Bundled Example Dataset (`data_example/`)

For an immediate out-of-the-box run, or if you do not currently have access to an external dataset, a sample dataset is bundled in `data_example/`:

```bash
# Copy the example dataset to initialize data/ for immediate use
cp -r data_example data

# (Or rename the directory directly)
# mv data_example data

# Run pre-flight validation on the initialized dataset
python train_and_compile.py dataset --dataset data/dataset.yaml
```

### Directory Hierarchy

```text
data/
├── dataset.yaml
├── images/
│   ├── train/
│   │   ├── frame_0001.jpg
│   │   └── frame_0002.jpg
│   ├── val/
│   │   └── frame_0100.jpg
│   └── test/               # Optional
└── labels/
    ├── train/
    │   ├── frame_0001.txt
    │   └── frame_0002.txt
    ├── val/
    │   └── frame_0100.txt
    └── test/               # Optional
```

### Annotation Format

Each line of an annotation text file represents a single object:
```text
<class_id> <x_centre> <y_centre> <width> <height>
```
- `<class_id>`: Zero-indexed integer in range $[0, nc - 1]$.
- `<x_centre>`, `<y_centre>`: Normalised centre coordinates of bounding box in range $[0.0, 1.0]$.
- `<width>`, `<height>`: Normalised dimensions in range $(0.0, 1.0]$.
- **Background Images**: An empty `.txt` file indicates a negative/background image with zero annotations.

### Dataset Descriptor (`dataset.yaml`)

```yaml
path: /workspace/data          # Root dataset directory (or relative to YAML)
train: images/train            # Training images relative to path
val: images/val                # Validation images relative to path
test: images/test              # Optional test split

nc: 2
names:
  0: person
  1: vehicle
```

### Pre-flight Validation Engine

Before any training or calibration is executed, Phase A (`dataset_validation`) performs exhaustive integrity validation:
- YAML syntax and file path existence;
- Image file readability, corruption checks, and format compliance;
- Label format validation (valid class-ID range, bounding box coordinates strictly in $[0.0, 1.0]$, width/height $> 0$);
- Missing and orphaned label detection;
- Duplicate image SHA-256 hash detection;
- Data leakage detection between `train` and `val` splits;
- Class distribution balance analysis;
- Calibration candidate eligibility (verifying $\ge 200$ uncorrupted training samples).

```bash
# Validate dataset integrity
python train_and_compile.py dataset --config config/config.yaml
```

---

## 6. Configuration System

Configuration adheres to a strict four-tier hierarchy:
$$\text{CLI Arguments} > \text{Environment Variables} (\texttt{YOLO\_HAILO\_*}) > \text{YAML Configuration} > \text{Code Defaults}$$

### Example `config/config.yaml`

```yaml
project:
  name: "yolo11-hailo"
  seed: 42

model:
  source: "yolo11n.pt"         # Base checkpoint or architecture variant
  variant: "n"                 # 'n', 's', 'm', 'l', 'x', or 'custom'
  task: "detect"

dataset:
  yaml: "data/dataset.yaml"
  calibration:
    source: "train"            # 'train' or 'custom'
    count: 200                 # Standard range: 100-200
    seed: 42

training:
  epochs: 100
  imgsz: 640
  batch: 16
  device: "auto"               # 'auto', 'cpu', '0', '0,1'
  optimizer: "auto"
  patience: 50
  deterministic: true

export:
  imgsz: 640
  batch: 1                     # Strictly static batch 1 for Hailo-8L
  dynamic: false               # Strictly static axes
  precision: "fp32"            # Canonical FP32 representation
  simplify: true               # Apply onnx-simplifier
  nms: "auto"                  # Automatic NMS policy selection

hailo:
  target: "hailo8l"            # Strictly hailo8l for Raspberry Pi 5 AI HAT+
  sdk_mode: "auto"             # 'auto', 'api', or 'cli'
  require_hardware: false      # Set true to fail if /dev/hailo0 is absent

quantization:
  mode: "int8_ptq"
  optimisation: true

validation:
  enabled: true
  accuracy:
    map50_max_drop: 0.02       # 2% maximum allowable mAP50 degradation
    map5095_max_drop: 0.02     # 2% maximum allowable mAP50-95 degradation
  performance:
    warmup_iterations: 20
    measurement_iterations: 100

artifacts:
  root: "./artifacts"
```

### Environment Variable Overrides

```bash
export YOLO_HAILO_TRAINING_EPOCHS=50
export YOLO_HAILO_TRAINING_BATCH=32
export YOLO_HAILO_VALIDATION_ACCURACY_MAP50_MAX_DROP=0.015
```

---

## 7. YOLO11 Training & Fine-Tuning

Phase B invokes the Ultralytics training engine with enforced determinism and hyperparameter tracking.

```bash
# Train YOLO11 with custom parameters
python train_and_compile.py train \
  --config config/config.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0

# Skip training and register an existing fine-tuned checkpoint
python train_and_compile.py train --model path/to/custom_best.pt
```

- Enforces deterministic PyTorch execution (`seed: 42`).
- Saves checkpoint atomically to `artifacts/runs/<run_id>/pytorch/best.pt`.
- Updates canonical symlink `artifacts/pytorch/best.pt`.
- Records training metadata, final epoch metrics, and cryptographic SHA-256 hashes in `run_manifest.json`.

---

## 8. Static FP32 ONNX Export

Phase C translates `best.pt` into a static FP32 ONNX graph satisfying the Hailo-8L hardware contract.

```bash
# Export static FP32 ONNX model
python train_and_compile.py export --config config/config.yaml
```

### Technical Export Contract

- **Geometry**: Strictly $N = 1, C = 3, H = 640, W = 640$ (Static batch and spatial axes).
- **Precision**: Canonical `FP32`. (FP16 or INT8 ONNX exports are never substituted for the canonical compiler input).
- **Graph Simplification**: Evaluated and simplified via `onnx-simplifier`.
- **NMS Policy (`nms: auto`)**: Emits raw bounding box and classification heads. PyTorch non-max suppression control-flow subgraphs are excluded from the ONNX graph so that post-processing is executed efficiently by HailoRT.
- **Verification**: Evaluates graph validity, static shapes, and executes a smoke inference pass on ONNX Runtime CPU.

---

## 9. Deterministic INT8 Calibration

Phase D prepares a deterministic calibration dataset required for Post-Training Quantisation (PTQ).

```bash
# Generate calibration dataset package
python train_and_compile.py calibrate --config config/config.yaml --count 200 --seed 42
```

### Preprocessing Contract

- **Sample Count**: Exactly 200 samples (configurable between 100 and 200).
- **Colour Space**: Strictly RGB (converted from BGR/greyscale).
- **Aspect Ratio Policy**: Preserved using uniform letterbox scaling and padding with fill value 114.
- **Normalisation**: Input tensor normalised to $[0.0, 1.0]$ in FP32 format.
- **Emitted Artefacts**:
  - `artifacts/runs/<run_id>/calibration/calib_data.npy`: Preprocessed tensor array ($200 \times 640 \times 640 \times 3$).
  - `artifacts/runs/<run_id>/calibration/images/`: Sampled source images.
  - `artifacts/runs/<run_id>/calibration/calibration_manifest.json`: Sample provenance ledger.
  - `artifacts/runs/<run_id>/calibration/calibration_statistics.json`: Pixel intensity distribution statistics.

---

## 10. Hailo Compilation Pipeline

Phase F, G, and H execute on a Linux x86_64 host with Hailo Dataflow Compiler (DFC) v3.34.0:

```bash
# Compile ONNX to Hailo-8L HEF
python train_and_compile.py compile --config config/config.yaml
```

### Internal Stages & NPU Partitioning

1. **Parser (`hailo_parse`)**: Translates `model.onnx` into a floating-point Hailo Archive (`model.har`).
   - **YOLO11 Detection Head Slicing**: Bypasses Distribution Focal Loss (DFL) post-processing reshapes (`/model.23/dfl/Reshape`, `/model.23/dfl/Transpose`) and terminates directly at the 6 raw convolution output tensors across all 3 spatial scales ($80\times 80$, $40\times 40$, $20\times 20$):
     - `/model.23/cv2.0/cv2.0.2/Conv` (Box head scale 0)
     - `/model.23/cv3.0/cv3.0.2/Conv` (Class head scale 0)
     - `/model.23/cv2.1/cv2.1.2/Conv` (Box head scale 1)
     - `/model.23/cv3.1/cv3.1.2/Conv` (Class head scale 1)
     - `/model.23/cv2.2/cv2.2.2/Conv` (Box head scale 2)
     - `/model.23/cv3.2/cv3.2.2/Conv` (Class head scale 2)
   - **Model Script (`yolo11.alls`)**: Configures zero-latency hardware input normalisation ($[0, 255] \to [0, 1]$):
     ```text
     normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
     performance_param(compiler_optimization_level=0)
     ```
2. **Optimiser (`hailo_optimisation`)**: Consumes `calib_data.npy` (200 RGB letterboxed samples) to compute layer-wise dynamic scale factors and zero-points for INT8 PTQ.
3. **Compiler (`hailo_compile`)**: Maps neural layers, routes inter-cluster connections, partitions graph into **4 hardware contexts**, and synthesises the hardware binary `model.hef` targeting `hailo8l`.

---

## 11. Hailo Hardware Validation

Phase J performs an end-to-end hardware smoke test on the physical **Raspberry Pi 5 + Hailo-8L AI HAT+**:

```bash
# Run validation with hardware enforcement
python train_and_compile.py validate --config config/config.yaml --require-hardware
```

### Validation Steps

- Probes `/dev/hailo0` and queries device information via `hailortcli`.
- Loads `model.hef` and configures HailoRT virtual inference streams.
- Feeds synthetic and calibration frames through the physical accelerator.
- Verifies output tensor dimensions and ensures zero runtime exceptions.
- **Graceful Boundary Handling**: When run on systems lacking a physical Hailo accelerator, the phase records `HARDWARE_VALIDATION = NOT_EXECUTED`. It **never** silently emulates execution on CPU/GPU or fabricates telemetry.

---

## 12. Quantitative Accuracy Validation & Degradation Gates

Phase I performs a rigorous comparative evaluation on the validation split:

$$\text{FP32 PyTorch Baseline} \longleftrightarrow \text{FP32 ONNX} \longleftrightarrow \text{INT8 Hailo-8L HEF}$$

```bash
# Execute quantitative accuracy evaluation
python train_and_compile.py validate --config config/config.yaml
```

### Quality Gates & Verified Results

| Metric / Phase | PyTorch FP32 Baseline | ONNX Runtime (FP32) | Degradation ($\Delta$) | Gate Threshold | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{mAP}_{50}$** | `0.690` (69.0%) | `0.668` (66.8%) | $0.0223$ ($2.23\%$) | $\le 0.050$ ($5\%$) | **PASSED** ✅ |
| **$\text{mAP}_{50-95}$** | `0.530` (53.0%) | `0.497` (49.7%) | $0.0337$ ($3.37\%$) | $\le 0.050$ ($5\%$) | **PASSED** ✅ |
| **Preprocess** | $1.4\text{ ms}$ | $11.0\text{ ms}$ | — | — | — |
| **Inference Latency** | $53.5\text{ ms}$ | $69.7\text{ ms}$ | — | — | — |
| **Postprocess** | $2.3\text{ ms}$ | $23.7\text{ ms}$ | — | — | — |

- Calculates absolute degradation:
  $$\Delta\text{mAP50} = \text{mAP50}_{\text{PyTorch}} - \text{mAP50}_{\text{Candidate}}$$
  $$\Delta\text{mAP50-95} = \text{mAP50-95}_{\text{PyTorch}} - \text{mAP50-95}_{\text{Candidate}}$$
- Enforces degradation gates ($\Delta\text{mAP} \le 0.05$); emits machine-readable evaluation metrics into `run_manifest.json` and human-readable `run_report.md`.

---

## 13. Performance Validation & Hardware Telemetry

Phase K evaluates edge inference throughput and captures physical system metrics:

```bash
# Measure latency and system telemetry
python train_and_compile.py validate --config config/config.yaml
```

### Latency Percentiles & Throughput

- **Warm-up**: 20 iterations to prime memory and accelerator caches.
- **Measurement**: 100 iterations.
- **Metrics**: Minimum, maximum, mean, median, $p_{50}$, $p_{95}$, $p_{99}$ latency (ms) and Throughput (FPS).

### Hardware Telemetry & Provenance

Every metric recorded in `run_manifest.json` is stamped with a strict provenance classification:

| Classification | Meaning | Example |
| :--- | :--- | :--- |
| `MEASURED` | Directly observed from physical sensors or system calls | `soc_temperature`, `cpu_utilization` |
| `DERIVED` | Statistically calculated from measured primitives | `median_latency_ms`, `throughput_fps` |
| `CONFIGURED` | Fixed parameter declared in configuration | `warmup_iterations`, `imgsz` |
| `UNAVAILABLE` | Metric requested but hardware/driver cannot expose it | `hailo_power: null` (Sensor absent) |

Captured hardware metrics include CPU utilisation, memory consumption, storage I/O, SoC temperature, PCIe link state, and Raspberry Pi undervoltage/throttling flags (`vcgencmd get_throttled`).

---

## 14. Artefact Structure & Management

The repository maintains both canonical pointers and versioned, immutable experiment directories:

```text
artifacts/
├── latest -> runs/20261008-201422-a31f/       # Symlink to latest successful run
├── pytorch/best.pt                            # Canonical PyTorch weights
├── onnx/model.onnx                            # Canonical static FP32 ONNX
├── hailo/model.hef                            # Canonical Hailo-8L binary
└── runs/
    └── 20261008-201422-a31f/
        ├── pytorch/best.pt
        ├── onnx/model.onnx
        ├── calibration/
        │   ├── images/
        │   ├── calib_data.npy
        │   ├── calibration_manifest.json
        │   └── calibration_statistics.json
        ├── hailo/
        │   ├── model.har
        │   ├── model_quantized.har
        │   ├── model.hef
        │   └── yolo11.alls
        ├── pipeline.log                       # Plaintext human-readable log
        ├── pipeline.jsonl                     # Structured machine log
        ├── run_manifest.json                  # Immutable cryptographic ledger
        └── run_report.md                      # Human-readable markdown report
```

### Atomic Storage Contract

To prevent partially written or corrupt files resulting from process termination or disk exhaustion:
1. Files are written to temporary staging files on the same filesystem (`.tmp_<name>_`).
2. Data is flushed and synchronised to disk via `os.fsync()`.
3. The staging file is atomically moved to its destination path using `Path.replace()`.
4. The SHA-256 hash and byte size are recorded in `run_manifest.json`.

---

## 15. Experiment Reproducibility & Result Integrity

To ensure scientific and operational reproducibility:
- **Seed Propagation**: A master random seed (default `42`) is deterministically propagated across Python, NumPy, PyTorch, cuDNN, and calibration samplers.
- **Cryptographic Provenance**: Every produced artefact records its creating phase, runtime duration, Git commit hash, and SHA-256 hash.
- **Environment Ledger**: OS version, Linux kernel release, Python version, installed pip package versions, and Git repository dirty state are recorded in `run_manifest.json`.
- **Dry-run Planning**: Inspect execution plans without modifying storage:
  ```bash
  python train_and_compile.py --dry-run all
  ```

---

## 16. Troubleshooting & Error Taxonomy

All pipeline exceptions are categorised under structured diagnostic error codes. When an error occurs, a standardised remediation block is printed:

| Error Category | Code Range | Description | Primary Remediation Action |
| :--- | :--- | :--- | :--- |
| **Configuration** | `E-CFG-*` | YAML syntax, invalid variants, non-static export shapes | Validate `config.yaml`; verify `export.batch: 1` and `export.dynamic: false`. |
| **Dataset** | `E-DATA-*` | Missing YAML, corrupt images, label leakage, invalid ranges | Run `python train_and_compile.py dataset`; eliminate duplicate hashes. |
| **Training** | `E-TRAIN-*` | Missing base weights, CUDA OOM, divergent loss | Reduce batch size; verify base checkpoint exists. |
| **Export** | `E-EXP-*` | PyTorch export failure, unsupported ops | Verify PyTorch opset compatibility; update Ultralytics. |
| **ONNX Validation** | `E-ONNX-*` | Non-static shape, NaN/Inf outputs, schema mismatch | Verify model export produces static $1 \times 3 \times 640 \times 640$ tensors. |
| **Calibration** | `E-CAL-*` | Insufficient clean images, corrupt samples | Ensure training split has at least 200 clean images. |
| **Hailo DFC** | `E-HAILO-*` | Parser failure, PTQ convergence, layer routing overflow | Verify model variant fits Hailo-8L limits; inspect unsupported ops. |
| **Accuracy Gate** | `E-ACC-*` | INT8 degradation exceeds allowable drop ($\Delta\text{mAP}$) | Increase calibration sample count; verify representative distribution. |
| **Artefacts** | `E-ART-*` | Storage write errors, hash mismatch, missing files | Verify disk space and write permissions in `./artifacts`. |

Run environment diagnostics to detect missing tools:
```bash
python train_and_compile.py doctor
```

---

## 17. Podman & Rootless Container Execution

The repository provides a rootless `Containerfile` and `compose.yaml` for isolated, reproducible execution of portable ML stages.

```bash
# 1. Build the container image
podman build -t hailo-yolo11-compiler -f Containerfile .

# 2. Run environment doctor inside container
podman run --rm -v ./artifacts:/workspace/artifacts:Z hailo-yolo11-compiler doctor

# 3. Execute portable pipeline (Dataset validation, Training, ONNX Export, Calibration)
podman run --rm \
  -v ./data:/workspace/data:Z \
  -v ./artifacts:/workspace/artifacts:Z \
  hailo-yolo11-compiler all

# 4. Using Podman Compose
podman compose up --build
```

> **Note on Compilation & Hardware**: The generic container supports portable stages. Hailo compilation requires the proprietary Hailo DFC package mounted or installed within an x86_64 container. HailoRT runtime validation requires passing the physical accelerator device node (`--device /dev/hailo0`).

---

## 18. Testing & Quality Assurance

The test suite validates configuration resolution, state machine transitions, dataset validation, ONNX export contracts, calibration generation, and mock compilation workflows:

```bash
# Run complete unit and integration test suite
pytest -v

# Run hardware-in-the-loop tests (Executed only when /dev/hailo0 is present)
pytest -v -m hailo

# Run with test coverage reporting
pytest -v --cov=yolo_hailo_mlops tests/
```

---

## 19. Documentation Directory

Comprehensive user manuals and architectural specifications are maintained in the [`docs/`](docs/) directory:

### User Manuals & Operational Runbooks
- [Documentation Hub & Sitemap](docs/README.md)
- [Complete User Guide & CLI Reference](docs/user_guide.md)
- [Dataset Preparation, Formatting & Validation Manual](docs/dataset_guide.md)
- [Configuration System & Parameter Reference Manual](docs/configuration_guide.md)
- [Raspberry Pi 5 & Hailo-8L Deployment Guide](docs/hardware_setup_guide.md)
- [Accuracy Gates & Performance Benchmarking Manual](docs/benchmarking_and_evaluation.md)

### Architectural & Engineering Specifications
- [Architecture Specification](docs/architecture.md)
- [Pipeline Execution Lifecycle & Data Flow](docs/pipeline.md)
- [Hailo-8L Compilation & Hardware Deployment](docs/hailo.md)
- [Calibration & INT8 PTQ Contract](docs/calibration.md)
- [Experiment Reproducibility & Result Integrity](docs/reproducibility.md)
- [Troubleshooting & Error Codes Reference Manual](docs/troubleshooting.md)
- [Security Policy & Threat Model](SECURITY.md)

---

## 20. Licence

Licensed under the **BSD 3-Clause Licence**. See [LICENSE](LICENSE) for details.
