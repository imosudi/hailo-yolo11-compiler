# hailo-yolo11-compiler

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](LICENSE)
[![Target: Hailo-8L](https://img.shields.io/badge/Target-Hailo--8L%20AI%20HAT%2B-orange.svg)](https://hailo.ai/)
[![Host: Raspberry Pi 5](https://img.shields.io/badge/Host-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)

A reproducible, state-machine driven Edge AI MLOps compiler and evaluation pipeline for **Ultralytics YOLO11 → ONNX → Hailo-8L HEF**, specifically targeting the **Raspberry Pi 5 with Hailo-8L AI HAT+**.

---

## 1. Project Purpose & Engineering Objectives

Deploying state-of-the-art computer vision models like YOLO11 to constrained edge neural network accelerators requires far more than one-off shell scripts. It demands:
- **Defensible Provenance**: Cryptographic SHA-256 artifact verification, hardware telemetry, and environment tracking.
- **Strict Execution Boundaries**: Clean separation between portable ML workflows, Hailo compiler steps, and physical edge device runtime testing.
- **Quantitative Quality Gates**: Automatic validation ensuring INT8 Post-Training Quantization (PTQ) degradation does not exceed configurable thresholds.
- **No Silent Fallbacks**: The system never fakes metrics, simulates hardware execution on CPU, or claims unverified accelerator performance.

```text
Dataset
   ↓
Pre-flight Validation
   ↓
YOLO11 Training / Fine-tuning
   ↓
best.pt
   ↓
FP32 Static ONNX Export (1×3×640×640)
   ↓
ONNX Graph Validation & Smoke Test
   ↓
Deterministic INT8 Calibration (200 RGB samples)
   ↓
ONNX → HAR (Hailo Archive with .alls script)
   ↓
Hailo INT8 Post-Training Quantization
   ↓
Hailo-8L Compilation & Scheduling
   ↓
model.hef
   ↓
HailoRT Physical Smoke Test
   ↓
Accuracy Validation (ΔmAP50, ΔmAP50-95 gates)
   ↓
Performance & Latency Telemetry (p50, p95, p99, FPS)
   ↓
run_manifest.json & run_report.md
```

---

## 2. Supported YOLO11 Variants

The pipeline dynamically inspects model architecture, class counts, and task compatibility:
- `yolo11n` (Nano - recommended for maximum edge framerate)
- `yolo11s` (Small)
- `yolo11m` (Medium)
- `yolo11l` (Large)
- `yolo11x` (Extra Large)
- **Custom fine-tuned YOLO11 checkpoints** (`.pt`)

---

## 3. Supported Execution Boundaries

| Mode | Target Host | Allowed Phases | Physical Hardware Needed? |
| :--- | :--- | :--- | :--- |
| **`portable`** | Linux Workstation / Server / Mac / RPi 5 | `doctor`, `dataset_validation`, `training`, `onnx_export`, `onnx_validation`, `calibration`, `baseline evaluation` | No |
| **`hailo_compile`**| Linux x86_64 Host with Hailo DFC | `hailo_parse` (ONNX→HAR), `hailo_optimisation` (PTQ), `hailo_compile` (HAR→HEF), `hef_validation` | No (Compiler only) |
| **`hailo_runtime`**| Raspberry Pi 5 + Hailo-8L AI HAT+ | `hailo_validation` (physical smoke test), latency/FPS benchmarks, accelerator telemetry | **Yes** (`/dev/hailo0`) |

---

## 4. Software Prerequisites & Installation

### Option A: Local Virtual Environment

```bash
# Clone the repository
git clone https://github.com/imosudi/hailo-yolo11-compiler.git
cd hailo-yolo11-compiler

# Bootstrap python virtual environment
./scripts/bootstrap.sh
source .venv/bin/activate

# Install required requirement layers
pip install -e ".[dev]"                    # Core & development
pip install -e ".[training,onnx]"          # Training & ONNX export
```

### Option B: Rootless Podman / Docker

```bash
# Build portable compilation container
podman build -t hailo-yolo11-compiler -f Containerfile .

# Run diagnostics inside container
podman run --rm -v ./artifacts:/workspace/artifacts:Z hailo-yolo11-compiler doctor
```

---

## 5. Quickstart & CLI Command Reference

The principal entry point is `train_and_compile.py`:

```bash
# 1. Run environment diagnostics
python train_and_compile.py doctor

# 2. Inspect execution plan (Dry-Run mode)
python train_and_compile.py --dry-run all

# 3. Validate dataset and train YOLO11
python train_and_compile.py train --epochs 50 --batch 16

# 4. Export static FP32 ONNX model
python train_and_compile.py export

# 5. Extract deterministic calibration dataset (200 letterbox samples)
python train_and_compile.py calibrate --seed 42

# 6. Compile ONNX to Hailo-8L HEF
python train_and_compile.py compile

# 7. Run quantitative accuracy gates and benchmarks
python train_and_compile.py validate

# 8. Run end-to-end automated pipeline
python train_and_compile.py all
```

### Supported CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--config`, `-c` | Path to YAML configuration file | `config/config.yaml` |
| `--run-id` | Explicit run identifier | Auto-generated (`YYYYMMDD-HHMMSS-<hash>`) |
| `--model` | Model source weights path or variant | Configured in YAML |
| `--dataset` | Path to YOLO dataset YAML | Configured in YAML |
| `--device` | Compute device (`cpu`, `0`, `0,1`) | `auto` |
| `--epochs` | Number of training epochs | `100` |
| `--imgsz` | Image dimension (square, e.g. 640) | `640` |
| `--batch` | Batch size | `16` |
| `--seed` | Random seed | `42` |
| `--dry-run` | Display plan without writing artifacts | `false` |
| `--force` | Force regeneration of existing artifacts | `false` |
| `--execution-mode` | Boundary mode (`auto`, `portable`, `hailo_compile`, `hailo_runtime`) | `auto` |

---

## 6. Configuration Guide

Configurations follow strict precedence:
$$\text{CLI Arguments} > \text{Environment Variables} (\texttt{YOLO\_HAILO\_*}) > \text{YAML Configuration} > \text{Defaults}$$

Example `config/config.yaml`:

```yaml
project:
  name: "yolo11-hailo"
  seed: 42

model:
  source: "yolo11n.pt"
  variant: "n"
  task: "detect"

dataset:
  yaml: "data/dataset.yaml"
  calibration:
    source: "train"
    count: 200
    seed: 42

training:
  epochs: 100
  imgsz: 640
  batch: 16
  device: "auto"
  optimizer: "auto"
  patience: 50
  deterministic: true

export:
  imgsz: 640
  batch: 1
  dynamic: false
  precision: "fp32"
  simplify: true
  nms: "auto"

hailo:
  target: "hailo8l"
  sdk_mode: "auto"
  require_hardware: false

quantization:
  mode: "int8_ptq"
  optimisation: true

validation:
  enabled: true
  accuracy:
    map50_max_drop: 0.02      # 2% maximum allowable degradation
    map5095_max_drop: 0.02
  performance:
    warmup_iterations: 20
    measurement_iterations: 100

artifacts:
  root: "./artifacts"
```

---

## 7. Artifact Hierarchy & Reproducibility

Artifacts are saved in deterministic, versioned run directories with immutable manifests:

```text
artifacts/
├── latest -> runs/20261008-201422-a31f/   # Symlink to latest run
├── pytorch/best.pt                        # Canonical PyTorch weights
├── onnx/model.onnx                        # Canonical FP32 static ONNX
├── hailo/model.hef                        # Canonical Hailo-8L binary
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
        ├── pipeline.log
        ├── pipeline.jsonl
        ├── run_manifest.json              # Complete machine-readable ledger
        └── run_report.md                  # Human-readable markdown report
```

---

## 8. Testing

Run the automated pytest test suite:

```bash
# Run all unit and integration tests
pytest -v

# Run hardware-in-the-loop tests (executed only when /dev/hailo0 is present)
pytest -v -m hailo
```

---

## 9. Detailed Documentation Links

- [Architecture Specification](docs/architecture.md)
- [Pipeline Execution Guide](docs/pipeline.md)
- [Hailo-8L Compilation & Hardware Setup](docs/hailo.md)
- [Calibration & Quantization Contract](docs/calibration.md)
- [Reproducibility & Manifest Provenance](docs/reproducibility.md)
- [Troubleshooting & Error Codes](docs/troubleshooting.md)
- [Security Policy & Threat Model](SECURITY.md)

---

## 10. License

Licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) for details.
