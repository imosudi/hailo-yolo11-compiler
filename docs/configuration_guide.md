# Configuration System & Parameter Reference Manual

This document provides a comprehensive reference for configuring the **`hailo-yolo11-compiler`** pipeline. It details the multi-tier precedence model, complete parameter dictionary, environment variable mappings, and real-world deployment profiles.

---

## Table of Contents

1. [Configuration Precedence Hierarchy](#1-configuration-precedence-hierarchy)
2. [Complete Schema Dictionary](#2-complete-schema-dictionary)
   - [`project`](#project)
   - [`model`](#model)
   - [`dataset`](#dataset)
   - [`training`](#training)
   - [`export`](#export)
   - [`hailo`](#hailo)
   - [`quantisation` / `quantization`](#quantisation--quantization)
   - [`validation`](#validation)
   - [`artefacts` / `artifacts`](#artefacts--artifacts)
3. [Environment Variable Reference (`YOLO_HAILO_*`)](#3-environment-variable-reference-yolo_hailo_)
4. [Precedence Resolution Example](#4-precedence-resolution-example)
5. [Operational Deployment Profiles](#5-operational-deployment-profiles)
   - [Profile A: Maximum Framerate Edge Deployment (YOLO11n)](#profile-a-maximum-framerate-edge-deployment-yolo11n)
   - [Profile B: High-Precision Defect Inspection (YOLO11s/m)](#profile-b-high-precision-defect-inspection-yolo11sm)
   - [Profile C: CI/CD Automated Regression Testing](#profile-c-cicd-automated-regression-testing)

---

## 1. Configuration Precedence Hierarchy

Configurations are evaluated in a strict four-tier order. Higher tiers completely override lower tiers:

$$\textbf{Tier 1: CLI Arguments} > \textbf{Tier 2: Environment Variables} > \textbf{Tier 3: YAML File} > \textbf{Tier 4: Code Defaults}$$

```text
[1. CLI Flag]         --epochs 50               (Highest Priority)
        ▲
[2. Environment]      YOLO_HAILO_TRAINING_EPOCHS=100
        ▲
[3. YAML File]        training: epochs: 150
        ▲
[4. Code Default]     epochs: 100               (Base Fallback)
```

---

## 2. Complete Schema Dictionary

The master configuration file is located at `config/config.yaml`. Both British English (`quantisation`, `artefacts`) and American English (`quantization`, `artifacts`) keys are supported.

### `project`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | `"yolo11-hailo"` | Project name used for logging and labelling manifests. |
| `seed` | `int` | `42` | Master random seed propagated across PyTorch, NumPy, and calibration samplers. |

---

### `model`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | `"yolo11n.pt"` | Base model checkpoint path or official Ultralytics weight name. |
| `variant` | `str` | `"n"` | Architecture variant: `'n'`, `'s'`, `'m'`, `'l'`, `'x'`, or `'custom'`. |
| `task` | `str` | `"detect"` | Vision task. Only `"detect"` is supported for Hailo-8L canonical contract. |

---

### `dataset`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `yaml` | `str` | `"data/dataset.yaml"` | Path to standard YOLO dataset YAML descriptor. |
| `calibration.source` | `str` | `"train"` | Calibration source: `"train"` (samples from training split) or `"custom"`. |
| `calibration.count` | `int` | `200` | Number of calibration images (recommended range: 100–200). |
| `calibration.seed` | `int` | `42` | Pseudo-random seed for deterministic calibration image selection. |
| `calibration.custom_dir`| `str \| null` | `null` | Path to custom calibration directory when `source: "custom"`. |

---

### `training`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `epochs` | `int` | `100` | Number of training epochs. |
| `imgsz` | `int` | `640` | Square image resolution (must remain 640 for Hailo-8L canonical export). |
| `batch` | `int` | `16` | Training batch size. |
| `device` | `str` | `"auto"` | Compute device: `"auto"`, `"cpu"`, `"0"`, or `"0,1"`. |
| `optimizer` | `str` | `"auto"` | Training optimiser: `"auto"`, `"SGD"`, `"Adam"`, or `"AdamW"`. |
| `lr0` | `float \| null` | `null` | Initial learning rate (`null` uses Ultralytics default). |
| `lrf` | `float \| null` | `null` | Final learning rate factor (`null` uses Ultralytics default). |
| `patience` | `int` | `50` | Early stopping patience in epochs without metric improvement. |
| `deterministic` | `bool` | `true` | Enforces deterministic PyTorch cuDNN operations. |

---

### `export`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `imgsz` | `int` | `640` | Export spatial resolution. Hailo-8L requires static 640. |
| `batch` | `int` | `1` | Export batch size. Hailo-8L requires strictly static batch 1. |
| `dynamic` | `bool` | `false` | Must remain `false`. Dynamic axes are strictly prohibited. |
| `precision` | `str` | `"fp32"` | Export precision (`"fp32"` is the canonical compiler contract). |
| `simplify` | `bool` | `true` | Applies `onnx-simplifier` to fold constant subgraphs. |
| `opset` | `int \| null` | `null` | ONNX opset version (`null` selects recommended opset 17). |
| `nms` | `str` | `"auto"` | NMS policy: `"auto"`, `"hailo_runtime"`, or `"embedded"`. |

---

### `hailo`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target` | `str` | `"hailo8l"` | Target accelerator architecture. Strictly `"hailo8l"`. |
| `sdk_mode` | `str` | `"auto"` | DFC interaction mode: `"auto"`, `"api"`, or `"cli"`. |
| `require_hardware` | `bool` | `false` | If `true`, fails execution if `/dev/hailo0` is absent. |
| `har_path` | `str \| null`| `null` | Path to existing HAR if resuming compilation from archive. |
| `hef_path` | `str \| null`| `null` | Path to existing HEF if running runtime validation directly. |
| `compiler_optimization_level` | `int` | `0` | Hailo compiler optimisation level in `.alls` (0 to 3). |
| `calib_batch_size` | `int` | `1` | Batch size used during PTQ calibration feeding. |

---

### `quantisation` / `quantization`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `str` | `"int8_ptq"` | Quantisation method. `"int8_ptq"` is standard for Hailo DFC. |
| `optimisation` | `bool` | `true` | Enables Hailo PTQ optimisation algorithms. |

---

### `validation`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` | Whether to perform quantitative validation and benchmarks. |
| `accuracy.map50_max_drop` | `float` | `0.02` | Maximum allowable drop in mAP50 ($2\%$) between PyTorch and Hailo. |
| `accuracy.map5095_max_drop` | `float` | `0.02` | Maximum allowable drop in mAP50-95 ($2\%$) between PyTorch and Hailo. |
| `performance.warmup_iterations` | `int` | `20` | Warm-up inference passes on physical Hailo-8L before timing. |
| `performance.measurement_iterations` | `int` | `100` | Timed inference passes used to calculate latency percentiles. |

---

### `artefacts` / `artifacts`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"./artifacts"` | Directory for canonical pointers and versioned run outputs. |

---

## 3. Environment Variable Reference (`YOLO_HAILO_*`)

Any configuration parameter can be overridden using environment variables without modifying `config.yaml`:

| Environment Variable | Target Parameter | Accepted Values |
| :--- | :--- | :--- |
| `YOLO_HAILO_CONFIG` | Config file path | Path string (e.g. `/path/to/custom.yaml`) |
| `YOLO_HAILO_PROJECT_SEED` | `project.seed` | Integer (e.g. `1337`) |
| `YOLO_HAILO_MODEL_SOURCE` | `model.source` | Path or model name |
| `YOLO_HAILO_DATASET_YAML` | `dataset.yaml` | Path to dataset YAML |
| `YOLO_HAILO_CALIB_COUNT` | `dataset.calibration.count` | Integer (e.g. `200`) |
| `YOLO_HAILO_TRAINING_EPOCHS` | `training.epochs` | Integer (e.g. `50`) |
| `YOLO_HAILO_TRAINING_BATCH` | `training.batch` | Integer (e.g. `32`) |
| `YOLO_HAILO_TRAINING_DEVICE` | `training.device` | String (`"cpu"`, `"0"`) |
| `YOLO_HAILO_HAILO_TARGET` | `hailo.target` | String (`"hailo8l"`) |
| `YOLO_HAILO_REQUIRE_HARDWARE` | `hailo.require_hardware` | `"true"`, `"1"`, `"false"` |
| `YOLO_HAILO_MAP50_MAX_DROP` | `validation.accuracy.map50_max_drop` | Float (e.g. `0.015`) |
| `YOLO_HAILO_MAP5095_MAX_DROP` | `validation.accuracy.map5095_max_drop`| Float (e.g. `0.015`) |
| `YOLO_HAILO_WARMUP` | `validation.performance.warmup_iterations` | Integer (e.g. `50`) |
| `YOLO_HAILO_MEASURE` | `validation.performance.measurement_iterations` | Integer (e.g. `200`) |
| `YOLO_HAILO_ARTIFACTS_ROOT` | `artifacts.root` | Directory path |

---

## 4. Precedence Resolution Example

Consider the following scenario:

1. **`config/config.yaml`** declares:
   ```yaml
   training:
     epochs: 100
     batch: 16
   ```
2. **Environment variable** is exported:
   ```bash
   export YOLO_HAILO_TRAINING_EPOCHS=75
   ```
3. **CLI Command** is executed:
   ```bash
   python train_and_compile.py train --batch 32
   ```

**Final Resolved Configuration:**
- `training.epochs`: **`75`** (Environment variable overrides YAML)
- `training.batch`: **`32`** (CLI argument overrides YAML and defaults)
- `training.imgsz`: **`640`** (Taken from YAML)

---

## 5. Operational Deployment Profiles

### Profile A: Maximum Framerate Edge Deployment (YOLO11n)
*Optimised for maximum real-time FPS on Raspberry Pi 5:*

```yaml
project:
  name: "edge-realtime-nano"
  seed: 42

model:
  source: "yolo11n.pt"
  variant: "n"

training:
  epochs: 80
  batch: 16
  imgsz: 640
  optimizer: "AdamW"

export:
  batch: 1
  dynamic: false
  precision: "fp32"
  simplify: true
  nms: "auto"

hailo:
  target: "hailo8l"
  compiler_optimization_level: 2

validation:
  accuracy:
    map50_max_drop: 0.02
```

---

### Profile B: High-Precision Defect Inspection (YOLO11s/m)
*Optimised for industrial quality control where accuracy drop must not exceed 1%:*

```yaml
project:
  name: "industrial-inspection"
  seed: 42

model:
  source: "yolo11s.pt"
  variant: "s"

dataset:
  yaml: "data/defects.yaml"
  calibration:
    count: 200
    seed: 42

training:
  epochs: 150
  batch: 16
  patience: 40

validation:
  accuracy:
    map50_max_drop: 0.01      # Strict 1% maximum drop gate
    map5095_max_drop: 0.015
  performance:
    warmup_iterations: 30
    measurement_iterations: 200
```

---

### Profile C: CI/CD Automated Regression Testing
*Optimised for fast automated test pipelines without physical hardware:*

```yaml
project:
  name: "ci-regression"
  seed: 42

model:
  source: "yolo11n.pt"
  variant: "n"

training:
  epochs: 1                   # Smoke training pass
  batch: 4

dataset:
  calibration:
    count: 50

hailo:
  require_hardware: false

validation:
  enabled: false              # Bypasses hardware-dependent evaluations
```
