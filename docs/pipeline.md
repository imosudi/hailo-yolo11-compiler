# Pipeline Execution Lifecycle & Data Flow Manual

This manual provides a detailed technical reference for the phase execution lifecycle, state transitions, input/output contracts, and data flow of the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [Architectural State Machine Lifecycle](#1-architectural-state-machine-lifecycle)
2. [End-to-End Execution Sequence](#2-end-to-end-execution-sequence)
3. [Phase-by-Phase Technical Specifications](#3-phase-by-phase-technical-specifications)
   - [Phase A: Dataset Pre-flight Validation](#phase-a-dataset-pre-flight-validation)
   - [Phase B: YOLO11 Training & Checkpoint Registration](#phase-b-yolo11-training--checkpoint-registration)
   - [Phase C: Static FP32 ONNX Export](#phase-c-static-fp32-onnx-export)
   - [Phase C2: ONNX Graph Validation & Smoke Inference](#phase-c2-onnx-graph-validation--smoke-inference)
   - [Phase D: Deterministic Calibration Dataset Sampling](#phase-d-deterministic-calibration-dataset-sampling)
   - [Phase E: Hailo Environment Detection](#phase-e-hailo-environment-detection)
   - [Phase F: ONNX → HAR Translation](#phase-f-onnx--har-translation)
   - [Phase G: INT8 Post-Training Quantisation (PTQ)](#phase-g-int8-post-training-quantisation-ptq)
   - [Phase H: Hailo-8L Compilation & Scheduling](#phase-h-hailo-8l-compilation--scheduling)
   - [Phase J: HailoRT Physical Accelerator Smoke Test](#phase-j-hailort-physical-accelerator-smoke-test)
   - [Phase I: Quantitative Accuracy Validation Gates](#phase-i-quantitative-accuracy-validation-gates)
   - [Phase K: Performance Telemetry & Latency Benchmarks](#phase-k-performance-telemetry--latency-benchmarks)
4. [Run Manifest & Final Report Generation](#4-run-manifest--final-report-generation)

---

## 1. Architectural State Machine Lifecycle

Every phase in the pipeline is managed by an explicit, deterministic state machine (`yolo_hailo_mlops.state.StateMachine`). Transitions between states are tracked with high-resolution timestamps:

```text
       ┌───────────┐
       │  PENDING  │
       └─────┬─────┘
             │ Start Phase
             ▼
       ┌───────────┐
       │  RUNNING  │
       └──┬───┬───┬┘
          │   │   │
  Success │   │   │ Failure / Exception
          │   │   └───────────────────────────────► ┌───────────┐
          │   │                                     │  FAILED   │
          │   │ Boundary Constraint (Missing HW)    └───────────┘
          │   └───────────────────────────────────► ┌───────────────┐
          │                                         │ NOT_EXECUTED  │
          ▼                                         └───────────────┘
    ┌───────────┐
    │  SUCCESS  │
    └───────────┘
```

### State Definitions

- **`PENDING`**: Registered phase scheduled for execution.
- **`RUNNING`**: Phase currently executing.
- **`SUCCESS`**: Phase completed successfully; all output artefacts are cryptographically hashed and verified.
- **`FAILED`**: Phase raised an unrecoverable exception; structured diagnostic block emitted.
- **`SKIPPED`**: Phase intentionally bypassed (e.g. valid cached artefact exists and `--force` was not set).
- **`NOT_EXECUTED`**: Phase could not execute due to environment boundary constraints (e.g. host lacks Hailo DFC or Raspberry Pi lacks `/dev/hailo0`).
- **`CANCELLED`**: Execution gracefully terminated by `SIGINT` (Ctrl+C) or `SIGTERM`.

---

## 2. End-to-End Execution Sequence

```text
Dataset YAML
     │
     ▼
[Phase A: dataset_validation]       ──► Produces validation statistics & audit
     │
     ▼
[Phase B: training]                 ──► Emits artifacts/runs/<run_id>/pytorch/best.pt
     │
     ▼
[Phase C: onnx_export]              ──► Emits artifacts/runs/<run_id>/onnx/model.onnx
     │
     ▼
[Phase C2: onnx_validation]         ──► Asserts 1×3×640×640 static shapes & no NaNs
     │
     ▼
[Phase D: calibration]              ──► Emits artifacts/runs/<run_id>/calibration/calib_data.npy
     │
     ▼
[Phase E: hailo_environment]        ──► Inspects Hailo DFC and HailoRT capabilities
     │
     ▼
[Phase F: hailo_parse]              ──► Translates ONNX to HAR (6-head detection conv slicing), emits yolo11.alls
     │
     ▼
[Phase G: hailo_optimisation]       ──► Executes INT8 PTQ using calib_data.npy
     │
     ▼
[Phase H: hailo_compile]            ──► Synthesises artifacts/runs/<run_id>/hailo/model.hef (4 contexts on Hailo-8L)
     │
     ▼
[Phase J: hailo_runtime]            ──► Physical smoke test on /dev/hailo0
     │
     ▼
[Phase I: accuracy_validation]      ──► Enforces ΔmAP50 ≤ 0.05 quality gates
     │
     ▼
[Phase K: performance_benchmarks]   ──► Measures p50, p95, p99 latency & FPS
     │
     ▼
[Manifest Ledger & Report]          ──► Writes run_manifest.json & run_report.md
```

---

## 3. Phase-by-Phase Technical Specifications

### Phase A: Dataset Pre-flight Validation
- **Phase Identifier**: `dataset_validation`
- **Execution Boundary**: `portable`
- **Inputs**: `dataset.yaml` path.
- **Operations**: Validates 18 integrity criteria (syntax, paths, coordinates, leakage, corruption).
- **Outputs**: Validation statistics dictionary, pre-flight report.
- **Assertion**: If critical defects exist, raises `DatasetValidationError` (`E-DATA-*`).

---

### Phase B: YOLO11 Training & Checkpoint Registration
- **Phase Identifier**: `training`
- **Execution Boundary**: `portable`
- **Inputs**: Base model (`yolo11n.pt` or custom checkpoint), dataset YAML, hyperparameters.
- **Operations**: Deterministic training via Ultralytics engine; checkpoint tracking.
- **Outputs**:
  - `artifacts/runs/<run_id>/pytorch/best.pt`
  - Canonical symlink `artifacts/pytorch/best.pt`
- **Assertion**: Verifies output `.pt` file exists, is non-empty, and loadable by PyTorch.

---

### Phase C: Static FP32 ONNX Export
- **Phase Identifier**: `onnx_export`
- **Execution Boundary**: `portable`
- **Inputs**: `artifacts/runs/<run_id>/pytorch/best.pt`.
- **Operations**: Ultralytics ONNX exporter with static axes (`dynamic=False`), $640 \times 640$, batch 1, `onnx-simplifier`.
- **Outputs**:
  - `artifacts/runs/<run_id>/onnx/model.onnx`
  - Canonical symlink `artifacts/onnx/model.onnx`
- **Assertion**: Raises `ExportError` (`E-EXP-*`) if export fails.

---

### Phase C2: ONNX Graph Validation & Smoke Inference
- **Phase Identifier**: `onnx_validation`
- **Execution Boundary**: `portable`
- **Inputs**: `artifacts/runs/<run_id>/onnx/model.onnx`.
- **Operations**:
  - Validates ONNX protobuf graph with `onnx.checker`.
  - Verifies input tensor shape is strictly $(1, 3, 640, 640)$ and datatype is `float32`.
  - Executes ONNX Runtime CPU smoke inference on a dummy frame; asserts zero `NaN` or `Inf` outputs.
- **Outputs**: ONNX validation audit metadata.
- **Assertion**: Raises `ONNXValidationError` (`E-ONNX-*`) if static geometry is violated.

---

### Phase D: Deterministic Calibration Dataset Sampling
- **Phase Identifier**: `calibration`
- **Execution Boundary**: `portable`
- **Inputs**: Training dataset images or custom directory, sample count (default 200), seed 42.
- **Operations**: Deterministic sampling, RGB conversion, letterbox aspect ratio preservation (fill 114), normalisation to $[0.0, 1.0]$.
- **Outputs**:
  - `artifacts/runs/<run_id>/calibration/calib_data.npy` (Array shape: $200 \times 640 \times 640 \times 3$)
  - `artifacts/runs/<run_id>/calibration/images/`
  - `artifacts/runs/<run_id>/calibration/calibration_manifest.json`
  - `artifacts/runs/<run_id>/calibration/calibration_statistics.json`

---

### Phase E: Hailo Environment Detection
- **Phase Identifier**: `hailo_environment`
- **Execution Boundary**: `portable`
- **Inputs**: System environment.
- **Operations**: Checks for Hailo DFC Python API (`hailo_sdk_client`), DFC CLI (`hailo`), HailoRT library, and `/dev/hailo0`.
- **Outputs**: Capability report stored in manifest.

---

### Phase F: ONNX → HAR Translation
- **Phase Identifier**: `hailo_parse`
- **Execution Boundary**: `hailo_compile`
- **Inputs**: `model.onnx`, model script `yolo11.alls`.
- **Operations**: Translates ONNX operators into Hailo Archive intermediate representation. Configures zero-latency hardware input normalisation.
- **Outputs**:
  - `artifacts/runs/<run_id>/hailo/model.har`
  - `artifacts/runs/<run_id>/hailo/yolo11.alls`

---

### Phase G: INT8 Post-Training Quantisation (PTQ)
- **Phase Identifier**: `hailo_optimisation`
- **Execution Boundary**: `hailo_compile`
- **Inputs**: `model.har`, `calib_data.npy`.
- **Operations**: Computes layer-wise dynamic scale factors and zero-points. Optimises activation dynamic ranges.
- **Outputs**:
  - `artifacts/runs/<run_id>/hailo/model_quantized.har`

---

### Phase H: Hailo-8L Compilation & Scheduling
- **Phase Identifier**: `hailo_compile`
- **Execution Boundary**: `hailo_compile`
- **Inputs**: `model_quantized.har`.
- **Operations**: Maps compute layers, routes dataflow, schedules execution, and compiles hardware binary for `hailo8l`.
- **Outputs**:
  - `artifacts/runs/<run_id>/hailo/model.hef`
  - Canonical symlink `artifacts/hailo/model.hef`

---

### Phase J: HailoRT Physical Accelerator Smoke Test
- **Phase Identifier**: `hailo_runtime`
- **Execution Boundary**: `hailo_runtime`
- **Inputs**: `model.hef`, `/dev/hailo0`.
- **Operations**: Loads HEF onto physical NPU, configures virtual stream, sends test frame, verifies output tensor dimensions.
- **Outputs**: Hardware execution status. If hardware is absent, records `HARDWARE_VALIDATION = NOT_EXECUTED`.

---

### Phase I: Quantitative Accuracy Validation Gates
- **Phase Identifier**: `accuracy_validation`
- **Execution Boundary**: `portable` / `hailo_compile`
- **Inputs**: PyTorch model, ONNX model, Hailo model (or emulator), validation dataset split.
- **Operations**: Computes mAP50, mAP50-95, precision, and recall. Enforces $\Delta\text{mAP} \le 0.02$ quality gates.
- **Outputs**: Accuracy metrics table, gate evaluation status. Raises `AccuracyValidationError` (`E-ACC-005`) if gates fail.

---

### Phase K: Performance Telemetry & Latency Benchmarks
- **Phase Identifier**: `performance_benchmarks`
- **Execution Boundary**: `hailo_runtime`
- **Inputs**: Physical accelerator, 20 warm-up iterations, 100 measurement iterations.
- **Operations**: High-resolution latency profiling ($p_{50}, p_{95}, p_{99}$), FPS derivation, SoC temperature, and throttling tracking.
- **Outputs**: Performance metrics table with strict provenance tags.

---

## 4. Run Manifest & Final Report Generation

At the conclusion of the pipeline:
1. `run_manifest.json` is generated atomically, recording:
   - Full environment configuration and Git commit.
   - Per-phase start times, durations, and exit states.
   - Comprehensive dictionary of all produced artefacts with byte sizes and SHA-256 hashes.
   - Evaluation metrics and telemetry with provenance classifications.
2. `run_report.md` is generated, formatting execution metrics into an executive summary markdown document.
3. Canonical symlink `artifacts/latest` is atomically updated to point to the new run directory.
