# Troubleshooting & Error Codes Reference Manual

This manual provides an exhaustive diagnostic taxonomy, error code catalogue, remediation procedures, and decision trees for resolving issues encountered across the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [Diagnostic Architecture & Error Structure](#1-diagnostic-architecture--error-structure)
2. [Automated System Diagnostics (`doctor`)](#2-automated-system-diagnostics-doctor)
3. [Configuration Errors (`E-CFG-*`)](#3-configuration-errors-e-cfg-)
4. [Dataset Validation Errors (`E-DATA-*`)](#4-dataset-validation-errors-e-data-)
5. [Training & Checkpoint Errors (`E-TRAIN-*`)](#5-training--checkpoint-errors-e-train-)
6. [ONNX Export & Graph Validation Errors (`E-EXP-*`, `E-ONNX-*`)](#6-onnx-export--graph-validation-errors-e-exp--e-onnx-)
7. [Calibration Errors (`E-CAL-*`)](#7-calibration-errors-e-cal-)
8. [Hailo Compilation & Hardware Errors (`E-HAILO-*`)](#8-hailo-compilation--hardware-errors-e-hailo-)
9. [Accuracy Degradation Gate Violations (`E-ACC-*`)](#9-accuracy-degradation-gate-violations-e-acc-)
10. [Artefact Management & I/O Errors (`E-ART-*`)](#10-artefact-management--io-errors-e-art-)
11. [Diagnostic Flowcharts & Decision Trees](#11-diagnostic-flowcharts--decision-trees)

---

## 1. Diagnostic Architecture & Error Structure

All pipeline exceptions inherit from `YoloHailoError` and emit a standardised diagnostic block to stderr and the structured log.

### Diagnostic Block Anatomy

```text
================================================================================
 PIPELINE FAILURE DIAGNOSTIC
================================================================================
[E-HAILO-RT-002] hailo_runtime: Physical Hailo accelerator device node (/dev/hailo0) missing
Phase:       hailo_runtime
Artefact:    /dev/hailo0
Remediation: Check PCIe ribbon cable connection on Raspberry Pi 5; verify 'dtparam=pciex1' in /boot/firmware/config.txt; verify 'dmesg | grep hailo'.
================================================================================
```

Every diagnostic block informs the user:
- **Error Code**: Precise identifier (`E-<DOMAIN>-<NUMBER>`).
- **Phase**: Exact pipeline state machine stage where the failure occurred.
- **Root Cause**: Plain-language explanation of what triggered the fault.
- **Affected Artefact**: The specific file, directory, or hardware device node involved.
- **Remediation Action**: Actionable command or configuration edit to resolve the issue.

---

## 2. Automated System Diagnostics (`doctor`)

Before manual debugging, run the automated environment doctor:

```bash
python train_and_compile.py doctor
```

The doctor interrogates:
- Host operating system, architecture, and kernel release.
- Python runtime version and virtual environment status.
- Core ML packages (PyTorch, TorchVision, Ultralytics, ONNX, ONNX Runtime).
- Hailo software (Hailo Dataflow Compiler, HailoRT userland, HailoRT CLI).
- Edge accelerator hardware (PCIe device `1e60:`, device node `/dev/hailo0`).
- Configured model checkpoints and dataset paths.

---

## 3. Configuration Errors (`E-CFG-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-CFG-001` | `configuration` | YAML parsing syntax error in config file | Validate YAML indentation; check for unescaped colons or unbalanced quotes using a YAML linter. |
| `E-CFG-002` | `configuration` | Missing required configuration file | Check path supplied to `--config` or verify existence of `config/config.yaml`. |
| `E-CFG-003` | `configuration` | Invalid seed value | Ensure `project.seed` is an integer $\ge 0$. |
| `E-CFG-004` | `configuration` | Invalid model variant specified | Set `model.variant` to one of `['n', 's', 'm', 'l', 'x', 'custom']`. |
| `E-CFG-005` | `configuration` | Missing dataset YAML path | Declare `dataset.yaml: "path/to/dataset.yaml"` in configuration or pass `--dataset`. |
| `E-CFG-006` | `configuration` | Invalid calibration count | Set `dataset.calibration.count` between 100 and 200 (or supply explicit test override). |
| `E-CFG-007` | `configuration` | Invalid training epochs | Set `training.epochs` to a positive integer $\ge 1$. |
| `E-CFG-008` | `configuration` | Invalid batch size | Set `training.batch` to a positive integer $\ge 1$. |
| `E-CFG-009` | `configuration` | Invalid image size | Set `training.imgsz: 640`. Hailo-8L canonical contract requires 640. |
| `E-CFG-010` | `configuration` | Invalid training device | Set `training.device` to `"auto"`, `"cpu"`, or a CUDA device ID such as `"0"`. |
| `E-CFG-011` | `configuration` | Invalid non-static export image size | Set `export.imgsz: 640`. |
| `E-CFG-012` | `configuration` | Non-static export batch size | Set `export.batch: 1`. Hailo-8L compiler requires static batch dimension 1. |
| `E-CFG-013` | `configuration` | Dynamic export axes enabled | Set `export.dynamic: false`. Dynamic spatial/batch axes are unsupported by Hailo-8L. |
| `E-CFG-014` | `configuration` | Invalid export precision | Set `export.precision: "fp32"`. |
| `E-CFG-015` | `configuration` | Unsupported Hailo target architecture | Set `hailo.target: "hailo8l"`. Raspberry Pi 5 AI HAT+ is strictly Hailo-8L. |
| `E-CFG-016` | `configuration` | Invalid accuracy gate threshold | Set `validation.accuracy.map50_max_drop` between `0.0` and `1.0`. |
| `E-CFG-017` | `configuration` | Invalid measurement iterations | Set `validation.performance.measurement_iterations` to $\ge 10$. |

---

## 4. Dataset Validation Errors (`E-DATA-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-DATA-001` | `dataset_validation` | Dataset YAML descriptor not found | Verify file path in `config.yaml` or run `python train_and_compile.py dataset --dataset path/to/dataset.yaml`. |
| `E-DATA-002` | `dataset_validation` | Malformed YAML in dataset descriptor | Check indentation and syntax of `dataset.yaml`. |
| `E-DATA-003` | `dataset_validation` | Root dataset path does not exist | Verify `path:` in `dataset.yaml` points to an existing directory. |
| `E-DATA-004` | `dataset_validation` | Missing class definition (`nc` or `names`) | Add `nc: <int>` and `names: [...]` mapping to `dataset.yaml`. |
| `E-DATA-005` | `dataset_validation` | Missing train or val image paths | Define both `train:` and `val:` directory paths in `dataset.yaml`. |
| `E-DATA-006` | `dataset_validation` | Out-of-bounds bounding box coordinates | Check labels: ensure $X, Y, W, H$ are normalised floats strictly in $[0.0, 1.0]$. |
| `E-DATA-007` | `dataset_validation` | Train/validation data leakage | Eliminate duplicate images sharing identical content hashes between train and val splits. |
| `E-DATA-008` | `dataset_validation` | Insufficient calibration candidates | Verify training split contains at least 200 clean, uncorrupted images. |
| `E-DATA-009` | `dataset_validation` | Corrupt image file detected | Remove or re-encode corrupted image files flagged in the diagnostic report. |
| `E-DATA-010` | `dataset_validation` | Class ID exceeds declared `nc` | Check label files: class IDs must satisfy $0 \le \text{class\_id} < nc$. |

---

## 5. Training & Checkpoint Errors (`E-TRAIN-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-TRAIN-001` | `training` | Training run failed with runtime exception | Inspect `pipeline.log` for CUDA out-of-memory errors; reduce `--batch` size. |
| `E-TRAIN-002` | `training` | Output checkpoint `best.pt` not created | Verify training completed at least one epoch; check disk write permissions. |
| `E-TRAIN-003` | `training` | Missing base checkpoint | Download base weights (`yolo11n.pt`) or provide path to pre-existing `.pt` file. |
| `E-TRAIN-007` | `training` | Ultralytics package missing | Run `pip install -e .[training]` to install PyTorch and Ultralytics. |

---

## 6. ONNX Export & Graph Validation Errors (`E-EXP-*`, `E-ONNX-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-EXP-001` | `onnx_export` | Missing PyTorch checkpoint input | Run training stage first or pass `--model path/to/best.pt`. |
| `E-EXP-004` | `onnx_export` | Ultralytics ONNX export execution failed | Ensure `onnx` and `onnxsim` are installed; verify PyTorch opset compatibility. |
| `E-ONNX-001` | `onnx_validation` | Exported ONNX file missing on disk | Check export stage logs to identify export termination cause. |
| `E-ONNX-002` | `onnx_validation` | ONNX model graph structure invalid | Run `python -c "import onnx; onnx.checker.check_model('model.onnx')"` to inspect graph integrity. |
| `E-ONNX-003` | `onnx_validation` | Exported ONNX violates static contract | Re-export with `export.batch: 1` and `export.dynamic: false`. Input shape must be $1 \times 3 \times 640 \times 640$. |
| `E-ONNX-004` | `onnx_validation` | Smoke inference produced NaN or Inf | Inspect model weights for numerical instability; retrain with lower learning rate. |

---

## 7. Calibration Errors (`E-CAL-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-CAL-001` | `calibration` | Source image directory not found | Ensure training image split exists or verify `custom_dir` path. |
| `E-CAL-002` | `calibration` | Insufficient clean images for sampling | Ensure source folder has at least `calibration.count` uncorrupted images. |
| `E-CAL-003` | `calibration` | Calibration preprocessor failed | Inspect image file formats; ensure all sampled images can be read by Pillow. |
| `E-CAL-004` | `calibration` | Output `calib_data.npy` array missing | Verify disk space and write permissions in `artifacts/runs/<run_id>/calibration/`. |
| `E-CAL-005` | `calibration` | NumPy calibration tensor dimension mismatch | Ensure tensor shape is strictly $(N, 640, 640, 3)$ matching the RGB contract. |

---

## 8. Hailo Compilation & Hardware Errors (`E-HAILO-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-HAILO-ENV-001` | `hailo_environment` | Hailo DFC or HailoRT not installed | Install Hailo DFC v3.28+ on compilation host or use container. |
| `E-HAILO-ENV-002` | `hailo_environment` | Incompatible Hailo DFC version | Upgrade Hailo DFC to supported version v3.28+. |
| `E-HAILO-PARSE-001` | `hailo_parse` | Input ONNX model missing | Run export stage first or supply valid path via `--model`. |
| `E-HAILO-PARSE-004` | `hailo_parse` | Hailo ONNX parser failure | Inspect unsupported ONNX operators; ensure NMS is not embedded in the ONNX graph (`nms: auto`). |
| `E-HAILO-OPT-001` | `hailo_optimisation` | Input HAR archive not found | Ensure parse stage succeeded and emitted `model.har`. |
| `E-HAILO-OPT-002` | `hailo_optimisation` | Calibration data `calib_data.npy` missing | Run calibration stage first or check file permissions. |
| `E-HAILO-OPT-005` | `hailo_optimisation` | Quantisation failed | Check calibration set preprocessing; ensure `calib_data.npy` matches model input dimensions. |
| `E-HAILO-COMP-001` | `hailo_compile` | Quantised HAR not found | Ensure PTQ optimisation stage succeeded and emitted `model_quantized.har`. |
| `E-HAILO-COMP-005` | `hailo_compile` | Compilation / layer routing failed | Verify model capacity fits Hailo-8L resource limits (13 TOPS); switch to `yolo11n` or `yolo11s`. |
| `E-HAILO-RT-001` | `hailo_runtime` | HailoRT runtime library missing | Run `sudo apt install -y hailo-all` on Raspberry Pi 5. |
| `E-HAILO-RT-002` | `hailo_runtime` | Physical Hailo accelerator missing | Verify Raspberry Pi 5 PCIe HAT+ ribbon cable connection; check `dmesg \| grep hailo`. |
| `E-HAILO-RT-003` | `hailo_runtime` | HEF binary loading failed | Recompile HEF; verify target architecture matches `hailo8l`. |
| `E-HAILO-RT-004` | `hailo_runtime` | Physical inference execution exception | Check system power supply (use official 27W USB-C); ensure PCIe Gen 3 is stable. |

---

## 9. Accuracy Degradation Gate Violations (`E-ACC-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-ACC-001` | `accuracy_validation` | Evaluation dataset missing | Verify validation split in `dataset.yaml` exists and contains labelled images. |
| `E-ACC-002` | `accuracy_validation` | Evaluation execution failed | Check PyTorch and ONNX Runtime dependencies. |
| `E-ACC-005` | `accuracy_validation` | mAP degradation exceeded allowed gate threshold | Increase calibration dataset image count (e.g. 200 samples); verify calibration set is representative of validation distribution; or increase `map50_max_drop`. |

---

## 10. Artefact Management & I/O Errors (`E-ART-*`)

| Error Code | Phase | Cause | Actionable Remediation |
| :--- | :--- | :--- | :--- |
| `E-ART-001` | `artifacts` | Destination directory not writable | Verify write permissions on `./artifacts` directory. |
| `E-ART-002` | `artifacts` | Atomic file write failed | Check available filesystem disk space with `df -h`. |
| `E-ART-003` | `artifacts` | Cryptographic SHA-256 mismatch | Detects corrupted or tampered files. Re-run phase with `--force`. |

---

## 11. Diagnostic Flowcharts & Decision Trees

### Decision Tree: Resolving Missing Hailo Accelerator (`/dev/hailo0`)

```text
/dev/hailo0 missing
  │
  ├─► Check dmesg: dmesg | grep -i hailo
  │     ├─► No PCIe device seen?
  │     │     └─► Power off, inspect 16-pin FPC ribbon cable orientation:
  │     │           - Pi 5 side: Contacts face HDMI ports.
  │     │           - HAT+ side: Contacts face metal pins.
  │     │           - Locking collars engaged firmly.
  │     └─► Device seen but driver error?
  │           └─► Verify kernel module: sudo modprobe hailo_pci
  │
  ├─► Check /boot/firmware/config.txt:
  │     ├─► Ensure dtparam=pciex1 is present.
  │     └─► Ensure dtparam=pciex1_gen=3 is present.
  │
  └─► Check permissions:
        └─► sudo usermod -aG hailo $USER && newgrp hailo
```

### Decision Tree: Resolving Accuracy Gate Failure (`E-ACC-005`)

```text
Accuracy Gate Failed (ΔmAP > 0.02)
  │
  ├─► Is the calibration set representative of the operational domain?
  │     └─► If training images are biased, provide custom domain images via:
  │           dataset.calibration.source: "custom"
  │           dataset.calibration.custom_dir: "path/to/operational_images"
  │
  ├─► Did the calibration set contain enough images?
  │     └─► Ensure dataset.calibration.count: 200 (maximum recommended for PTQ).
  │
  └─► Is the model variant too large or quantisation-sensitive?
        └─► Verify that input normalisation in yolo11.alls matches [0.0, 255.0].
        └─► Consider tuning validation.accuracy.map50_max_drop if 2% is overly strict for the task.
```
