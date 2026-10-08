# Troubleshooting & Error Codes Reference

## Diagnostic Taxonomy

The pipeline categorizes all failures into structured error codes. When an error occurs, the CLI prints a standardized diagnostic block containing the error code, phase, cause, affected artifact, and remediation action.

---

### Configuration Errors (`E-CFG-*`)

| Error Code | Phase | Cause | Remediation |
| :--- | :--- | :--- | :--- |
| `E-CFG-001` | `configuration` | YAML parsing error in configuration file | Inspect YAML syntax with a validator; verify indentation and quotes. |
| `E-CFG-004` | `configuration` | Invalid model variant specified | Set `model.variant` to one of `['n', 's', 'm', 'l', 'x', 'custom']`. |
| `E-CFG-012` | `configuration` | Non-static export batch size | Set `export.batch: 1`. Hailo-8L requires static batch dimension 1. |
| `E-CFG-013` | `configuration` | Dynamic export enabled | Set `export.dynamic: false`. Dynamic spatial axes are unsupported. |
| `E-CFG-015` | `configuration` | Unsupported Hailo target architecture | Set `hailo.target: "hailo8l"`. Raspberry Pi 5 AI HAT+ is Hailo-8L. |

---

### Dataset Errors (`E-DATA-*`)

| Error Code | Phase | Cause | Remediation |
| :--- | :--- | :--- | :--- |
| `E-DATA-001` | `dataset_validation` | Dataset YAML file not found | Verify path in `config.yaml` or provide `--dataset path/to/dataset.yaml`. |
| `E-DATA-004` | `dataset_validation` | Missing class definition (`nc` or `names`) | Ensure dataset YAML defines `nc: <int>` and `names: [...]`. |
| `E-DATA-005` | `dataset_validation` | Missing train or val image paths | Define both `train:` and `val:` split paths in dataset YAML. |
| `E-DATA-006` | `dataset_validation` | Data leakage or corrupt annotations | Check report reasons: eliminate duplicate images across train/val splits; fix bounding box coordinates outside `[0, 1]`. |

---

### Training & Export Errors (`E-TRAIN-*`, `E-EXP-*`, `E-ONNX-*`)

| Error Code | Phase | Cause | Remediation |
| :--- | :--- | :--- | :--- |
| `E-TRAIN-002` | `training` | Checkpoint file not found | Ensure base checkpoint exists or run training stage. |
| `E-TRAIN-007` | `training` | Ultralytics package missing | Run `pip install -e .[training]`. |
| `E-EXP-004` | `onnx_export` | ONNX export generation failed | Check PyTorch model compatibility and ensure opset is supported. |
| `E-ONNX-003` | `onnx_validation` | Exported ONNX violates contract | Re-export ensuring fixed `1x3x640x640` input and static output shapes. |

---

### Hailo Compilation & Runtime Errors (`E-HAILO-*`)

| Error Code | Phase | Cause | Remediation |
| :--- | :--- | :--- | :--- |
| `E-HAILO-ENV-001` | `hailo_environment` | Hailo DFC or HailoRT not installed | Install Hailo DFC v3.28+ on compilation host or use container. |
| `E-HAILO-PARSE-004` | `hailo_parse` | Hailo ONNX parser failure | Inspect unsupported ONNX operators; ensure NMS is not embedded in the ONNX graph. |
| `E-HAILO-OPT-005` | `hailo_optimisation` | Quantization failed | Check calibration set preprocessing; ensure `calib_data.npy` matches model input dimensions. |
| `E-HAILO-COMP-005` | `hailo_compile` | Compilation / layer routing failed | Verify model capacity fits Hailo-8L resource limits; reduce model variant to `n` or `s`. |
| `E-HAILO-RT-002` | `hailo_runtime` | Physical Hailo accelerator missing | Verify Raspberry Pi 5 PCIe HAT+ ribbon cable connection; check `dmesg \| grep hailo`. |

---

### Accuracy Gate Errors (`E-ACC-*`)

| Error Code | Phase | Cause | Remediation |
| :--- | :--- | :--- | :--- |
| `E-ACC-005` | `accuracy_validation` | mAP degradation exceeded allowed gate threshold | Increase calibration dataset image count; verify calibration set is representative of validation distribution; or increase `map50_max_drop`. |
