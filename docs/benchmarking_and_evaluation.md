# Accuracy Gates, Telemetry & Performance Benchmarking Manual

This manual provides an in-depth reference for quantitative accuracy validation, degradation gates, latency percentiles, and hardware telemetry tracking in the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [3-Way Comparative Accuracy Evaluation](#1-3-way-comparative-accuracy-evaluation)
2. [Computed Vision Metrics](#2-computed-vision-metrics)
3. [Configurable Accuracy Degradation Gates](#3-configurable-accuracy-degradation-gates)
4. [Edge Performance & Latency Benchmarking](#4-edge-performance--latency-benchmarking)
5. [Hardware Resource Telemetry](#5-hardware-resource-telemetry)
6. [Telemetry Provenance Taxonomy](#6-telemetry-provenance-taxonomy)
7. [Auditability in `run_manifest.json` and `run_report.md`](#7-auditability-in-run_manifestjson-and-run_reportmd)

---

## 1. 3-Way Comparative Accuracy Evaluation

Quantising deep learning models from 32-bit floating-point (`FP32`) to 8-bit integer (`INT8`) introduces rounding and truncation. To ensure that quantisation noise does not degrade operational performance, the pipeline executes a **3-way comparative evaluation** across the validation split:

```text
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│   FP32 PyTorch Base    │      │    FP32 ONNX Model     │      │   INT8 Hailo-8L HEF    │
│  - Reference baseline  │ ──►  │  - Validates export    │ ──►  │  - Validates PTQ       │
│  - Ground-truth target │      │    numerical fidelity  │      │    accuracy retention  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
            │                               │                               │
            └───────────────────────────────┴───────────────────────────────┘
                                            ▼
                           Comparative Delta Analysis (ΔmAP)
```

All three representations are evaluated against the **identical validation dataset** using the **identical preprocessing contract** (RGB, $640 \times 640$, letterbox pad 114, normalised to $[0.0, 1.0]$).

---

## 2. Computed Vision Metrics

For each evaluation stage, the pipeline computes:

| Metric | Mathematical Description | Purpose |
| :--- | :--- | :--- |
| **mAP50** | Mean Average Precision at $\text{IoU} = 0.50$ | Assesses rough localisation and classification accuracy. |
| **mAP50-95** | Mean Average Precision across 10 IoU thresholds ($0.50$ to $0.95$ in $0.05$ increments) | COCO standard for strict localisation accuracy. |
| **Precision** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ | Measures resistance to false alarms. |
| **Recall** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ | Measures detection coverage of real objects. |
| **Per-Class AP** | Average Precision computed independently for each class | Identifies class-specific quantisation degradation. |
| **Detection Count** | Total number of bounding boxes predicted above confidence threshold | Verifies bounding box density. |

---

## 3. Configurable Accuracy Degradation Gates

The pipeline enforces hard pass/fail quality gates. Rather than merely logging precision degradation, it halts automated deployment if accuracy drops below acceptable limits:

### Mathematical Definitions

$$\Delta\text{mAP50} = \text{mAP50}_{\text{PyTorch}} - \text{mAP50}_{\text{Hailo}}$$
$$\Delta\text{mAP50-95} = \text{mAP50-95}_{\text{PyTorch}} - \text{mAP50-95}_{\text{Hailo}}$$

### Gate Configuration (`config/config.yaml`)

```yaml
validation:
  enabled: true
  accuracy:
    map50_max_drop: 0.02      # Maximum permitted drop in mAP50 (2%)
    map5095_max_drop: 0.02    # Maximum permitted drop in mAP50-95 (2%)
```

### Gate Decision Logic

```text
if ΔmAP50 > map50_max_drop or ΔmAP50-95 > map5095_max_drop:
    Status = FAILED
    ErrorCode = E-ACC-005
    Remediation = "Increase calibration image count or check calibration dataset distribution."
else:
    Status = SUCCESS
```

Machine-readable gate status is captured directly in the run manifest:
```json
{
  "accuracy_gate": {
    "status": "passed",
    "map50_drop": 0.007,
    "map50_threshold": 0.020,
    "map5095_drop": 0.012,
    "map5095_threshold": 0.020
  }
}
```

---

## 4. Edge Performance & Latency Benchmarking

When executed on the Raspberry Pi 5 with Hailo-8L, Phase K measures physical inference latency and throughput.

### Benchmarking Methodology

1. **Warm-up Phase (20 iterations)**:
   Feeds dummy frames through the HailoRT virtual stream to prime DMA memory mappings, hardware queues, and NPU caches. These iterations are excluded from metrics to prevent cold-start distortion.
2. **Measurement Phase (100 iterations)**:
   Measures time per inference pass using monotonic high-resolution timers (`time.perf_counter_ns`).

### Statistical Latency Percentiles

From the measured latency distribution, the pipeline computes:
- **Minimum Latency**: Theoretical hardware best-case limit.
- **Maximum Latency**: Worst-case tail latency.
- **Mean Latency**: Average latency over all runs.
- **Median ($p_{50}$)**: Representative latency unaffected by outliers.
- **95th Percentile ($p_{95}$)**: Strict latency bound for $95\%$ of frames.
- **99th Percentile ($p_{99}$)**: Tail latency bound under peak system contention.
- **Throughput (FPS)**: Derived framerate ($\text{FPS} = 1000 / p_{50}\text{ ms}$).

---

## 5. Hardware Resource Telemetry

Concurrently with inference benchmarks, the pipeline records host system vitals to ensure edge stability:

- **CPU Utilisation**: Percentage CPU load per core and overall.
- **Memory Consumption**: Resident memory (RAM) used by the inference process and host system.
- **Storage I/O**: Read/write operations and available disk space.
- **SoC Temperature**: Broadcom BCM2712 processor core temperature.
- **Thermal Throttling State**: Monitored via Raspberry Pi `vcgencmd get_throttled` to detect frequency caps or thermal throttling.
- **PCIe Link Status**: Validates Gen 3 link negotiation and payload size.

---

## 6. Telemetry Provenance Taxonomy

To ensure research integrity and prevent unsubstantiated marketing claims, every telemetry datum is labelled with an explicit **provenance tag**:

| Provenance Classification | Definition | Concrete Example |
| :--- | :--- | :--- |
| **`MEASURED`** | Directly observed from physical sensors, hardware counters, or OS system calls. | `soc_temperature: 48.2` (via sysfs thermal zone) |
| **`DERIVED`** | Mathematically calculated from one or more `MEASURED` primitives. | `throughput_fps: 82.4` (calculated from $1000 / \text{latency}_{\text{ms}}$) |
| **`CONFIGURED`** | Static user or pipeline parameter supplied via configuration or CLI. | `warmup_iterations: 20`, `imgsz: 640` |
| **`UNAVAILABLE`** | Metric requested by schema, but hardware, driver, or host cannot supply it. | `hailo_power: null` (Hailo-8L M.2 lacks onboard shunt resistor) |

> [!CRITICAL]
> The system **never** substitutes missing telemetry with fabricated zeros. If a sensor does not exist, the entry is explicitly recorded as `UNAVAILABLE` with a human-readable explanation.

---

## 7. Auditability in `run_manifest.json` and `run_report.md`

All evaluation outcomes, gate decisions, percentiles, and hardware vitals are persisted into `run_manifest.json` alongside cryptographic SHA-256 hashes of all input and output files. 

Human-readable summaries are simultaneously formatted in `run_report.md` with markdown tables:

### Example Comparative Report Table

| Representation | Precision | mAP50 | mAP50-95 | Precision | Recall | $\Delta\text{mAP50}$ | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | FP32 | 0.824 | 0.612 | 0.841 | 0.795 | Baseline | Baseline |
| **ONNX** | FP32 | 0.824 | 0.611 | 0.840 | 0.795 | 0.000 | PASS |
| **Hailo-8L** | INT8 | 0.816 | 0.603 | 0.832 | 0.789 | **-0.008** | **PASS ($\le 0.02$)** |

### Example Latency & Telemetry Table

| Metric | Value | Provenance | Notes |
| :--- | :--- | :--- | :--- |
| **$p_{50}$ Latency** | `11.8 ms` | `MEASURED` | Median edge inference time |
| **$p_{95}$ Latency** | `12.4 ms` | `MEASURED` | 95th percentile bound |
| **$p_{99}$ Latency** | `13.1 ms` | `MEASURED` | Peak tail latency |
| **Throughput** | `84.7 FPS` | `DERIVED` | Framerate from $p_{50}$ |
| **SoC Temperature** | `51.3 °C` | `MEASURED` | Raspberry Pi 5 Active Cooler enabled |
| **Throttling State** | `0x0 (None)` | `MEASURED` | No undervoltage or frequency capping |
