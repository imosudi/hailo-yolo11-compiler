# Hailo-8L Compilation & Hardware Deployment Manual

This manual provides an in-depth reference for compiling neural networks with the **Hailo Dataflow Compiler (DFC)** and executing hardware inference with **HailoRT** on the **Raspberry Pi 5 with Hailo-8L AI HAT+**.

---

## Table of Contents

1. [Hailo-8L Accelerator Architecture](#1-hailo-8l-accelerator-architecture)
2. [Hailo Compilation Chain (DFC Workflow)](#2-hailo-compilation-chain-dfc-workflow)
3. [Model Script (`.alls`) Directives & Syntax](#3-model-script-alls-directives--syntax)
4. [Post-Processing Policy (`nms: auto`)](#4-post-processing-policy-nms-auto)
5. [HailoRT Host Runtime & Stream Architecture](#5-hailort-host-runtime--stream-architecture)
6. [Raspberry Pi 5 Hardware Setup & Driver Configuration](#6-raspberry-pi-5-hardware-setup--driver-configuration)
7. [Physical Smoke Testing & Edge Health Diagnostics](#7-physical-smoke-testing--edge-health-diagnostics)

---

## 1. Hailo-8L Accelerator Architecture

The **Hailo-8L AI HAT+** is a compact neural processing unit tailored for the Raspberry Pi 5:

- **Compute Capability**: 13 TOPS (Tera Operations Per Second) of INT8 tensor processing.
- **Form Factor**: M.2 Key-M (2230/2242) mounted on the Raspberry Pi AI HAT+.
- **Host Interconnect**: Single-lane PCIe Gen 2/3 ($5.0\text{ GT/s}$ to $8.0\text{ GT/s}$) via a 16-pin flexible flat cable (FFC).
- **Power Envelope**: Typical operating consumption between $1.5\text{W}$ and $2.5\text{W}$.
- **Target Identifier**: Strictly `hailo8l`. (Do not substitute `hailo8` or `hailo10h` without adjusting resource allocation).

Unlike traditional SIMD GPUs or vector DSPs, Hailo NPUs utilize a **spatial dataflow architecture**. Neural compute layers are mapped directly to physical clusters of compute units and on-chip SRAM memory, enabling continuous pipelined execution with minimal off-chip DRAM accesses.

---

## 2. Hailo Compilation Chain (DFC Workflow)

Translating a standard static FP32 ONNX graph into a hardware bitstream involves three discrete compiler phases:

```text
model.onnx (FP32, 1×3×640×640)
       │
       ▼
[Phase F: hailo_parse]
  ├── Input Stream: images (1×3×640×640, float32)
  └── Model Script: yolo11.alls
       │
       ▼
model.har (Floating-point intermediate representation)
       │
       ▼
[Phase G: hailo_optimisation]
  └── Calibration Data: calib_data.npy (200 RGB letterboxed samples)
       │
       ▼
model_quantized.har (Quantised INT8 representation)
       │
       ▼
[Phase H: hailo_compile]
  ├── Layer Placement & Cluster Allocation
  ├── Inter-cluster Dataflow Routing
  ├── Cycle-accurate Execution Scheduling
  └── Target Architecture: hailo8l
       │
       ▼
model.hef (Hailo Executable Format binary)
```

---

## 3. Model Script (`.alls`) Directives & Syntax

During Phase F (`hailo_parse`), the pipeline synthesises a Hailo model script (`yolo11.alls`). This script configures layer-specific hardware transformations:

```text
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
performance_param(compiler_optimization_level=0)
```

### Directives Explained

1. `normalization([mean], [std])`:
   Declares hardware input normalisation. The Hailo-8L input DMA engine accepts raw `uint8` image bytes $[0, 255]$ over PCIe and computes $(x - \text{mean}) / \text{std}$ in dedicated hardware circuitry, completely relieving the Raspberry Pi CPU of image normalisation duties.
2. `performance_param(compiler_optimization_level=<0-3>)`:
   Controls compiler heuristic effort during layer routing and scheduling:
   - `0`: Balanced compile time and throughput (recommended default).
   - `1-3`: Higher compiler search depth for maximum FPS (requires longer compile times).

---

## 4. Post-Processing Policy (`nms: auto`)

For Hailo-8L deployments, raw bounding box and classification heads are preserved in the ONNX graph:

- **Clean Graph Parsing**: Hailo DFC parses the convolution layers and prediction heads without encountering unsupported ONNX control-flow operators.
- **Efficient Post-Processing**: Non-maximum suppression (NMS) and box decoding are executed by the HailoRT host runtime or dedicated post-processing layers on the Raspberry Pi 5 CPU, which can decode boxes in parallel with NPU frame inference.

---

## 5. HailoRT Host Runtime & Stream Architecture

HailoRT provides the C++ and Python userland runtime library for loading and executing HEF binaries on physical hardware:

```text
┌────────────────────────────────────────────────────────┐
│                   HailoRT Application                  │
│                                                        │
│  1. VDevice::create()        ──► Discovers /dev/hailo0 │
│  2. create_infer_model(hef)  ──► Loads HEF bitstream   │
│  3. configure()              ──► Sets up DMA streams   │
│  4. infer()                  ──► Zero-copy DMA I/O     │
└────────────────────────────────────────────────────────┘
```

HailoRT manages virtual streams (`VStreams`), handling quantisation/de-quantisation and buffer packing automatically across the PCIe boundary.

---

## 6. Raspberry Pi 5 Hardware Setup & Driver Configuration

To prepare the physical edge host:

1. **Enable PCIe Gen 3 in Kernel Config**:
   Edit `/boot/firmware/config.txt`:
   ```ini
   dtparam=pciex1
   dtparam=pciex1_gen=3
   ```
2. **Install Official Hailo Driver Package**:
   ```bash
   sudo apt update
   sudo apt install -y hailo-all
   ```
3. **Verify Device Recognition**:
   ```bash
   hailortcli scan
   ```
   *Expected Output:*
   ```text
   Running Scan:
   |- Device: 0000:01:00.0 (Hailo-8L)
   ```
4. **Configure Non-Root Permissions**:
   ```bash
   sudo usermod -aG hailo $USER
   sudo usermod -aG plugdev $USER
   ```

---

## 7. Physical Smoke Testing & Edge Health Diagnostics

Execute Phase J to verify end-to-end hardware execution:

```bash
python train_and_compile.py validate --config config/config.yaml --require-hardware
```

### Graceful Degradation Principle
If this command is executed on a host lacking physical Hailo hardware, the pipeline enforces strict research integrity:
- It **never** silently emulates Hailo execution on CPU or GPU.
- It **never** fabricates zero latency or false throughput numbers.
- It explicitly records `HARDWARE_VALIDATION = NOT_EXECUTED` in `run_manifest.json`.
