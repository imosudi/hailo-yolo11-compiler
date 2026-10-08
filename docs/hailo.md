# Hailo-8L Compilation & Hardware Deployment

## Hailo-8L Overview

The **Hailo-8L AI HAT+** is a compact neural processing accelerator designed specifically for the **Raspberry Pi 5**, delivering up to 13 TOPS (Tera Operations Per Second) of INT8 compute capability over a single-lane PCIe Gen 2/3 interface.

Target architecture: `hailo8l`. Do not substitute Hailo-8 (26 TOPS), Hailo-10H, or other accelerators without adjusting resource constraints.

## Compilation Workflow

The compilation chain translates standard FP32 neural network graphs into hardware-scheduled dataflow bitstreams (HEF):

```text
model.onnx (FP32, 1x3x640x640)
       │
       ▼
Hailo Parser (ONNX → HAR)
  ├── Input Stream: images (1x3x640x640, float32)
  └── Model Script: yolo11.alls
       │
       ▼
model.har (Floating point Hailo representation)
       │
       ▼
Hailo Optimizer (INT8 Post-Training Quantization)
  └── Consumes: calib_data.npy (200 RGB letterboxed samples)
       │
       ▼
model_quantized.har (Quantized INT8 representation)
       │
       ▼
Hailo Compiler (Resource allocation & layer routing)
  └── Target Architecture: hailo8l
       │
       ▼
model.hef (Hailo Executable Format)
```

## Model Script (`.alls`) Configuration

YOLO11 models trained with Ultralytics expect normalized inputs in `[0.0, 1.0]`. When using raw `uint8` image inputs on edge devices, Hailo hardware can perform input normalization with zero latency:

```text
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
performance_param(compiler_optimization_level=0)
```

## Post-processing Policy (`nms: auto`)

For Hailo-8L deployments, raw bounding box and classification heads are preferred over embedding standard PyTorch non-max suppression ops in the ONNX graph:
- Hailo DFC parses the convolution layers and prediction heads cleanly.
- Post-processing is delegated to the HailoRT host runtime or Hailo NMS layer, avoiding unoptimized ONNX control flow ops.

## Raspberry Pi 5 Physical Runtime Setup

To prepare the Raspberry Pi 5 with the AI HAT+:

1. **Kernel Configuration**: Ensure PCIe Gen 3 is enabled in `/boot/firmware/config.txt`:
   ```ini
   dtparam=pciex1
   dtparam=pciex1_gen=3
   ```
2. **HailoRT Driver Installation**:
   ```bash
   sudo apt update
   sudo apt install -y hailo-all
   ```
3. **Verify Device Recognition**:
   ```bash
   hailortcli scan
   # Expected output: Device: Hailo-8L [PCIe 0000:01:00.0]
   ```
4. **Permissions**:
   Ensure the executing user is part of the `hailo` or `plugdev` user group:
   ```bash
   sudo usermod -aG hailo $USER
   ```
