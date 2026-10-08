# Calibration & INT8 Post-Training Quantisation Manual

This manual provides an in-depth mathematical and operational reference for generating calibration datasets and executing INT8 Post-Training Quantisation (PTQ) within the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [Mathematical Foundations of INT8 PTQ](#1-mathematical-foundations-of-int8-ptq)
2. [Why Calibration Sets are Essential](#2-why-calibration-sets-are-essential)
3. [The Hailo Input Preprocessing Contract](#3-the-hailo-input-preprocessing-contract)
4. [Uniform Letterbox Scaling & Aspect Ratio Policy](#4-uniform-letterbox-scaling--aspect-ratio-policy)
5. [Sampling Strategy & Deterministic Selection](#5-sampling-strategy--deterministic-selection)
6. [Hardware Input Normalisation in `.alls` Scripts](#6-hardware-input-normalisation-in-alls-scripts)
7. [Calibration Output Artefacts & Audit Ledgers](#7-calibration-output-artefacts--audit-ledgers)

---

## 1. Mathematical Foundations of INT8 PTQ

Deep neural networks are typically trained using 32-bit floating-point numbers (`FP32`), where each weight and activation value occupies 4 bytes. The **Hailo-8L AI HAT+** is an 8-bit integer (`INT8`) dataflow processor, achieving up to 13 TOPS of computational throughput with minimal power consumption ($< 2.5\text{W}$).

Post-Training Quantisation maps continuous floating-point values $x \in [x_{\min}, x_{\max}]$ to discrete 8-bit integer representations $q \in [-128, 127]$:

$$q = \text{clamp}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, -128, 127\right)$$

where:
- **$S$ (Scale Factor)**: Positive real number representing the step size between quantisation bins:
  $$S = \frac{x_{\max} - x_{\min}}{255}$$
- **$Z$ (Zero-Point)**: Integer offset corresponding to real zero, ensuring that floating-point zero maps exactly to an integer without rounding bias.
- **$\lfloor \cdot \rceil$**: Rounding to the nearest integer.

De-quantisation reconstructs the real approximation $\hat{x}$:
$$\hat{x} = S \cdot (q - Z)$$

---

## 2. Why Calibration Sets are Essential

While neural network weights are static and known prior to compilation, **activation tensors** vary dynamically based on the input images being processed.

To determine optimal dynamic ranges $[x_{\min}, x_{\max}]$ and clipping thresholds for intermediate activation layers without retraining the entire network, the **Hailo Dataflow Compiler (DFC)** feeds a representative set of unlabelled images through the floating-point Hailo Archive (`model.har`).

During this calibration pass:
1. Intermediate activation histograms are accumulated across all layers.
2. Outlier activations are clipped using Kullback-Leibler (KL) divergence or mean squared error (MSE) minimisation.
3. Optimal scale factors $S$ and zero-points $Z$ are calculated for each layer.

---

## 3. The Hailo Input Preprocessing Contract

Quantisation accuracy is exceptionally sensitive to preprocessing discrepancies. Any divergence between training preprocessing and calibration preprocessing degrades model detection capability:

| Parameter | Mandatory Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Colour Space** | `RGB` | YOLO11 models are trained on RGB. BGR inputs distort colour feature maps. |
| **Spatial Dimensions** | $640 \times 640$ | Matches static spatial contract of exported ONNX graph. |
| **Aspect Ratio Policy** | `Letterbox` | Preserves geometric aspect ratio; pads with neutral grey ($114$). |
| **Sample Count** | `200` | Recommended range: 100 to 200 images for statistically robust histograms. |
| **Data Type** | `float32` | Normalised to $[0.0, 1.0]$ in calibration NumPy array. |
| **Tensor Layout** | `NHWC` / `NCHW` | Converted to $(N, 640, 640, 3)$ array for Hailo DFC feeding. |

---

## 4. Uniform Letterbox Scaling & Aspect Ratio Policy

Direct stretching of non-square images distorts aspect ratios, stretching bounding boxes and degrading feature extraction. The pipeline enforces a strict **letterbox transformation**:

```text
Input Image (1920 × 1080)
           │
           ▼ Uniform scale factor: min(640/1920, 640/1080) = 0.3333
Scaled Image (640 × 360)
           │
           ▼ Pad top/bottom with 140 pixels (fill value: 114)
Letterboxed Image (640 × 640)
```

```python
import numpy as np
from PIL import Image

def letterbox_image(image: Image.Image, target_size: int = 640, pad_value: int = 114) -> Image.Image:
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), (pad_value, pad_value, pad_value))
    
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))
    return padded
```

---

## 5. Sampling Strategy & Deterministic Selection

Phase D (`calibration`) selects images according to configured policies:

### 1. Training Set Sampling (`source: "train"`)
- Inspects the training split defined in `dataset.yaml`.
- Initialises a deterministic pseudo-random sampler using `project.seed` (default `42`).
- Rejects corrupt or invalid image files.
- Extracts exactly `dataset.calibration.count` images (default 200).

### 2. Custom Domain Sampling (`source: "custom"`)
- Reads images from an isolated operational directory declared by `custom_dir`.
- Enables calibration against camera feeds from the physical edge deployment environment.

---

## 6. Hardware Input Normalisation in `.alls` Scripts

Standard vision models trained with Ultralytics expect pixel intensities normalised to $[0.0, 1.0]$. 

On edge microprocessors, converting incoming `uint8` camera frames to `float32` and dividing by `255.0` consumes significant CPU cycles. Hailo hardware features **zero-latency hardware normalisation**:

During Phase F (`hailo_parse`), the pipeline generates `artifacts/runs/<run_id>/hailo/yolo11.alls`:

```text
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
performance_param(compiler_optimization_level=0)
```

This instructs the Hailo-8L input DMA engine to accept raw `uint8` image bytes $[0, 255]$ directly over the PCIe bus and execute normalisation in dedicated hardware circuitry with zero CPU overhead.

---

## 7. Calibration Output Artefacts & Audit Ledgers

The calibration phase writes complete audit artefacts to `artifacts/runs/<run_id>/calibration/`:

```text
artifacts/runs/<run_id>/calibration/
├── calib_data.npy                 # Processed NumPy tensor (200, 640, 640, 3)
├── images/                        # The 200 sampled raw images
├── calibration_manifest.json      # Cryptographic provenance ledger
└── calibration_statistics.json    # Channel mean, std, min, max distribution
```

### `calibration_statistics.json` Example
```json
{
  "sample_count": 200,
  "dimensions": [640, 640, 3],
  "mean_intensity": [118.42, 115.19, 111.08],
  "std_intensity": [58.12, 57.44, 59.21],
  "min_intensity": [0.0, 0.0, 0.0],
  "max_intensity": [255.0, 255.0, 255.0]
}
```
