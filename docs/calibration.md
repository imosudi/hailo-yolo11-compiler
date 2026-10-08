# Calibration & INT8 Post-Training Quantization

## Objective

Post-Training Quantization (PTQ) maps continuous 32-bit floating-point weights and activation tensors into discrete 8-bit integer values (`INT8`). To compute the optimal dynamic ranges (scale factors and zero-points) for each layer without full model retraining, Hailo DFC requires a representative calibration dataset.

## Calibration Contract

To prevent numerical accuracy degradation between training, ONNX export, and hardware execution, the calibration set strictly enforces the input contract:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Color Space** | `RGB` | 3 channels (converted from BGR or Grayscale) |
| **Dimensions** | `640 × 640` | Square spatial resolution matching YOLO11 input |
| **Aspect Ratio Policy** | `Letterbox` | Maintained using uniform scaling and padding (value 114) |
| **Sample Count** | `200` | Recommended range: 100 – 200 samples |
| **Deterministic Seed** | `42` | Seeded random pseudo-shuffle for exact repeatability |
| **Format** | `calib_data.npy` | Contiguous numpy array of shape `(N, 640, 640, 3)` |

## Calibration Sampling Strategy

The pipeline supports two calibration sources configured in `config/config.yaml`:

```yaml
dataset:
  calibration:
    source: "train"     # "train" or "custom"
    count: 200          # Sample count
    seed: 42            # Random seed
    custom_dir: null    # Path if source is "custom"
```

1. **Training Set Sampling (`source: train`)**:
   - The selector scans the training split defined in `dataset.yaml`.
   - Filters out corrupt images or images without readable headers.
   - Deterministically sorts the discovered paths to eliminate filesystem traversal variability across different operating systems.
   - Samples 200 images using `random.Random(seed)`.

2. **Dedicated Directory (`source: custom`)**:
   - Gathers images from an isolated operational domain folder specified by `custom_dir`.

## Output Artifacts

Calibration produces an isolated package in `artifacts/runs/<run_id>/calibration/`:

```text
artifacts/runs/<run_id>/calibration/
├── images/                         # Processed letterboxed images
│   ├── calib_0000.jpg
│   └── ...
├── calib_data.npy                  # Packed numpy array for Hailo DFC
├── calibration_manifest.json       # Record of source paths, hashes, and contract
└── calibration_statistics.json     # Size and verification checksums
```
