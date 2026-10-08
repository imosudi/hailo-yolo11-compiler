# Dataset Preparation, Formatting & Validation Manual

This manual provides an exhaustive guide to preparing, structuring, and verifying computer vision datasets for the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [YOLO Dataset Format Standard](#1-yolo-dataset-format-standard)
2. [Dataset YAML Descriptor Schema](#2-dataset-yaml-descriptor-schema)
3. [Bounding Box Annotation Specification](#3-bounding-box-annotation-specification)
4. [Background & Negative Samples](#4-background--negative-samples)
5. [Pre-flight Validation Engine (Phase A)](#5-pre-flight-validation-engine-phase-a)
6. [The 18 Automated Dataset Integrity Checks](#6-the-18-automated-dataset-integrity-checks)
7. [Diagnosing & Remedying Dataset Validation Errors](#7-diagnosing--remedying-dataset-validation-errors)
8. [Deterministic Calibration Set Selection](#8-deterministic-calibration-set-selection)

---

## 1. YOLO Dataset Format Standard

The compiler requires datasets organised strictly according to the Ultralytics YOLO object detection directory format.

### Standard Directory Tree

```text
dataset_root/
├── dataset.yaml               # Master dataset descriptor
├── images/
│   ├── train/                 # Training images
│   │   ├── frame_00001.jpg
│   │   ├── frame_00002.jpg
│   │   └── ...
│   ├── val/                   # Validation images (used for accuracy evaluation)
│   │   ├── frame_00501.jpg
│   │   └── ...
│   └── test/                  # Optional testing images
│       └── frame_00901.jpg
└── labels/
    ├── train/                 # Training annotations (.txt)
    │   ├── frame_00001.txt
    │   ├── frame_00002.txt
    │   └── ...
    ├── val/                   # Validation annotations (.txt)
    │   ├── frame_00501.txt
    │   └── ...
    └── test/                  # Optional testing annotations
        └── frame_00901.txt
```

> [!IMPORTANT]
> The directory names `images` and `labels` must be parallel directories at the same depth, and subdirectories (`train`, `val`, `test`) must share identical names.

---

## 2. Dataset YAML Descriptor Schema

The `dataset.yaml` file declares dataset roots, split locations, class count, and class names.

### Example `dataset.yaml`

```yaml
# Root directory. Can be an absolute path or relative to dataset.yaml
path: /workspace/data/industrial_inspection

# Split paths relative to 'path'
train: images/train
val: images/val
test: images/test              # Optional

# Number of classes
nc: 3

# Class names mapping
names:
  0: defect_scratch
  1: defect_dent
  2: defect_burr
```

### Alternatively (List Format for `names`)
```yaml
names:
  - defect_scratch
  - defect_dent
  - defect_burr
```

---

## 3. Bounding Box Annotation Specification

Each image has a corresponding `.txt` file with identical basename in the parallel `labels/` directory (e.g. `images/train/sample_01.jpg` $\leftrightarrow$ `labels/train/sample_01.txt`).

### Format Syntax

Each line contains space-separated values defining one bounding box:

```text
<class_id> <x_centre> <y_centre> <width> <height>
```

| Parameter | Type | Valid Range | Description |
| :--- | :--- | :--- | :--- |
| `<class_id>` | Integer | $[0, nc - 1]$ | Zero-indexed integer identifier matching `names` in `dataset.yaml`. |
| `<x_centre>` | Float | $[0.0, 1.0]$ | Normalised horizontal centre of the bounding box ($X / \text{ImageWidth}$). |
| `<y_centre>` | Float | $[0.0, 1.0]$ | Normalised vertical centre of the bounding box ($Y / \text{ImageHeight}$). |
| `<width>` | Float | $(0.0, 1.0]$ | Normalised width of the bounding box ($\text{Width} / \text{ImageWidth}$). |
| `<height>` | Float | $(0.0, 1.0]$ | Normalised height of the bounding box ($\text{Height} / \text{ImageHeight}$). |

### Concrete Annotation Example

For an image with dimensions $1920 \times 1080$:
- Bounding box 1: Class `0`, centre at $(960, 540)$, width $480$, height $270$.
- Bounding box 2: Class `1`, centre at $(200, 300)$, width $100$, height $150$.

The resulting `.txt` file contains:
```text
0 0.500000 0.500000 0.250000 0.250000
1 0.104167 0.277778 0.052083 0.138889
```

---

## 4. Background & Negative Samples

In real-world edge vision systems, preventing false positives is critical. Datasets should include background images containing no objects of interest.

### Negative Sample Representation
To declare an image as a negative background sample:
1. Place the image in `images/train/` or `images/val/`.
2. Create an **empty file** (0 bytes) with the identical name and `.txt` extension in `labels/train/` or `labels/val/`.

> [!NOTE]
> Do not omit the `.txt` file for background images. A missing `.txt` file triggers a `Missing labels` warning or failure in pre-flight validation.

---

## 5. Pre-flight Validation Engine (Phase A)

Before starting GPU-intensive training or Hailo quantisation, the pipeline runs Phase A (`dataset_validation`). This catches data integrity bugs early before compute resources are wasted.

Run validation independently:
```bash
python train_and_compile.py dataset --dataset data/dataset.yaml
```

### Structured Diagnostic Report Example

```text
================================================================================
 DATASET PRE-FLIGHT VALIDATION REPORT
================================================================================
Dataset YAML:        /workspace/data/industrial_inspection/dataset.yaml
Class Count (nc):    3 classes
Class Names:         ['defect_scratch', 'defect_dent', 'defect_burr']

Images Analyzed:     18,421
Labels Analyzed:     18,421
Missing Labels:      0
Invalid Annotations: 0
Duplicate Images:    0
Train/Val Leakage:   0
Corrupted Images:    0

Calibration Set:     9,812 clean candidates available (Minimum required: 200)

STATUS:              SUCCESS
================================================================================
```

---

## 6. The 18 Automated Dataset Integrity Checks

The validation engine performs 18 automated checks:

1. **YAML File Existence**: Verifies `dataset.yaml` path exists and is readable.
2. **YAML Syntax**: Parses YAML content and ensures well-formed structure.
3. **Split Directory Existence**: Verifies that paths specified for `train` and `val` exist on disk.
4. **Image Directory Readability**: Verifies image directories have read permissions.
5. **Label Directory Readability**: Verifies label directories have read permissions.
6. **Class Count (`nc`)**: Ensures `nc` is defined as a positive integer ($\ge 1$).
7. **Class Names Consistency**: Checks that `names` contains exactly `nc` entries and matches indices.
8. **Class-ID Range**: Asserts that every annotation class ID satisfies $0 \le \text{class\_id} < nc$.
9. **Malformed Annotation Lines**: Detects lines with incorrect token counts ($\ne 5$) or non-numeric tokens.
10. **Coordinate Normalisation**: Ensures centre coordinates, widths, and heights fall strictly within $[0.0, 1.0]$.
11. **Non-Degenerate Bounding Boxes**: Asserts that widths and heights are strictly positive ($> 0.0$).
12. **Missing Labels**: Detects images that lack a corresponding label `.txt` file.
13. **Orphaned Labels**: Detects label `.txt` files that lack a corresponding image.
14. **Corrupt Image Detection**: Attempts opening image files with Pillow, identifying corrupt headers or truncated byte streams.
15. **Supported Image Formats**: Verifies extensions belong to supported formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
16. **Duplicate Image Hashes**: Computes SHA-256 image hashes to detect redundant identical files.
17. **Train/Validation Leakage**: Asserts that no image in the `val` split is identical in content to any image in `train`.
18. **Calibration Eligibility**: Asserts that the training split contains at least 200 clean, uncorrupted images for Hailo INT8 calibration.

---

## 7. Diagnosing & Remedying Dataset Validation Errors

### Error `E-DATA-004`: Missing Class Definition
- **Cause**: `dataset.yaml` lacks `nc:` or `names:`.
- **Fix**: Open `dataset.yaml` and declare `nc: <int>` and the matching list or dictionary of `names`.

### Error `E-DATA-006`: Out-of-Bounds Coordinates
- **Diagnostic Snippet**:
  ```text
  Invalid label in labels/train/frame_042.txt: coordinate 1.05 exceeds [0.0, 1.0] range
  ```
- **Cause**: Annotations created by tools that exported absolute pixel coordinates rather than normalised coordinates, or bounding boxes extending outside the frame.
- **Fix**: Normalise coordinates by dividing by image width and height, and clamp values to $[0.0, 1.0]$.

### Error `E-DATA-007`: Train/Validation Leakage
- **Diagnostic Snippet**:
  ```text
  Data leakage detected: 4 images in validation split share SHA-256 hashes with training split
  ```
- **Cause**: Identical images exist in both `images/train/` and `images/val/`. Evaluating models on training samples yields artificially inflated, invalid validation metrics.
- **Fix**: Remove duplicate image files from `images/val/`.

### Error `E-DATA-008`: Insufficient Calibration Candidates
- **Diagnostic Snippet**:
  ```text
  Calibration candidates: 84 (Minimum required: 200)
  ```
- **Cause**: Training split has fewer than 200 valid images.
- **Fix**: Add more training images, or set `dataset.calibration.count: 80` in `config/config.yaml` (Hailo requires a minimum of 100 for reliable quantisation, but permits explicit overrides down to 50 for testing).

---

## 8. Deterministic Calibration Set Selection

Hailo Post-Training Quantisation requires a representative subset of images to calculate activation histograms and scale factors.

### Sampling Contract
- **Default Count**: Exactly 200 images.
- **Selection Source**: Deterministically sampled from `images/train/` using pseudo-random seed `42`.
- **Corruption Filter**: Corrupt or unreadable images are rejected during selection.
- **Auditability**: Selected sample file paths, SHA-256 hashes, and dimensions are recorded in:
  `artifacts/runs/<run_id>/calibration/calibration_manifest.json`

### Using a Dedicated Calibration Directory
If you have a dedicated set of domain-specific calibration images:
In `config/config.yaml`:
```yaml
dataset:
  calibration:
    source: "custom"
    custom_dir: "/workspace/data/calibration_images"
    count: 200
    seed: 42
```
The sampler will validate and letterbox images directly from `custom_dir`.
