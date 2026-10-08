"""Phase A: Comprehensive dataset pre-flight validation engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import yaml
from PIL import Image

from yolo_hailo_mlops.dataset.statistics import DatasetValidationReport, SplitStatistics
from yolo_hailo_mlops.exceptions import DatasetValidationError
from yolo_hailo_mlops.utils.hashing import compute_sha256

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_label_file(img_path: Path, label_dirs: List[Path]) -> Optional[Path]:
    """Locate matching .txt label file for an image across standard YOLO directories."""
    stem = img_path.stem
    # 1. Look in explicit label directory
    for l_dir in label_dirs:
        cand = l_dir / f"{stem}.txt"
        if cand.is_file():
            return cand

    # 2. Look in directory named 'labels' parallel to 'images'
    parent_parts = list(img_path.parent.parts)
    if "images" in parent_parts:
        idx = len(parent_parts) - 1 - parent_parts[::-1].index("images")
        parent_parts[idx] = "labels"
        cand = Path(*parent_parts) / f"{stem}.txt"
        if cand.is_file():
            return cand

    # 3. Look alongside image
    cand = img_path.parent / f"{stem}.txt"
    if cand.is_file():
        return cand

    return None


def validate_yolo_label_file(
    label_path: Path,
    num_classes: int,
) -> Tuple[bool, int, List[int], Optional[str]]:
    """Validate syntax and bounds of a YOLO format bounding-box annotation file."""
    box_count = 0
    class_ids = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    return False, box_count, class_ids, f"Line {line_no} has fewer than 5 tokens"

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    return False, box_count, class_ids, f"Line {line_no} has non-numeric coordinates"

                if not (0 <= cls_id < num_classes):
                    return (
                        False,
                        box_count,
                        class_ids,
                        f"Line {line_no}: Class ID {cls_id} out of range [0, {num_classes - 1}]",
                    )

                # Tolerance for coordinates slightly outside [0, 1] due to floating point rounding
                tol = 0.05
                if not (-tol <= xc <= 1.0 + tol and -tol <= yc <= 1.0 + tol):
                    return False, box_count, class_ids, f"Line {line_no}: Centre out of bounds"
                if w <= 0 or h <= 0 or w > (1.0 + tol) or h > (1.0 + tol):
                    return False, box_count, class_ids, f"Line {line_no}: Invalid width/height"

                box_count += 1
                class_ids.append(cls_id)
        return True, box_count, class_ids, None
    except Exception as e:
        return False, box_count, class_ids, str(e)


def inspect_split(
    split_name: str,
    split_path: Path,
    num_classes: int,
    base_dir: Path,
) -> Tuple[SplitStatistics, Dict[str, Path]]:
    """Inspect images and labels within a split directory or text file."""
    stats = SplitStatistics(name=split_name)
    image_hashes: Dict[str, Path] = {}

    image_paths: List[Path] = []
    resolved_split = split_path if split_path.is_absolute() else (base_dir / split_path).resolve()

    if resolved_split.is_file():
        # Text file with image paths
        with open(resolved_split, "r", encoding="utf-8") as f:
            for line in f:
                p_str = line.strip()
                if p_str:
                    p = Path(p_str)
                    image_paths.append(p if p.is_absolute() else base_dir / p)
    elif resolved_split.is_dir():
        # Directory containing images
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(resolved_split.rglob(f"*{ext}"))
            image_paths.extend(resolved_split.rglob(f"*{ext.upper()}"))
    else:
        return stats, image_hashes

    label_dirs = []
    # Identify label directory
    labels_candidate = resolved_split.parent / "labels" / resolved_split.name
    if labels_candidate.is_dir():
        label_dirs.append(labels_candidate)

    for img_path in image_paths:
        if not img_path.is_file():
            continue

        stats.num_images += 1

        # Check image corruption & dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                stats.widths.append(w)
                stats.heights.append(h)
        except Exception:
            stats.corrupt_images += 1
            continue

        # Check SHA-256 hash for deduplication
        try:
            img_hash = compute_sha256(img_path)
            image_hashes[img_hash] = img_path
        except Exception:
            pass

        # Check corresponding label
        lbl_path = find_label_file(img_path, label_dirs)
        if lbl_path is None:
            stats.missing_labels += 1
        else:
            stats.num_labels += 1
            valid, b_cnt, cls_ids, err = validate_yolo_label_file(lbl_path, num_classes)
            if not valid:
                stats.missing_labels += 1
            else:
                stats.num_boxes += b_cnt
                if b_cnt == 0:
                    stats.empty_images += 1
                for cid in cls_ids:
                    stats.class_counts[cid] = stats.class_counts.get(cid, 0) + 1

    return stats, image_hashes


class DatasetValidator:
    """Pre-flight dataset validation engine."""

    def __init__(self, dataset_yaml: Union[str, Path]) -> None:
        self.yaml_path = Path(dataset_yaml).resolve()

    def validate(self, fail_fast: bool = True) -> DatasetValidationReport:
        """Execute full dataset pre-flight validation."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.yaml_path.is_file():
            raise DatasetValidationError(
                f"Dataset YAML file does not exist: {self.yaml_path}",
                code="E-DATA-001",
                artifact=str(self.yaml_path),
            )

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise DatasetValidationError(
                f"Failed to parse dataset YAML '{self.yaml_path}': {e}",
                code="E-DATA-002",
                artifact=str(self.yaml_path),
            ) from e

        if not isinstance(data, dict):
            raise DatasetValidationError(
                "Dataset YAML root must be a mapping",
                code="E-DATA-003",
                artifact=str(self.yaml_path),
            )

        # Check classes
        num_classes = data.get("nc")
        class_names = data.get("names", [])

        if num_classes is None:
            if isinstance(class_names, list) and class_names:
                num_classes = len(class_names)
            elif isinstance(class_names, dict) and class_names:
                num_classes = len(class_names)
            else:
                raise DatasetValidationError(
                    "Dataset YAML must define 'nc' (number of classes)",
                    code="E-DATA-004",
                    artifact=str(self.yaml_path),
                )

        if isinstance(class_names, dict):
            names_list = [class_names.get(i, f"class_{i}") for i in range(num_classes)]
        elif isinstance(class_names, list):
            names_list = class_names
        else:
            names_list = [f"class_{i}" for i in range(num_classes)]

        base_dir = self.yaml_path.parent
        root_dir_raw = data.get("path")
        if root_dir_raw:
            base_dir = (base_dir / root_dir_raw).resolve()

        train_path_raw = data.get("train")
        val_path_raw = data.get("val")
        test_path_raw = data.get("test")

        if not train_path_raw:
            errors.append("Dataset YAML missing mandatory 'train' path")
        if not val_path_raw:
            errors.append("Dataset YAML missing mandatory 'val' path")

        if errors:
            raise DatasetValidationError(
                "; ".join(errors),
                code="E-DATA-005",
                artifact=str(self.yaml_path),
            )

        # Inspect train split
        train_path = Path(str(train_path_raw))
        train_stats, train_hashes = inspect_split("train", train_path, num_classes, base_dir)

        # Inspect val split
        val_path = Path(str(val_path_raw))
        val_stats, val_hashes = inspect_split("val", val_path, num_classes, base_dir)

        # Inspect optional test split
        test_stats = None
        if test_path_raw:
            test_path = Path(str(test_path_raw))
            test_stats, _ = inspect_split("test", test_path, num_classes, base_dir)

        # Check data leakage between train and val
        leakage_count = len(set(train_hashes.keys()) & set(val_hashes.keys()))
        if leakage_count > 0:
            errors.append(f"Detected {leakage_count} duplicate image(s) between train and val splits (leakage)")

        # Check duplicates within splits
        duplicate_images = (train_stats.num_images - len(train_hashes)) + (
            val_stats.num_images - len(val_hashes)
        )
        if duplicate_images > 0:
            warnings.append(f"Detected {duplicate_images} duplicate image file(s)")

        # Verify dataset non-empty
        if train_stats.num_images == 0:
            errors.append("Training dataset contains 0 readable images")
        if val_stats.num_images == 0:
            errors.append("Validation dataset contains 0 readable images")

        # Missing labels
        if train_stats.missing_labels > 0:
            warnings.append(f"{train_stats.missing_labels} training image(s) have missing or corrupt labels")

        calibration_candidates = train_stats.num_images - train_stats.corrupt_images

        is_valid = len(errors) == 0
        status = "PASSED" if is_valid else "FAILED"

        report = DatasetValidationReport(
            is_valid=is_valid,
            status=status,
            train_stats=train_stats,
            val_stats=val_stats,
            test_stats=test_stats,
            duplicate_images=duplicate_images,
            train_val_leakage=leakage_count,
            calibration_candidates=calibration_candidates,
            num_classes=num_classes,
            class_names=names_list,
            errors=errors,
            warnings=warnings,
        )

        if fail_fast and not is_valid:
            raise DatasetValidationError(
                message=f"Dataset pre-flight validation failed: {'; '.join(errors)}",
                code="E-DATA-006",
                artifact=str(self.yaml_path),
            )

        return report
