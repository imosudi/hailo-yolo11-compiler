"""Calibration dataset selector and preprocessed calibration package generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
import yaml

from yolo_hailo_mlops.calibration.manifest import CalibrationManifest, CalibrationSampleRecord
from yolo_hailo_mlops.calibration.preprocessor import preprocess_calibration_image
from yolo_hailo_mlops.config import AppConfig
from yolo_hailo_mlops.dataset.sampler import sample_images
from yolo_hailo_mlops.dataset.validator import SUPPORTED_IMAGE_EXTENSIONS
from yolo_hailo_mlops.exceptions import CalibrationError
from yolo_hailo_mlops.utils.filesystem import atomic_write, ensure_dir, safe_symlink_or_copy
from yolo_hailo_mlops.utils.hashing import compute_sha256


class CalibrationSelector:
    """Manages deterministic calibration image selection, preprocessing, and manifest output."""

    def __init__(self, config: AppConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.output_dir = ensure_dir(run_dir / "calibration")
        self.images_dir = ensure_dir(self.output_dir / "images")
        self.canonical_dir = ensure_dir(Path(config.artifacts.root) / "calibration")

    def _discover_source_images(self) -> List[Path]:
        calib_cfg = self.config.dataset.calibration
        if calib_cfg.source == "custom":
            if not calib_cfg.custom_dir or not Path(calib_cfg.custom_dir).is_dir():
                raise CalibrationError(
                    f"Custom calibration directory not found: {calib_cfg.custom_dir}",
                    code="E-CAL-004",
                    artifact=str(calib_cfg.custom_dir),
                )
            src_p = Path(calib_cfg.custom_dir)
            images: List[Path] = []
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.extend(src_p.rglob(f"*{ext}"))
                images.extend(src_p.rglob(f"*{ext.upper()}"))
            return images

        # Sample from train split in dataset YAML
        yaml_path = Path(self.config.dataset.yaml).resolve()
        if not yaml_path.is_file():
            raise CalibrationError(
                f"Dataset YAML missing for calibration: {yaml_path}",
                code="E-CAL-005",
                artifact=str(yaml_path),
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        base_dir = yaml_path.parent
        if data.get("path"):
            base_dir = (base_dir / data["path"]).resolve()

        train_entry = data.get("train")
        if not train_entry:
            raise CalibrationError("Dataset YAML missing 'train' split", code="E-CAL-006")

        train_path = Path(train_entry)
        resolved_train = train_path if train_path.is_absolute() else base_dir / train_path

        images: List[Path] = []
        if resolved_train.is_file():
            with open(resolved_train, "r", encoding="utf-8") as f:
                for line in f:
                    p_str = line.strip()
                    if p_str:
                        p = Path(p_str)
                        images.append(p if p.is_absolute() else base_dir / p)
        elif resolved_train.is_dir():
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.extend(resolved_train.rglob(f"*{ext}"))
                images.extend(resolved_train.rglob(f"*{ext.upper()}"))

        return images

    def generate(self) -> Tuple[Path, Path]:
        """Generate calibration dataset, numpy archive, and manifest."""
        calib_cfg = self.config.dataset.calibration
        count = calib_cfg.count
        seed = calib_cfg.seed

        source_candidates = self._discover_source_images()
        if not source_candidates:
            raise CalibrationError(
                "No candidate images available to build calibration set.",
                code="E-CAL-007",
            )

        selected_paths = sample_images(source_candidates, count=count, seed=seed)

        target_size = (self.config.export.imgsz, self.config.export.imgsz)
        preprocessed_arrays: List[np.ndarray] = []
        sample_records: List[CalibrationSampleRecord] = []

        for idx, src_p in enumerate(selected_paths):
            dest_name = f"calib_{idx:04d}{src_p.suffix.lower()}"
            dest_path = self.images_dir / dest_name

            with Image.open(src_p) as img:
                orig_w, orig_h = img.size

            arr = preprocess_calibration_image(src_p, target_size=target_size)
            preprocessed_arrays.append(arr)

            # Save processed image copy
            Image.fromarray(arr).save(dest_path)
            sample_sha = compute_sha256(dest_path)

            sample_records.append(
                CalibrationSampleRecord(
                    source_path=str(src_p),
                    calib_filename=dest_name,
                    sha256=sample_sha,
                    original_size=[orig_w, orig_h],
                    preprocessed_shape=list(arr.shape),
                )
            )

        # Save single numpy file for Hailo DFC optimisation consumption
        calib_npy_path = self.output_dir / "calib_data.npy"
        np.save(str(calib_npy_path), np.stack(preprocessed_arrays))

        # Build and save manifest
        contract = {
            "colour_space": "RGB",
            "channel_order": "HWC_for_npy_NCHW_for_model",
            "resize_policy": "letterbox",
            "padding_value": 114,
            "target_dimensions": list(target_size),
            "dtype": "uint8_raw_with_alls_normalisation",
        }

        manifest = CalibrationManifest(
            count=len(sample_records),
            seed=seed,
            source=calib_cfg.source,
            preprocessing_contract=contract,
            samples=sample_records,
        )
        manifest_path = self.output_dir / "calibration_manifest.json"
        manifest.save(manifest_path)

        # Statistics
        stats_path = self.output_dir / "calibration_statistics.json"
        stats = {
            "sample_count": len(sample_records),
            "total_bytes": calib_npy_path.stat().st_size,
            "sha256": compute_sha256(calib_npy_path),
            "contract": contract,
        }
        atomic_write(stats_path, json.dumps(stats, indent=2))

        # Link canonical artifacts
        safe_symlink_or_copy(calib_npy_path, self.canonical_dir / "calib_data.npy")
        safe_symlink_or_copy(manifest_path, self.canonical_dir / "calibration_manifest.json")

        return calib_npy_path, manifest_path
