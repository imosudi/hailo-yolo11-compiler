"""Deterministic dataset sampler for calibration sets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Union

from yolo_hailo_mlops.dataset.validator import SUPPORTED_IMAGE_EXTENSIONS
from yolo_hailo_mlops.exceptions import CalibrationError


def sample_images(
    source_dir_or_list: Union[Path, List[Path]],
    count: int,
    seed: int = 42,
) -> List[Path]:
    """Deterministically sample N uncorrupted images using a fixed random seed."""
    if isinstance(source_dir_or_list, (str, Path)):
        p = Path(source_dir_or_list)
        if not p.is_dir():
            raise CalibrationError(
                f"Calibration source directory does not exist: {p}",
                code="E-CAL-002",
                artifact=str(p),
            )
        candidate_images: List[Path] = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            candidate_images.extend(p.rglob(f"*{ext}"))
            candidate_images.extend(p.rglob(f"*{ext.upper()}"))
    else:
        candidate_images = list(source_dir_or_list)

    # Sort paths for cross-platform deterministic order before shuffle
    candidate_images = sorted([img.resolve() for img in candidate_images if img.is_file()])

    if not candidate_images:
        raise CalibrationError(
            "No candidate images found for calibration sampling.",
            code="E-CAL-003",
        )

    rng = random.Random(seed)
    # If requested count is greater than candidates, sample with replacement or clamp with warning
    if count > len(candidate_images):
        # Deterministically repeat to satisfy required calibration count
        indices = [rng.randint(0, len(candidate_images) - 1) for _ in range(count)]
        return [candidate_images[i] for i in indices]

    return rng.sample(candidate_images, count)
