"""Preprocessing contract for calibration: RGB letterboxing to 640x640."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image


def letterbox_image(
    im: Image.Image,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Resize image with aspect ratio preservation and padding to target size."""
    target_w, target_h = target_size
    orig_w, orig_h = im.size

    # Calculate scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Bilinear resize
    resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Paste onto padded canvas
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, (pad_x, pad_y)


def preprocess_calibration_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Load, convert to RGB, letterbox, and return uint8 HWC array."""
    p = Path(image_path)
    with Image.open(p) as img:
        img_rgb = img.convert("RGB")
        boxed, _, _ = letterbox_image(img_rgb, target_size=target_size)
        arr = np.array(boxed, dtype=np.uint8)
        return arr
