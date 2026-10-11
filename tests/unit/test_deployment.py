"""Unit tests for YOLO11 live deployment module: sources, postprocess, annotator, and pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from yolo_hailo_mlops.deployment import (
    Detection,
    ImageDirectorySource,
    LetterboxResult,
    LiveDeploymentPipeline,
    VideoSource,
    VisualAnnotator,
    YOLODecoder,
    create_frame_source,
    create_inference_engine,
    letterbox,
)
from yolo_hailo_mlops.exceptions import DeploymentError


def test_letterbox_resizing() -> None:
    """Verify that letterbox preserves aspect ratio and outputs expected dimensions."""
    # Create non-square test image (800x400)
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    res = letterbox(img, target_shape=(640, 640))

    assert res.image.shape == (640, 640, 3)
    assert res.orig_shape == (400, 800)
    # Ratio: 640 / 800 = 0.8
    assert pytest.approx(res.ratio, 0.01) == 0.8
    # Height unpadded: 400 * 0.8 = 320 -> vertical padding dw = 0, dh = (640 - 320) / 2 = 160
    assert pytest.approx(res.pad_w, 1.0) == 0.0
    assert pytest.approx(res.pad_h, 1.0) == 160.0


def test_yolo_decoder_onnx() -> None:
    """Verify decoding of ONNX output (1, 84, 8400)."""
    decoder = YOLODecoder(num_classes=80)
    # Synthetic ONNX output: (1, 84, 10)
    raw_out = np.zeros((1, 84, 10), dtype=np.float32)

    # Box 0: cx=320, cy=320, w=100, h=100, class 0 (person) score = 0.95
    raw_out[0, 0, 0] = 320.0
    raw_out[0, 1, 0] = 320.0
    raw_out[0, 2, 0] = 100.0
    raw_out[0, 3, 0] = 100.0
    raw_out[0, 4, 0] = 0.95  # class 0

    decoded = decoder.decode_onnx_output(raw_out)
    assert decoded.shape == (10, 84)

    # Verify xyxy conversion: x1 = 320 - 50 = 270, y1 = 270, x2 = 370, y2 = 370
    assert pytest.approx(decoded[0, 0], 0.1) == 270.0
    assert pytest.approx(decoded[0, 1], 0.1) == 270.0
    assert pytest.approx(decoded[0, 2], 0.1) == 370.0
    assert pytest.approx(decoded[0, 3], 0.1) == 370.0

    letter_res = LetterboxResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        ratio=1.0,
        pad_w=0.0,
        pad_h=0.0,
        orig_shape=(640, 640),
    )

    dets = decoder.postprocess(decoded, letter_res, conf_threshold=0.5, iou_threshold=0.45)
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].class_name == "person"
    assert pytest.approx(dets[0].confidence, 0.01) == 0.95


def test_yolo_decoder_hailo_heads() -> None:
    """Verify decoding of Hailo-8L 6-head output dictionary."""
    decoder = YOLODecoder(num_classes=80, strides=(8, 16, 32))

    # Construct mock heads
    outputs = {
        "box_0": np.zeros((1, 80, 80, 64), dtype=np.float32),
        "cls_0": np.full((1, 80, 80, 80), -10.0, dtype=np.float32),  # very low logits
        "box_1": np.zeros((1, 40, 40, 64), dtype=np.float32),
        "cls_1": np.full((1, 40, 40, 80), -10.0, dtype=np.float32),
        "box_2": np.zeros((1, 20, 20, 64), dtype=np.float32),
        "cls_2": np.full((1, 20, 20, 80), -10.0, dtype=np.float32),
    }

    # Inject high score at center of scale 0 for class 2 (car)
    outputs["cls_0"][0, 40, 40, 2] = 5.0  # sigmoid(5.0) ~ 0.993

    decoded = decoder.decode_hailo_heads(outputs)
    # Total anchors: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
    assert decoded.shape == (8400, 84)

    letter_res = LetterboxResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        ratio=1.0,
        pad_w=0.0,
        pad_h=0.0,
        orig_shape=(640, 640),
    )

    dets = decoder.postprocess(decoded, letter_res, conf_threshold=0.5, iou_threshold=0.45)
    assert len(dets) >= 1
    assert dets[0].class_id == 2
    assert dets[0].class_name == "car"
    assert dets[0].confidence > 0.9


def test_image_directory_source(tmp_path: Path) -> None:
    """Verify ImageDirectorySource ingestion."""
    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()

    # Create 3 dummy images
    for i in range(3):
        img_p = img_dir / f"img_{i:02d}.png"
        img = Image.new("RGB", (320, 240), color=(i * 50, 100, 150))
        img.save(img_p)

    source = ImageDirectorySource(img_dir)
    assert source.source_type == "images"
    assert source.total_frames == 3

    packets = list(source)
    assert len(packets) == 3
    assert packets[0].frame.shape == (240, 320, 3)
    assert packets[2].is_last is True
    source.release()


def test_video_source(tmp_path: Path) -> None:
    """Verify VideoSource frame reading and EOF detection."""
    vid_p = tmp_path / "test_video.mp4"
    out = cv2.VideoWriter(str(vid_p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(5):
        frame = np.full((240, 320, 3), i * 40, dtype=np.uint8)
        out.write(frame)
    out.release()

    source = VideoSource(vid_p)
    assert source.source_type == "video"
    assert source.total_frames == 5

    packets = list(source)
    assert len(packets) == 5
    assert packets[0].frame.shape == (240, 320, 3)
    source.release()


def test_visual_annotator() -> None:
    """Verify VisualAnnotator bounding box overlay and HUD drawing."""
    annotator = VisualAnnotator(num_classes=80, hud_enabled=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = Detection(
        bbox=(50.0, 60.0, 200.0, 300.0),
        confidence=0.88,
        class_id=0,
        class_name="person",
    )

    stats = {
        "fps": 45.0,
        "infer_latency_ms": 12.5,
        "backend_name": "Hailo-8L",
        "num_detections": 1,
        "source_name": "cam",
    }

    annotated = annotator.annotate(frame, [det], stats)
    assert annotated.shape == (480, 640, 3)
    # Verify pixels are modified (not black anymore)
    assert np.any(annotated > 0)


def test_create_frame_source_factory(tmp_path: Path) -> None:
    """Verify create_frame_source factory resolution."""
    # Directory
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "a.jpg").touch()
    cv2.imwrite(str(d / "a.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))

    src = create_frame_source(d)
    assert isinstance(src, ImageDirectorySource)
    src.release()

    # Invalid path
    with pytest.raises(DeploymentError):
        create_frame_source(tmp_path / "non_existent_file.xyz")
