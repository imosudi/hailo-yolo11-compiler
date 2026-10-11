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


def test_picamera2_fallback_to_opencv() -> None:
    """Verify that Picamera2Source seamlessly falls back to OpenCV when picam2 is missing."""
    with patch("yolo_hailo_mlops.deployment.sources.try_import_picamera2", return_value=None):
        with patch("yolo_hailo_mlops.deployment.sources.cv2.VideoCapture") as mock_cap:
            mock_inst = MagicMock()
            mock_inst.isOpened.return_value = True
            mock_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_inst.get.return_value = 30.0
            mock_cap.return_value = mock_inst

            from yolo_hailo_mlops.deployment.sources import Picamera2Source
            source = Picamera2Source(camera_num=0, fallback_to_opencv=True)
            assert source.source_type == "camera"
            packet = source.read_frame()
            assert packet is not None
            source.release()

        # Strict mode without fallback raises DeploymentError
        with pytest.raises(DeploymentError) as exc_info:
            Picamera2Source(camera_num=0, fallback_to_opencv=False)
        assert exc_info.value.code == "E-DEPLOY-PICAM-001"


def test_opencv_camera_probing_when_index_0_empty() -> None:
    """Verify that OpenCVCameraSource automatically discovers an active camera node if /dev/video0 is empty."""
    from yolo_hailo_mlops.deployment.sources import OpenCVCameraSource

    # Mock device 0 failing to read frames (PiSP memory node), device 2 succeeding
    def mock_video_capture(idx, *args, **kwargs):
        inst = MagicMock()
        inst.isOpened.return_value = True
        if idx == 0:
            inst.read.return_value = (False, None)
        elif idx == 2:
            inst.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
        else:
            inst.isOpened.return_value = False
        inst.get.return_value = 30.0
        return inst

    with patch("yolo_hailo_mlops.deployment.sources.cv2.VideoCapture", side_effect=mock_video_capture):
        with patch("glob.glob", return_value=["/dev/video0", "/dev/video1", "/dev/video2"]):
            source = OpenCVCameraSource(device_index=0)
            assert source.device_index == 2
            packet = source.read_frame()
            assert packet is not None
            assert packet.source_name == "camera_2"
            source.release()


def test_opencv_camera_all_fail_raises_error() -> None:
    """Verify that OpenCVCameraSource raises E-DEPLOY-CAM-002 if all video devices fail to capture."""
    from yolo_hailo_mlops.deployment.sources import OpenCVCameraSource

    def mock_video_capture(idx, *args, **kwargs):
        inst = MagicMock()
        inst.isOpened.return_value = True
        inst.read.return_value = (False, None)
        return inst

    with patch("yolo_hailo_mlops.deployment.sources.cv2.VideoCapture", side_effect=mock_video_capture):
        with patch("glob.glob", return_value=["/dev/video0"]):
            with pytest.raises(DeploymentError) as exc_info:
                OpenCVCameraSource(device_index=0)
            assert exc_info.value.code == "E-DEPLOY-CAM-002"


def test_libcamera_stream_source() -> None:
    """Verify that LibcameraStreamSource decodes YUV420 byte stream from rpicam-vid."""
    from yolo_hailo_mlops.deployment.sources import LibcameraStreamSource

    w, h = 640, 480
    yuv_bytes = np.zeros((h * 3 // 2, w), dtype=np.uint8).tobytes()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdout.read.side_effect = [yuv_bytes, yuv_bytes, b""]

    with patch("shutil.which", return_value="/usr/bin/rpicam-vid"):
        with patch("subprocess.Popen", return_value=mock_proc):
            source = LibcameraStreamSource(camera_num=0, target_resolution=(w, h))
            assert source.source_type == "camera"
            assert source.resolution == (w, h)

            # First frame comes from cached packet
            packet = source.read_frame()
            assert packet is not None
            assert packet.frame.shape == (h, w, 3)
            assert packet.source_name == "libcamera_0"

            source.release()
            mock_proc.terminate.assert_called()


def test_picamera2_fallback_to_libcamera() -> None:
    """Verify Picamera2Source uses native rpicam-vid stream when picamera2 module is missing."""
    from yolo_hailo_mlops.deployment.sources import Picamera2Source

    w, h = 640, 480
    yuv_bytes = np.zeros((h * 3 // 2, w), dtype=np.uint8).tobytes()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdout.read.side_effect = [yuv_bytes, yuv_bytes, b""]

    with patch("yolo_hailo_mlops.deployment.sources.try_import_picamera2", return_value=None):
        with patch("shutil.which", return_value="/usr/bin/rpicam-vid"):
            with patch("subprocess.Popen", return_value=mock_proc):
                source = Picamera2Source(camera_num=0, target_resolution=(w, h))
                assert source.source_type == "camera"
                packet = source.read_frame()
                assert packet is not None
                assert packet.source_name == "libcamera_0"
                source.release()


def test_hailo_engine_fallback_when_platform_missing(tmp_path: Path) -> None:
    """Verify create_inference_engine gracefully falls back to ONNX/PyTorch if hailo_platform is missing."""
    from yolo_hailo_mlops.deployment.inference import ONNXInferenceEngine, create_inference_engine

    # Setup dummy HEF and dummy ONNX in root artifacts
    hef_dir = tmp_path / "hailo"
    hef_dir.mkdir(parents=True)
    (hef_dir / "model.hef").touch()

    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(parents=True)
    dummy_onnx = onnx_dir / "model.onnx"
    dummy_onnx.touch()

    mock_env = MagicMock()
    mock_env.has_hardware = True
    mock_env.has_hailort = True

    with patch("yolo_hailo_mlops.deployment.inference.detect_hailo_environment", return_value=mock_env):
        with patch("yolo_hailo_mlops.deployment.inference.try_import_hailo_platform", return_value=(None, "missing hailo_platform", "install it")):
            with patch("yolo_hailo_mlops.deployment.inference.ONNXInferenceEngine") as mock_onnx:
                mock_onnx.return_value = MagicMock()
                engine = create_inference_engine(backend="auto", root_artifacts=tmp_path)
                assert engine is not None
                mock_onnx.assert_called_once()


def test_create_inference_engine_target_cpu(tmp_path: Path) -> None:
    """Verify backend='cpu' forces CPU execution even when .hef is present."""
    from yolo_hailo_mlops.deployment.inference import create_inference_engine

    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(parents=True)
    dummy_onnx = onnx_dir / "model.onnx"
    dummy_onnx.touch()

    # Also touch a .hef to verify it is bypassed in favor of CPU model
    hef_dir = tmp_path / "hailo"
    hef_dir.mkdir(parents=True)
    (hef_dir / "model.hef").touch()

    with patch("yolo_hailo_mlops.deployment.inference.ONNXInferenceEngine") as mock_onnx:
        mock_onnx.return_value = MagicMock()
        engine = create_inference_engine(backend="cpu", root_artifacts=tmp_path)
        assert engine is not None
        mock_onnx.assert_called_once_with(dummy_onnx)


def test_create_inference_engine_target_hailo_aliases(tmp_path: Path) -> None:
    """Verify backend='hailo-hat' and 'npu' resolve to HailoInferenceEngine."""
    from yolo_hailo_mlops.deployment.inference import create_inference_engine

    hef_dir = tmp_path / "hailo"
    hef_dir.mkdir(parents=True)
    dummy_hef = hef_dir / "model.hef"
    dummy_hef.touch()

    with patch("yolo_hailo_mlops.deployment.inference.HailoInferenceEngine") as mock_hailo:
        mock_hailo.return_value = MagicMock()
        engine = create_inference_engine(backend="hailo-hat", root_artifacts=tmp_path)
        assert engine is not None
        mock_hailo.assert_called_once_with(dummy_hef)


def test_hailo_engine_missing_platform_diagnostic(tmp_path: Path) -> None:
    """Verify HailoInferenceEngine raises DeploymentError with detailed diagnostic."""
    import pytest
    from yolo_hailo_mlops.deployment.inference import HailoInferenceEngine
    from yolo_hailo_mlops.exceptions import DeploymentError

    dummy_hef = tmp_path / "model.hef"
    dummy_hef.touch()

    mock_env = MagicMock()
    mock_env.has_hardware = True

    with patch("yolo_hailo_mlops.deployment.inference.detect_hailo_environment", return_value=mock_env):
        with patch(
            "yolo_hailo_mlops.deployment.inference.try_import_hailo_platform",
            return_value=(None, "Installed for system Python 3.11", "Run /usr/bin/python3 ..."),
        ):
            with pytest.raises(DeploymentError) as exc_info:
                HailoInferenceEngine(dummy_hef)
            assert exc_info.value.code == "E-DEPLOY-RT-001"
            assert "Installed for system Python 3.11" in exc_info.value.message
            assert "Run /usr/bin/python3" in exc_info.value.remediation


def test_hailo_engine_successful_initialization_with_hef_stream_infos(tmp_path: Path) -> None:
    """Verify HailoInferenceEngine initializes cleanly when stream infos are on HEF."""
    from yolo_hailo_mlops.deployment.inference import HailoInferenceEngine

    dummy_hef = tmp_path / "model.hef"
    dummy_hef.touch()

    mock_env = MagicMock()
    mock_env.has_hardware = True

    mock_hailo = MagicMock()
    mock_hef_instance = MagicMock()
    in_info = MagicMock()
    in_info.name = "input_layer"
    out_info = MagicMock()
    out_info.name = "output_layer"
    mock_hef_instance.get_input_vstream_infos.return_value = [in_info]
    mock_hef_instance.get_output_vstream_infos.return_value = [out_info]
    mock_hailo.HEF.return_value = mock_hef_instance

    mock_vdev = MagicMock()
    mock_hailo.VDevice.return_value = mock_vdev

    mock_infer_pipeline = MagicMock()
    mock_infer_pipeline.infer.return_value = {"output_layer": np.zeros((1, 84, 8400), dtype=np.float32)}
    mock_hailo.InferVStreams.return_value = mock_infer_pipeline

    with patch("yolo_hailo_mlops.deployment.inference.detect_hailo_environment", return_value=mock_env):
        with patch("yolo_hailo_mlops.deployment.inference.try_import_hailo_platform", return_value=(mock_hailo, None, None)):
            engine = HailoInferenceEngine(dummy_hef)
            assert engine._input_name == "input_layer"
            assert engine._output_names == ["output_layer"]

            # Test infer
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            outputs, latency = engine.infer(dummy_img)
            assert "output_layer" in outputs
            assert latency >= 0

            engine.close()






