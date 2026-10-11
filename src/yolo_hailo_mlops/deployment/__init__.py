"""Live deployment module for physical Raspberry Pi 5 + Hailo-8L accelerator."""

from __future__ import annotations

from yolo_hailo_mlops.deployment.annotator import VisualAnnotator
from yolo_hailo_mlops.deployment.inference import (
    BaseInferenceEngine,
    HailoInferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    create_inference_engine,
)
from yolo_hailo_mlops.deployment.pipeline import LiveDeploymentPipeline
from yolo_hailo_mlops.deployment.postprocess import Detection, LetterboxResult, YOLODecoder, letterbox
from yolo_hailo_mlops.deployment.sources import (
    BaseFrameSource,
    CameraSource,
    FramePacket,
    ImageDirectorySource,
    OpenCVCameraSource,
    Picamera2Source,
    VideoSource,
    create_frame_source,
)

__all__ = [
    "BaseFrameSource",
    "OpenCVCameraSource",
    "Picamera2Source",
    "VideoSource",
    "ImageDirectorySource",
    "FramePacket",
    "create_frame_source",
    "BaseInferenceEngine",
    "HailoInferenceEngine",
    "ONNXInferenceEngine",
    "PyTorchInferenceEngine",
    "create_inference_engine",
    "Detection",
    "LetterboxResult",
    "letterbox",
    "YOLODecoder",
    "VisualAnnotator",
    "LiveDeploymentPipeline",
]
