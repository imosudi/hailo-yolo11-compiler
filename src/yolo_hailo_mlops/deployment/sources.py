"""Input stream ingestion for camera, recorded video, and image directory sources."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from yolo_hailo_mlops.exceptions import DeploymentError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


@dataclass
class FramePacket:
    """Represents a single ingested frame and metadata."""
    frame_id: int
    frame: np.ndarray  # BGR format (H, W, 3)
    timestamp: float
    source_name: str
    is_last: bool = False
    metadata: Optional[dict] = None


class BaseFrameSource(ABC):
    """Abstract base class for all frame sources."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type of source: 'camera', 'video', or 'images'."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Native or configured frames per second."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Source resolution as (width, height)."""
        pass

    @property
    @abstractmethod
    def total_frames(self) -> Optional[int]:
        """Total frame count if known, or None for live streams."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[FramePacket]:
        """Read and return next frame, or None if stream ended."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying hardware or file handles."""
        pass

    def __iter__(self) -> Iterator[FramePacket]:
        return self

    def __next__(self) -> FramePacket:
        pkt = self.read_frame()
        if pkt is None:
            raise StopIteration
        return pkt

    def __enter__(self) -> BaseFrameSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class OpenCVCameraSource(BaseFrameSource):
    """Live camera stream using OpenCV VideoCapture (V4L2 / USB webcam)."""

    def __init__(
        self,
        device_index: Union[int, str] = 0,
        target_fps: Optional[float] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.device_index = int(device_index) if str(device_index).isdigit() else device_index
        self._target_fps = target_fps
        self._target_res = target_resolution
        self._frame_count = 0

        # Attempt V4L2 backend first on Linux
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise DeploymentError(
                f"Failed to open camera device: {self.device_index}",
                code="E-DEPLOY-CAM-001",
                remediation="Ensure camera is connected and permissions allow access to /dev/video* (e.g. user in 'video' group).",
            )

        # Minimize buffer size to eliminate latency on live streams
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self._target_res:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_res[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_res[1])

        if self._target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def source_type(self) -> str:
        return "camera"

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> Optional[int]:
        return None

    def read_frame(self) -> Optional[FramePacket]:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        self._frame_count += 1
        return FramePacket(
            frame_id=self._frame_count,
            frame=frame,
            timestamp=time.perf_counter(),
            source_name=f"camera_{self.device_index}",
            is_last=False,
        )

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None


def try_import_picamera2() -> Optional[Any]:
    """Attempt to import Picamera2 from current environment or system dist-packages."""
    try:
        from picam2 import Picamera2  # type: ignore
        return Picamera2
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from picamera2 import Picamera2  # type: ignore
        return Picamera2
    except (ImportError, ModuleNotFoundError):
        pass

    # Check system dist-packages (common when installed via 'sudo apt install python3-picamera2' on Raspberry Pi)
    import sys
    for dist_dir in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
        if os.path.exists(dist_dir) and dist_dir not in sys.path:
            sys.path.append(dist_dir)
            try:
                from picam2 import Picamera2  # type: ignore
                return Picamera2
            except (ImportError, ModuleNotFoundError):
                pass
            try:
                from picamera2 import Picamera2  # type: ignore
                return Picamera2
            except (ImportError, ModuleNotFoundError):
                pass

    return None


class Picamera2Source(BaseFrameSource):
    """Raspberry Pi CSI Camera Module via Picamera2 library."""

    def __init__(
        self,
        camera_num: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
    ) -> None:
        self.camera_num = camera_num
        self._frame_count = 0
        self._width = target_resolution[0] if target_resolution else 640
        self._height = target_resolution[1] if target_resolution else 480
        self._fps = target_fps or 30.0

        picam2_cls = try_import_picamera2()
        if picam2_cls is None:
            raise DeploymentError(
                "Picamera2 is not installed. Install with 'sudo apt install -y python3-picamera2' or use OpenCV camera.",
                code="E-DEPLOY-PICAM-001",
                remediation="Run 'sudo apt-get install -y python3-picamera2' on Raspberry Pi OS, or use '--camera-type opencv' for USB webcam.",
            )

        try:
            self.picam2 = picam2_cls(camera_num=self.camera_num)
            config = self.picam2.create_preview_configuration(
                main={"size": (self._width, self._height), "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
        except Exception as e:
            raise DeploymentError(
                f"Failed to start Picamera2 hardware device: {e}",
                code="E-DEPLOY-PICAM-002",
                remediation="Verify ribbon cable connection and run 'rpicam-hello' to test hardware detection.",
            ) from e

    @property
    def source_type(self) -> str:
        return "camera"

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> Optional[int]:
        return None

    def read_frame(self) -> Optional[FramePacket]:
        if not hasattr(self, "picam2") or self.picam2 is None:
            return None
        try:
            frame = self.picam2.capture_array()
            self._frame_count += 1
            return FramePacket(
                frame_id=self._frame_count,
                frame=frame,
                timestamp=time.perf_counter(),
                source_name=f"picam2_{self.camera_num}",
                is_last=False,
            )
        except Exception:
            return None

    def release(self) -> None:
        if hasattr(self, "picam2") and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None


# General CameraSource alias
CameraSource = OpenCVCameraSource


class VideoSource(BaseFrameSource):
    """Recorded video file ingestion (.mp4, .avi, .mkv, etc.)."""

    def __init__(self, video_path: Union[str, Path]) -> None:
        self.video_path = Path(video_path).resolve()
        if not self.video_path.is_file():
            raise DeploymentError(
                f"Specified video file does not exist: {self.video_path}",
                code="E-DEPLOY-VID-001",
                artifact=str(self.video_path),
            )

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise DeploymentError(
                f"Failed to open video file: {self.video_path}",
                code="E-DEPLOY-VID-002",
                artifact=str(self.video_path),
            )

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = 0

    @property
    def source_type(self) -> str:
        return "video"

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> Optional[int]:
        return self._total_frames if self._total_frames > 0 else None

    def read_frame(self) -> Optional[FramePacket]:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        self._frame_count += 1
        is_last = (self._total_frames > 0 and self._frame_count >= self._total_frames)
        return FramePacket(
            frame_id=self._frame_count,
            frame=frame,
            timestamp=time.perf_counter(),
            source_name=self.video_path.name,
            is_last=is_last,
        )

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None


class ImageDirectorySource(BaseFrameSource):
    """Image files ingestion from directory or single image file."""

    def __init__(self, target_path: Union[str, Path]) -> None:
        p = Path(target_path).resolve()
        if not p.exists():
            raise DeploymentError(
                f"Image path does not exist: {p}",
                code="E-DEPLOY-IMG-001",
                artifact=str(p),
            )

        self.image_files: List[Path] = []
        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise DeploymentError(
                    f"Unsupported image file extension: {p.suffix}",
                    code="E-DEPLOY-IMG-002",
                    artifact=str(p),
                )
            self.image_files = [p]
        else:
            for item in sorted(p.iterdir()):
                if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    self.image_files.append(item)

        if not self.image_files:
            raise DeploymentError(
                f"No supported images found in directory: {p}",
                code="E-DEPLOY-IMG-003",
                artifact=str(p),
            )

        # Inspect first image for dimensions
        first_img = cv2.imread(str(self.image_files[0]))
        if first_img is None:
            raise DeploymentError(
                f"Unable to read initial image file: {self.image_files[0]}",
                code="E-DEPLOY-IMG-004",
                artifact=str(self.image_files[0]),
            )

        self._height, self._width = first_img.shape[:2]
        self._current_idx = 0
        self._total_frames = len(self.image_files)

    @property
    def source_type(self) -> str:
        return "images"

    @property
    def fps(self) -> float:
        return 1.0  # Default 1 frame per second step for static images

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> Optional[int]:
        return self._total_frames

    def read_frame(self) -> Optional[FramePacket]:
        if self._current_idx >= len(self.image_files):
            return None

        img_p = self.image_files[self._current_idx]
        frame = cv2.imread(str(img_p))
        if frame is None:
            raise DeploymentError(
                f"Corrupted or unreadable image file: {img_p}",
                code="E-DEPLOY-IMG-005",
                artifact=str(img_p),
            )

        self._current_idx += 1
        is_last = (self._current_idx >= len(self.image_files))

        return FramePacket(
            frame_id=self._current_idx,
            frame=frame,
            timestamp=time.perf_counter(),
            source_name=img_p.name,
            is_last=is_last,
            metadata={"file_path": str(img_p)},
        )

    def release(self) -> None:
        self.image_files = []


def create_frame_source(
    source: Union[str, int, Path],
    camera_type: str = "auto",
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> BaseFrameSource:
    """Factory function creating the appropriate frame source based on input descriptor."""
    str_src = str(source).strip()

    # 1. Camera identification
    is_cam = (
        str_src.isdigit()
        or str_src.lower() in ("camera", "cam", "webcam")
        or str_src.startswith("/dev/video")
    )
    if is_cam:
        cam_idx = int(str_src) if str_src.isdigit() else 0
        if str_src.startswith("/dev/video"):
            try:
                cam_idx = int(str_src.replace("/dev/video", ""))
            except ValueError:
                cam_idx = str_src  # type: ignore

        if camera_type == "picamera2":
            return Picamera2Source(camera_num=int(cam_idx), target_resolution=target_resolution, target_fps=target_fps)
        elif camera_type == "opencv":
            return OpenCVCameraSource(device_index=cam_idx, target_fps=target_fps, target_resolution=target_resolution)
        else:
            # Auto: Try Picamera2 on Raspberry Pi if available, fallback to OpenCV
            picam2_cls = try_import_picamera2()
            if picam2_cls is not None:
                try:
                    return Picamera2Source(camera_num=int(cam_idx), target_resolution=target_resolution, target_fps=target_fps)
                except Exception:
                    pass
            return OpenCVCameraSource(device_index=cam_idx, target_fps=target_fps, target_resolution=target_resolution)

    # 2. File or Directory identification
    path_obj = Path(str_src).resolve()
    if not path_obj.exists():
        raise DeploymentError(
            f"Input source '{source}' does not exist as a file, directory, or camera identifier.",
            code="E-DEPLOY-SRC-001",
        )

    if path_obj.is_dir():
        return ImageDirectorySource(path_obj)

    if path_obj.is_file():
        ext = path_obj.suffix.lower()
        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            return VideoSource(path_obj)
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            return ImageDirectorySource(path_obj)
        else:
            raise DeploymentError(
                f"Unrecognized file format '{ext}' for input source: {path_obj}",
                code="E-DEPLOY-SRC-002",
                artifact=str(path_obj),
            )

    raise DeploymentError(
        f"Unable to resolve frame source for: {source}",
        code="E-DEPLOY-SRC-003",
    )
