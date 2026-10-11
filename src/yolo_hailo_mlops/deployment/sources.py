"""Input stream ingestion for camera, recorded video, and image directory sources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
        self._cached_first_frame: Optional[np.ndarray] = None

        # Attempt V4L2 backend first on Linux
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.device_index)

        # Minimize buffer size to eliminate latency on live streams
        try:
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self._target_res and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_res[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_res[1])

        if self._target_fps and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Verify whether this camera device can actually capture frames
        can_capture = False
        if self.cap.isOpened():
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                can_capture = True
                self._cached_first_frame = test_frame
                self._width = int(test_frame.shape[1])
                self._height = int(test_frame.shape[0])

        # If chosen device cannot stream (e.g. Raspberry Pi 5 ISP node /dev/video0)
        # and default index 0 was requested, scan for an active camera device
        if not can_capture and self.device_index == 0:
            import glob
            dev_paths = sorted(
                glob.glob("/dev/video*"),
                key=lambda p: int(p.replace("/dev/video", "")) if p.replace("/dev/video", "").isdigit() else 999,
            )
            # Filter out known Raspberry Pi internal ISP, codec, and hardware accelerator nodes
            # to avoid blocking select() timeouts
            IGNORED_DRIVERS = ("pisp", "bcm2835", "codec", "isp", "unicam", "rpivid", "m2m", "cfe")
            valid_candidates = []
            for cand_path in dev_paths:
                cand_idx = int(cand_path.replace("/dev/video", "")) if cand_path.replace("/dev/video", "").isdigit() else None
                if cand_idx is None or cand_idx == 0:
                    continue
                sys_name_path = Path(f"/sys/class/video4linux/video{cand_idx}/name")
                if sys_name_path.is_file():
                    try:
                        name_str = sys_name_path.read_text(errors="ignore").lower()
                        if any(ign in name_str for ign in IGNORED_DRIVERS):
                            continue
                    except Exception:
                        pass
                valid_candidates.append(cand_idx)

            for cand_idx in valid_candidates:
                test_cap = cv2.VideoCapture(cand_idx, cv2.CAP_V4L2)
                if not test_cap.isOpened():
                    test_cap = cv2.VideoCapture(cand_idx)
                if test_cap.isOpened():
                    try:
                        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    r, f = test_cap.read()
                    if r and f is not None:
                        if self.cap.isOpened():
                            self.cap.release()
                        self.cap = test_cap
                        self.device_index = cand_idx
                        self._cached_first_frame = f
                        self._width = int(f.shape[1])
                        self._height = int(f.shape[0])
                        can_capture = True
                        print(f"[INFO] Auto-detected functional USB camera capture stream on /dev/video{cand_idx} (index 0 was non-streaming node).")
                        break
                    test_cap.release()

        if not can_capture:
            if self.cap.isOpened():
                self.cap.release()
            raise DeploymentError(
                f"Camera device '{self.device_index}' could not capture frames (read returned empty). "
                "On Raspberry Pi 5, CSI camera modules require Picamera2 ('sudo apt install -y python3-picamera2'), "
                "while USB webcams may require specifying an active capture node (e.g. '--source 20').",
                code="E-DEPLOY-CAM-002",
                remediation="For Raspberry Pi CSI cameras, install python3-picamera2 ('./scripts/bootstrap.sh'). For USB webcams, pass '--source <index>' (e.g. --source 20).",
            )

        if not hasattr(self, "_width"):
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

        if self._cached_first_frame is not None:
            frame = self._cached_first_frame
            self._cached_first_frame = None
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Brief retry to handle momentary frame drop on USB/V4L2
                for _ in range(3):
                    time.sleep(0.005)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
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


def try_auto_install_picamera2() -> bool:
    """Attempt non-interactive apt installation of python3-picamera2 on Raspberry Pi OS."""
    if not sys.platform.startswith("linux") or not shutil.which("apt-get"):
        return False
    try:
        # Check sudo non-interactive access
        res = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=2)
        if res.returncode == 0:
            print("[INFO] Attempting automated installation of python3-picamera2 via apt...")
            install_res = subprocess.run(
                ["sudo", "-n", "apt-get", "install", "-y", "--no-install-recommends", "python3-picamera2", "libcamera-tools", "v4l-utils"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if install_res.returncode == 0:
                print("[SUCCESS] python3-picamera2 installed successfully.")
                from yolo_hailo_mlops.provenance import ensure_system_dist_packages
                ensure_system_dist_packages()
                return True
    except Exception:
        pass
    return False


def try_import_picamera2() -> Optional[Any]:
    """Attempt to import Picamera2 from current environment or system dist-packages."""
    from yolo_hailo_mlops.provenance import ensure_system_dist_packages
    ensure_system_dist_packages()

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

    # If not found, attempt on-the-fly auto-installation via apt (if passwordless sudo is available)
    if try_auto_install_picamera2():
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


class LibcameraStreamSource(BaseFrameSource):
    """Raspberry Pi CSI Camera / AI Camera (Sony IMX500) frame ingestion via rpicam-vid / libcamera-vid."""

    def __init__(
        self,
        camera_num: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
    ) -> None:
        self.camera_num = camera_num
        self._width = target_resolution[0] if target_resolution else 640
        self._height = target_resolution[1] if target_resolution else 480
        self._fps = target_fps or 30.0
        self._frame_count = 0
        self._frame_bytes = (self._width * self._height * 3) // 2  # YUV420 format
        self._cached_first_frame: Optional[np.ndarray] = None

        rpicam_bin = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not rpicam_bin:
            raise DeploymentError(
                "Neither 'rpicam-vid' nor 'libcamera-vid' binary found on system.",
                code="E-DEPLOY-LIBCAM-001",
                remediation="Install libcamera-apps with 'sudo apt-get install -y libcamera-apps' or './scripts/bootstrap.sh'.",
            )

        cmd = [
            rpicam_bin,
            "-t", "0",
            "--camera", str(self.camera_num),
            "--width", str(self._width),
            "--height", str(self._height),
            "--framerate", str(int(self._fps)),
            "--codec", "yuv420",
            "--nopreview",
            "-o", "-",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_bytes * 2,
            )
        except Exception as e:
            raise DeploymentError(
                f"Failed to start libcamera capture process ({rpicam_bin}): {e}",
                code="E-DEPLOY-LIBCAM-002",
            ) from e

        # Read first frame to verify hardware pipeline is delivering frames
        packet = self.read_frame()
        if packet is None:
            self.release()
            raise DeploymentError(
                "Raspberry Pi Camera (libcamera) started but produced no frame data.",
                code="E-DEPLOY-LIBCAM-003",
                remediation="Ensure camera ribbon cable is firmly seated and run 'rpicam-hello' to test sensor.",
            )
        self._cached_first_frame = packet.frame

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
        if not hasattr(self, "process") or self.process is None or self.process.poll() is not None:
            return None

        if self._cached_first_frame is not None:
            frame_bgr = self._cached_first_frame
            self._cached_first_frame = None
        else:
            assert self.process.stdout is not None
            raw = self.process.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                return None
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self._height * 3 // 2, self._width))
            frame_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        self._frame_count += 1
        return FramePacket(
            frame_id=self._frame_count,
            frame=frame_bgr,
            timestamp=time.perf_counter(),
            source_name=f"libcamera_{self.camera_num}",
            is_last=False,
        )

    def release(self) -> None:
        if hasattr(self, "process") and self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


class Picamera2Source(BaseFrameSource):
    """Raspberry Pi CSI Camera Module / AI Camera (Sony IMX500) via Picamera2 or native libcamera stream."""

    def __init__(
        self,
        camera_num: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
        fallback_to_opencv: bool = True,
    ) -> None:
        self.camera_num = camera_num
        self._frame_count = 0
        self._width = target_resolution[0] if target_resolution else 640
        self._height = target_resolution[1] if target_resolution else 480
        self._fps = target_fps or 30.0
        self._fallback_libcam: Optional[LibcameraStreamSource] = None
        self._fallback_opencv: Optional[OpenCVCameraSource] = None
        self.picam2 = None

        # 1. Attempt Picamera2 Python library
        picam2_cls = try_import_picamera2()
        if picam2_cls is not None:
            try:
                self.picam2 = picam2_cls(camera_num=self.camera_num)
                config = self.picam2.create_preview_configuration(
                    main={"size": (self._width, self._height), "format": "BGR888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                return
            except Exception as e:
                print(f"[NOTICE] Picamera2 device initialisation error ({e}). Attempting libcamera stream fallback...")

        # 2. Attempt native Raspberry Pi libcamera stream (rpicam-vid / libcamera-vid)
        # Directly supports Raspberry Pi AI Camera (Sony IMX500), Camera Module 3, and CSI sensors
        if shutil.which("rpicam-vid") or shutil.which("libcamera-vid"):
            try:
                print(f"[INFO] Seamlessly utilising native Raspberry Pi Camera stream (rpicam-vid/libcamera) for camera {camera_num}...")
                self._fallback_libcam = LibcameraStreamSource(
                    camera_num=self.camera_num,
                    target_resolution=(self._width, self._height),
                    target_fps=self._fps,
                )
                return
            except Exception as e:
                print(f"[NOTICE] Native libcamera stream error ({e}).")

        # 3. Fallback to OpenCV if permitted
        if fallback_to_opencv:
            print(
                f"[NOTICE] Picamera2/rpicam-vid not available. Attempting fallback to OpenCV camera (/dev/video{camera_num})...\n"
                "         (To enable native Raspberry Pi CSI Camera Module, run './scripts/bootstrap.sh' or 'sudo apt install -y python3-picamera2 libcamera-apps')"
            )
            try:
                self._fallback_opencv = OpenCVCameraSource(
                    device_index=camera_num,
                    target_fps=target_fps,
                    target_resolution=target_resolution,
                )
                return
            except DeploymentError as e:
                raise DeploymentError(
                    f"CSI camera stream unavailable, and fallback OpenCV camera failed: {e.message}",
                    code="E-DEPLOY-PICAM-001",
                    remediation="Install libcamera-apps / python3-picamera2 with './scripts/bootstrap.sh' or 'sudo apt-get install -y python3-picamera2 libcamera-apps'.",
                ) from e

        raise DeploymentError(
            "Raspberry Pi CSI camera requires Picamera2 or rpicam-apps ('rpicam-vid'). Install via './scripts/bootstrap.sh' or 'sudo apt install -y python3-picamera2 libcamera-apps'.",
            code="E-DEPLOY-PICAM-001",
            remediation="Run './scripts/bootstrap.sh' or 'sudo apt-get install -y python3-picamera2 libcamera-apps', or use '--camera-type opencv'.",
        )

    @property
    def source_type(self) -> str:
        return "camera"

    @property
    def fps(self) -> float:
        if self._fallback_libcam is not None:
            return self._fallback_libcam.fps
        if self._fallback_opencv is not None:
            return self._fallback_opencv.fps
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._fallback_libcam is not None:
            return self._fallback_libcam.resolution
        if self._fallback_opencv is not None:
            return self._fallback_opencv.resolution
        return (self._width, self._height)

    @property
    def total_frames(self) -> Optional[int]:
        return None

    def read_frame(self) -> Optional[FramePacket]:
        if self._fallback_libcam is not None:
            return self._fallback_libcam.read_frame()
        if self._fallback_opencv is not None:
            return self._fallback_opencv.read_frame()

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
        if self._fallback_libcam is not None:
            self._fallback_libcam.release()
            self._fallback_libcam = None
        if self._fallback_opencv is not None:
            self._fallback_opencv.release()
            self._fallback_opencv = None

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

        if camera_type in ("picamera2", "libcamera", "rpicam"):
            return Picamera2Source(camera_num=int(cam_idx), target_resolution=target_resolution, target_fps=target_fps)
        elif camera_type == "opencv":
            return OpenCVCameraSource(device_index=cam_idx, target_fps=target_fps, target_resolution=target_resolution)
        else:
            # Auto: Try Picamera2 / native libcamera on Raspberry Pi first without forcing OpenCV fallback
            try:
                return Picamera2Source(
                    camera_num=int(cam_idx),
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    fallback_to_opencv=False,
                )
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
