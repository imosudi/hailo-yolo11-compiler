"""Live YOLO11 deployment pipeline controller for camera, recorded video, and image directory streams."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from yolo_hailo_mlops.deployment.annotator import VisualAnnotator
from yolo_hailo_mlops.deployment.inference import BaseInferenceEngine, create_inference_engine
from yolo_hailo_mlops.deployment.postprocess import YOLODecoder, letterbox
from yolo_hailo_mlops.deployment.sources import (
    BaseFrameSource,
    FramePacket,
    ImageDirectorySource,
    create_frame_source,
)
from yolo_hailo_mlops.exceptions import DeploymentError
from yolo_hailo_mlops.utils.filesystem import ensure_dir


def is_gui_display_available() -> bool:
    """Check whether a graphical display server (Wayland or X11) is actively connected and reachable."""
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}" if hasattr(os, "getuid") else None)
    if wayland_display and xdg_runtime:
        wayland_sock = Path(xdg_runtime) / wayland_display
        if wayland_sock.exists():
            return True

    x11_display = os.environ.get("DISPLAY")
    if x11_display:
        disp_num = x11_display.lstrip(":").split(".")[0]
        if disp_num.isdigit():
            x_sock = Path(f"/tmp/.X11-unix/X{disp_num}")
            if x_sock.exists():
                return True
        return True

    return False


# OpenCV Qt backend lacks libqwayland.so on PyPI wheels; defaulting to xcb allows XWayland interoperability without crashing
if "QT_QPA_PLATFORM" not in os.environ and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
    os.environ["QT_QPA_PLATFORM"] = "xcb"


class LiveDeploymentPipeline:
    """Manages end-to-end ingestion, inference, postprocessing, annotation, and display/sink."""

    def __init__(
        self,
        source: Union[str, int, Path] = "camera",
        model_path: Optional[Union[str, Path]] = None,
        backend: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        show: Optional[bool] = None,
        save_dir: Optional[Union[str, Path]] = None,
        camera_type: str = "auto",
        fps_limit: Optional[float] = None,
        max_frames: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        num_classes: int = 80,
        root_artifacts: Optional[Union[str, Path]] = None,
        hud_enabled: bool = True,
    ) -> None:
        self.source_desc = source
        self.model_path = model_path
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.save_dir = Path(save_dir).resolve() if save_dir else None
        self.camera_type = camera_type
        self.fps_limit = fps_limit
        self.max_frames = max_frames
        self.class_names = class_names
        self.num_classes = num_classes
        self.root_artifacts = root_artifacts
        self.hud_enabled = hud_enabled

        # Auto-detect display availability if not explicitly specified
        if show is None:
            self.show = is_gui_display_available()
        else:
            self.show = show

        # Initialize core components
        self.source: BaseFrameSource = create_frame_source(
            source=self.source_desc,
            camera_type=self.camera_type,
            target_fps=self.fps_limit,
        )

        self.engine: BaseInferenceEngine = create_inference_engine(
            model_path=self.model_path,
            backend=self.backend,
            root_artifacts=self.root_artifacts,
        )

        self.decoder = YOLODecoder(
            num_classes=self.num_classes,
            class_names=self.class_names,
        )

        self.annotator = VisualAnnotator(
            num_classes=self.num_classes,
            hud_enabled=self.hud_enabled,
        )

        # Video writer handle if saving video/stream
        self._video_writer: Optional[cv2.VideoWriter] = None
        if self.save_dir:
            ensure_dir(self.save_dir)

    def _setup_video_writer(self, width: int, height: int, fps: float) -> Path:
        """Initialize OpenCV VideoWriter for saving output stream."""
        assert self.save_dir is not None
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_video_path = self.save_dir / f"deployment_run_{ts}.mp4"

        # Try H.264 or MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        actual_fps = fps if (fps and fps > 0) else 30.0
        self._video_writer = cv2.VideoWriter(str(out_video_path), fourcc, actual_fps, (width, height))
        return out_video_path

    def run(self) -> Dict[str, Any]:
        """Execute live deployment loop until completion or user interruption."""
        print("================================================================================")
        print(" Hailo-8L / YOLO11 Live Model Deployment")
        print("================================================================================")
        print(f"  Target Accelerator:   {self.engine.backend_name}")
        print(f"  Input Source:         {self.source.source_type} ({self.source_desc})")
        print(f"  Native Resolution:    {self.source.resolution[0]}x{self.source.resolution[1]}")
        print(f"  Confidence Threshold: {self.conf_threshold}")
        print(f"  IoU NMS Threshold:    {self.iou_threshold}")
        print(f"  Display Window:       {'ENABLED' if self.show else 'HEADLESS'}")
        if self.save_dir:
            print(f"  Save Directory:       {self.save_dir}")
        print("================================================================================\n")

        fps_history = deque(maxlen=30)
        infer_latency_history = deque(maxlen=50)

        total_frames = 0
        total_detections_count = 0
        t_start = time.perf_counter()
        t_prev_frame = t_start

        window_name = f"Hailo YOLO11 Live ({self.engine.backend_name})"
        saved_video_path = None
        saved_images_count = 0

        # Create window if GUI display enabled
        if self.show:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, min(1280, self.source.resolution[0]), min(720, self.source.resolution[1]))
            except Exception as e:
                print(f"[WARNING] GUI window initialization failed ({e}). Switching to headless mode.")
                self.show = False

        paused = False

        try:
            for packet in self.source:
                if self.max_frames and total_frames >= self.max_frames:
                    break

                t_frame_start = time.perf_counter()

                frame_bgr = packet.frame
                h_orig, w_orig = frame_bgr.shape[:2]

                # Setup video writer on first frame if saving stream/video
                if self.save_dir and self._video_writer is None and self.source.source_type in ("camera", "video"):
                    saved_video_path = self._setup_video_writer(w_orig, h_orig, self.source.fps)

                # 1. Preprocessing: Letterbox
                letterbox_res = letterbox(frame_bgr, target_shape=(self.imgsz, self.imgsz))

                # 2. Convert to RGB for inference
                frame_rgb = cv2.cvtColor(letterbox_res.image, cv2.COLOR_BGR2RGB)

                # 3. Model Inference
                raw_outputs, infer_ms = self.engine.infer(frame_rgb)
                infer_latency_history.append(infer_ms)

                # 4. YOLO11 Post-processing & NMS
                if isinstance(raw_outputs, dict):
                    # Hailo 6 detection heads
                    raw_preds = self.decoder.decode_hailo_heads(raw_outputs)
                else:
                    # ONNX / PyTorch output
                    raw_preds = self.decoder.decode_onnx_output(raw_outputs)

                detections = self.decoder.postprocess(
                    raw_preds=raw_preds,
                    letterbox_info=letterbox_res,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )

                total_frames += 1
                total_detections_count += len(detections)

                # Calculate instantaneous FPS
                t_frame_end = time.perf_counter()
                dt = t_frame_end - t_prev_frame
                t_prev_frame = t_frame_end
                instant_fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(instant_fps)
                current_fps = float(np.mean(fps_history))

                # Telemetry stats bundle
                perf_stats = {
                    "fps": current_fps,
                    "infer_latency_ms": infer_ms,
                    "backend_name": self.engine.backend_name,
                    "inference_device": "NPU (Hailo HAT)" if self.engine.is_hardware_accelerated else "CPU",
                    "num_detections": len(detections),
                    "source_name": packet.source_name,
                    "frame_id": packet.frame_id,
                }

                # 5. Visual Annotation
                annotated = self.annotator.annotate(frame_bgr, detections, perf_stats)

                # 6. Sinks: Save output
                if self.save_dir:
                    if self._video_writer:
                        self._video_writer.write(annotated)
                    elif self.source.source_type == "images":
                        out_img_path = self.save_dir / f"annotated_{packet.source_name}"
                        cv2.imwrite(str(out_img_path), annotated)
                        saved_images_count += 1

                # 7. Sinks: Live GUI Display
                if self.show:
                    try:
                        cv2.imshow(window_name, annotated)

                        # Dynamic delay for video playback or camera
                        delay = 1
                        if self.fps_limit and self.fps_limit > 0:
                            target_delay = (1.0 / self.fps_limit) - (time.perf_counter() - t_frame_start)
                            if target_delay > 0:
                                delay = int(target_delay * 1000)

                        key = cv2.waitKey(max(1, delay)) & 0xFF
                        if key in (ord("q"), 27):  # 'q' or ESC
                            print("\n[INFO] User requested termination (Exit key pressed).")
                            break
                        elif key == ord("p"):  # 'p' toggle pause
                            paused = not paused
                            while paused:
                                k = cv2.waitKey(100) & 0xFF
                                if k == ord("p"):
                                    paused = False
                                elif k in (ord("q"), 27):
                                    break
                        elif key == ord("s"):  # 's' snapshot
                            snap_dir = self.save_dir or Path("./artifacts/snapshots")
                            ensure_dir(snap_dir)
                            snap_p = snap_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(str(snap_p), annotated)
                            print(f"  [SNAPSHOT SAVED] {snap_p}")
                    except Exception as e:
                        print(f"\n[WARNING] Live GUI display error ({e}). Seamlessly switching to headless mode.")
                        self.show = False

                # Terminal log update if running headless or periodically
                if not self.show or total_frames % 60 == 0:
                    det_summary = ", ".join(f"{d.class_name} ({d.confidence:.2f})" for d in detections[:3])
                    if len(detections) > 3:
                        det_summary += f", +{len(detections)-3} more"
                    det_str = f" | [{det_summary}]" if det_summary else " | [No objects]"
                    sys.stdout.write(
                        f"\r[RUN] Frame: {total_frames:5d} | FPS: {current_fps:5.1f} | Latency: {infer_ms:5.1f}ms{det_str}"
                    )
                    sys.stdout.flush()

                if packet.is_last:
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Deployment interrupted by user (SIGINT).")

        finally:
            # Clean shutdown of devices and windows
            t_total_elapsed = time.perf_counter() - t_start
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None

            if self.show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            self.engine.close()
            self.source.release()

        avg_fps = (total_frames / t_total_elapsed) if t_total_elapsed > 0 else 0.0
        avg_infer = float(np.mean(infer_latency_history)) if infer_latency_history else 0.0

        print("\n\n================================================================================")
        print(" DEPLOYMENT SESSION SUMMARY")
        print("================================================================================")
        print(f"  Frames Processed:       {total_frames}")
        print(f"  Elapsed Time:           {t_total_elapsed:.2f}s")
        print(f"  Average Pipeline FPS:   {avg_fps:.2f} FPS")
        print(f"  Average Infer Latency:  {avg_infer:.2f} ms")
        print(f"  Total Detections:       {total_detections_count}")
        if saved_video_path:
            print(f"  Saved Output Video:     {saved_video_path}")
        if saved_images_count > 0:
            print(f"  Saved Annotated Images: {saved_images_count} (in {self.save_dir})")
        print("================================================================================\n")

        return {
            "frames_processed": total_frames,
            "elapsed_seconds": round(t_total_elapsed, 3),
            "avg_fps": round(avg_fps, 2),
            "avg_infer_latency_ms": round(avg_infer, 2),
            "total_detections": total_detections_count,
            "saved_video_path": str(saved_video_path) if saved_video_path else None,
            "saved_images_count": saved_images_count,
            "backend": self.engine.backend_name,
        }
