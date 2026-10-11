"""Inference execution backends for Hailo-8L accelerator, ONNX Runtime, and PyTorch."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from yolo_hailo_mlops.exceptions import DeploymentError, HailoRuntimeError
from yolo_hailo_mlops.hailo.environment import detect_hailo_environment


class BaseInferenceEngine(ABC):
    """Abstract base class for inference execution backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend and target description."""
        pass

    @property
    @abstractmethod
    def is_hardware_accelerated(self) -> bool:
        """Whether inference runs on physical NPU/accelerator hardware."""
        pass

    @abstractmethod
    def infer(self, image_rgb: np.ndarray) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], float]:
        """Execute inference on a single (640, 640, 3) RGB image.

        Returns (outputs, inference_latency_ms).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release device contexts, streams, and sessions."""
        pass

    def __enter__(self) -> BaseInferenceEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class HailoInferenceEngine(BaseInferenceEngine):
    """Real-time inference engine executing on physical Hailo-8L accelerator via PCIe."""

    def __init__(self, hef_path: Union[str, Path]) -> None:
        self.hef_path = Path(hef_path).resolve()
        if not self.hef_path.is_file():
            raise DeploymentError(
                f"Compiled Hailo Executable Format (HEF) file not found: {self.hef_path}",
                code="E-DEPLOY-HEF-001",
                artifact=str(self.hef_path),
            )

        env = detect_hailo_environment()
        if not env.has_hardware:
            raise DeploymentError(
                "No physical Hailo-8L accelerator detected on host.",
                code="E-DEPLOY-HW-001",
                remediation="Ensure Raspberry Pi 5 AI HAT+ is seated securely and 'lspci' lists Hailo-8L.",
            )

        try:
            import hailo_platform  # type: ignore
        except (ImportError, ModuleNotFoundError) as err:
            raise DeploymentError(
                "HailoRT Python library ('hailo_platform') is required for live accelerator deployment.",
                code="E-DEPLOY-RT-001",
                remediation="Install HailoRT Python wheel or run inside a configured Hailo container/virtualenv.",
            ) from err

        self._hailo = hailo_platform
        self._target = None
        self._network_group = None
        self._activated_network = None
        self._infer_pipeline = None

        try:
            # 1. Initialize VDevice
            params = self._hailo.VDevice.create_params()
            self._target = self._hailo.VDevice(params)

            # 2. Load HEF
            self.hef = self._hailo.HEF(str(self.hef_path))

            # 3. Configure PCIe Stream Interface
            configure_params = self._hailo.ConfigureParams.create_from_hef(
                self.hef, interface=self._hailo.HailoStreamInterface.PCIe
            )
            network_groups = self._target.configure(self.hef, configure_params)
            self._network_group = network_groups[0]
            network_group_params = self._network_group.create_params()

            # 4. Configure Virtual Streams
            input_vstreams_params = self._hailo.InputVStreamParams.make(
                self._network_group, quantized=False, format_type=self._hailo.FormatType.UINT8
            )
            output_vstreams_params = self._hailo.OutputVStreamParams.make(
                self._network_group, quantized=False, format_type=self._hailo.FormatType.FLOAT32
            )

            # 5. Activate network and instantiate InferVStreams for low-latency streaming
            self._activated_network = self._network_group.activate(network_group_params)
            self._activated_network.__enter__()

            self._infer_pipeline = self._hailo.InferVStreams(
                self._network_group, input_vstreams_params, output_vstreams_params
            )
            self._infer_pipeline.__enter__()

            self._input_name = self._infer_pipeline.get_input_vstream_infos()[0].name
            self._output_names = [info.name for info in self._infer_pipeline.get_output_vstream_infos()]

        except Exception as e:
            self.close()
            raise DeploymentError(
                f"Failed to initialize Hailo-8L VDevice and configure streams: {e}",
                code="E-DEPLOY-INIT-001",
                artifact=str(self.hef_path),
            ) from e

    @property
    def backend_name(self) -> str:
        return "Hailo-8L Accelerator (PCIe)"

    @property
    def is_hardware_accelerated(self) -> bool:
        return True

    def infer(self, image_rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
        if self._infer_pipeline is None:
            raise DeploymentError("Hailo inference pipeline is not active or has been closed.")

        # Ensure batch dimension: (1, 640, 640, 3) uint8
        if image_rgb.ndim == 3:
            inp_batch = np.expand_dims(image_rgb, axis=0)
        else:
            inp_batch = image_rgb

        if inp_batch.dtype != np.uint8:
            inp_batch = inp_batch.astype(np.uint8)

        t0 = time.perf_counter()
        raw_outputs = self._infer_pipeline.infer({self._input_name: inp_batch})
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        return raw_outputs, latency_ms

    def close(self) -> None:
        if self._infer_pipeline is not None:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self._infer_pipeline = None

        if self._activated_network is not None:
            try:
                self._activated_network.__exit__(None, None, None)
            except Exception:
                pass
            self._activated_network = None

        if self._target is not None:
            try:
                self._target.release()
            except Exception:
                pass
            self._target = None


class ONNXInferenceEngine(BaseInferenceEngine):
    """CPU/GPU inference engine using ONNX Runtime."""

    def __init__(self, onnx_path: Union[str, Path]) -> None:
        self.onnx_path = Path(onnx_path).resolve()
        if not self.onnx_path.is_file():
            raise DeploymentError(
                f"ONNX model file not found: {self.onnx_path}",
                code="E-DEPLOY-ONNX-001",
                artifact=str(self.onnx_path),
            )

        try:
            import onnxruntime as ort
        except (ImportError, ModuleNotFoundError) as err:
            raise DeploymentError(
                "onnxruntime package is required for ONNX deployment fallback.",
                code="E-DEPLOY-ONNX-002",
                remediation="Run 'pip install onnxruntime' to enable ONNX CPU execution.",
            ) from err

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def backend_name(self) -> str:
        return "ONNX Runtime (CPU Fallback)"

    @property
    def is_hardware_accelerated(self) -> bool:
        return False

    def infer(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        # Preprocess to (1, 3, 640, 640) float32 normalized [0.0, 1.0]
        img = image_rgb.astype(np.float32) / 255.0
        if img.ndim == 3:
            # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            img = np.transpose(img, (2, 0, 1))
            inp_batch = np.expand_dims(img, axis=0)
        else:
            inp_batch = img

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: inp_batch})
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        return outputs[0], latency_ms

    def close(self) -> None:
        self.session = None


class PyTorchInferenceEngine(BaseInferenceEngine):
    """PyTorch inference fallback using Ultralytics YOLO checkpoint."""

    def __init__(self, checkpoint_path: Union[str, Path]) -> None:
        self.ckpt_path = Path(checkpoint_path).resolve()
        if not self.ckpt_path.is_file():
            raise DeploymentError(
                f"PyTorch checkpoint file not found: {self.ckpt_path}",
                code="E-DEPLOY-PT-001",
                artifact=str(self.ckpt_path),
            )

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.ckpt_path))
        except (ImportError, ModuleNotFoundError) as err:
            raise DeploymentError(
                "Ultralytics package is required for PyTorch model execution.",
                code="E-DEPLOY-PT-002",
            ) from err

    @property
    def backend_name(self) -> str:
        return "PyTorch (CPU Fallback)"

    @property
    def is_hardware_accelerated(self) -> bool:
        return False

    def infer(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch

        # Transpose (H, W, C) -> (1, C, H, W) float32 [0.0, 1.0]
        img = np.transpose(image_rgb, (2, 0, 1))
        inp = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        t0 = time.perf_counter()
        with torch.no_grad():
            preds = self.model.model(inp)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            out_arr = preds.cpu().numpy()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        return out_arr, latency_ms

    def close(self) -> None:
        self.model = None


def create_inference_engine(
    model_path: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    root_artifacts: Optional[Union[str, Path]] = None,
) -> BaseInferenceEngine:
    """Create inference engine according to target availability and backend preferences."""
    root = Path(root_artifacts) if root_artifacts else Path("artifacts")

    # Resolve candidate model paths if not specified
    if not model_path:
        # Search canonical artifacts locations
        candidates = [
            root / "latest" / "hailo" / "model.hef",
            root / "hailo" / "model.hef",
            root / "runs" / "latest" / "hailo" / "model.hef",
            root / "latest" / "onnx" / "model.onnx",
            root / "onnx" / "model.onnx",
            root / "runs" / "latest" / "onnx" / "model.onnx",
            root / "latest" / "pytorch" / "best.pt",
            root / "pytorch" / "best.pt",
            root / "runs" / "latest" / "pytorch" / "best.pt",
            Path("yolo11n.pt"),
        ]
        resolved_path = None
        for cand in candidates:
            if cand.is_file():
                resolved_path = cand
                break

        # Fallback: find most recent compiled HEF or ONNX within runs directory
        if not resolved_path and root.is_dir():
            hef_runs = sorted(root.glob("runs/**/hailo/*.hef"), key=lambda p: p.stat().st_mtime, reverse=True)
            if hef_runs:
                resolved_path = hef_runs[0]
            else:
                onnx_runs = sorted(root.glob("runs/**/onnx/*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
                if onnx_runs:
                    resolved_path = onnx_runs[0]

        if not resolved_path:
            raise DeploymentError(
                "No trained or compiled model artifact found. Specify '--model <path>' explicitly.",
                code="E-DEPLOY-MDL-001",
            )
        model_p = resolved_path
    else:
        model_p = Path(model_path).resolve()
        if not model_p.is_file():
            raise DeploymentError(
                f"Specified model path does not exist: {model_p}",
                code="E-DEPLOY-MDL-002",
                artifact=str(model_p),
            )

    ext = model_p.suffix.lower()

    # Explicit backend requests
    if backend == "hailo":
        if ext != ".hef":
            raise DeploymentError(
                f"Hailo backend requires a compiled .hef binary, got '{model_p.name}'",
                code="E-DEPLOY-BCK-001",
                artifact=str(model_p),
            )
        return HailoInferenceEngine(model_p)

    elif backend == "onnx":
        if ext != ".onnx":
            # Attempt to locate corresponding onnx
            onnx_alt = root / "onnx" / "model.onnx"
            if onnx_alt.is_file():
                model_p = onnx_alt
            else:
                raise DeploymentError(
                    f"ONNX backend requires an .onnx model, got '{model_p.name}'",
                    code="E-DEPLOY-BCK-002",
                )
        return ONNXInferenceEngine(model_p)

    elif backend == "pytorch":
        return PyTorchInferenceEngine(model_p)

    # Auto resolution
    if ext == ".hef":
        env = detect_hailo_environment()
        if env.has_hardware and env.has_hailort:
            return HailoInferenceEngine(model_p)
        else:
            # Fallback to ONNX if available
            onnx_cand = root / "onnx" / "model.onnx"
            if onnx_cand.is_file():
                return ONNXInferenceEngine(onnx_cand)
            pt_cand = root / "pytorch" / "best.pt"
            if pt_cand.is_file():
                return PyTorchInferenceEngine(pt_cand)
            # Re-raise informative error from Hailo engine
            return HailoInferenceEngine(model_p)

    elif ext == ".onnx":
        return ONNXInferenceEngine(model_p)
    elif ext == ".pt":
        return PyTorchInferenceEngine(model_p)
    else:
        raise DeploymentError(
            f"Unsupported model extension '{ext}'. Expected .hef, .onnx, or .pt.",
            code="E-DEPLOY-MDL-003",
            artifact=str(model_p),
        )
