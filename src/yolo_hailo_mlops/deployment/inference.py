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


def try_auto_install_hailort() -> bool:
    """Attempt non-interactive apt installation of hailo-all / python3-hailort on Raspberry Pi OS."""
    import shutil
    import subprocess
    import sys
    if not sys.platform.startswith("linux") or not shutil.which("apt-get"):
        return False
    try:
        res = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=2)
        if res.returncode == 0:
            print("[INFO] Attempting automated installation of HailoRT packages via apt...")
            pkgs = ["python3-hailort", "hailort"]
            cache_check = subprocess.run(["apt-cache", "show", "hailo-all"], capture_output=True, timeout=3)
            if cache_check.returncode == 0:
                pkgs.insert(0, "hailo-all")

            install_res = subprocess.run(
                ["sudo", "-n", "apt-get", "install", "-y"] + pkgs,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if install_res.returncode == 0:
                print("[SUCCESS] HailoRT packages installed successfully.")
                from yolo_hailo_mlops.provenance import ensure_system_dist_packages
                ensure_system_dist_packages()
                return True
    except Exception:
        pass
    return False


def try_import_hailo_platform() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Attempt to import hailo_platform, linking system packages or installing python3-hailort.

    Returns (hailo_module, diagnostic_cause, remediation_hint).
    """
    import glob
    import importlib
    import os
    import subprocess
    import sys
    from yolo_hailo_mlops.provenance import ensure_system_dist_packages

    ensure_system_dist_packages()
    importlib.invalidate_caches()

    last_err: Optional[str] = None

    # 1. Direct import
    try:
        import hailo_platform  # type: ignore
        return hailo_platform, None, None
    except Exception as exc:
        last_err = str(exc)

    # 2. Try importing hailort or pyhailort
    try:
        import hailort  # type: ignore
        if hasattr(hailort, "VDevice") or hasattr(hailort, "HEF"):
            return hailort, None, None
    except Exception:
        pass

    # 3. Check for pre-built Hailo wheels in common locations
    wheel_patterns = [
        "/usr/share/hailort*/*.whl",
        "/usr/share/hailort*/**/*.whl",
        "/opt/hailo*/*.whl",
        "/opt/hailo*/**/*.whl",
        "/home/*/.hailo*/*.whl",
        "/home/*/.hailo*/**/*.whl",
        "/home/*/*.whl",
    ]
    for pattern in wheel_patterns:
        for whl in glob.glob(pattern, recursive=True):
            if "hailo" in whl.lower():
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", whl], capture_output=True, timeout=60)
                    importlib.invalidate_caches()
                    import hailo_platform  # type: ignore
                    return hailo_platform, None, None
                except Exception:
                    pass

    # 4. Try auto-installing via apt
    if try_auto_install_hailort():
        importlib.invalidate_caches()
        try:
            import hailo_platform  # type: ignore
            return hailo_platform, None, None
        except Exception as exc:
            last_err = str(exc)

    # 5. Check if system Python interpreter (/usr/bin/python3) has hailo_platform
    system_py = "/usr/bin/python3"
    curr_py = sys.executable
    if os.path.isfile(system_py) and Path(system_py).resolve() != Path(curr_py).resolve():
        try:
            chk = subprocess.run(
                [system_py, "-c", "import hailo_platform; print('OK')"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if "OK" in chk.stdout:
                sys_ver = subprocess.run(
                    [system_py, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                curr_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                cause = (
                    f"HailoRT Python library ('hailo_platform') is installed for system Python {sys_ver} ({system_py}), "
                    f"but current environment is running Python {curr_ver} ({curr_py}). "
                    f"C-extensions cannot be loaded across different Python minor versions."
                )
                remediation = (
                    f"1. Run live Hailo deployment using system Python:\n"
                    f"   {system_py} train_and_compile.py deploy --source 0 --camera-type picamera2 --hailo\n"
                    f"2. Or run live deployment on Host CPU in current environment:\n"
                    f"   python train_and_compile.py deploy --source 0 --camera-type picamera2 --cpu"
                )
                return None, cause, remediation
        except Exception:
            pass

    cause = f"HailoRT Python library ('hailo_platform') is required for live accelerator deployment ({last_err or 'Module not found'})."
    remediation = (
        "1. Install Hailo software suite with 'sudo apt install -y hailo-all python3-hailort' on your Raspberry Pi 5.\n"
        "2. Or run live deployment on Host CPU with:\n"
        "   python train_and_compile.py deploy --source 0 --camera-type picamera2 --cpu"
    )
    return None, cause, remediation


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

        hailo_mod, diag_cause, diag_rem = try_import_hailo_platform()
        if hailo_mod is None:
            raise DeploymentError(
                diag_cause or "HailoRT Python library ('hailo_platform') is required for live accelerator deployment.",
                code="E-DEPLOY-RT-001",
                remediation=diag_rem or "Run './scripts/bootstrap.sh' or 'sudo apt install -y python3-hailort' on your Raspberry Pi 5.",
            )

        self._hailo = hailo_mod
        self._target = None
        self._network_group = None
        self._activated_network = None
        self._infer_pipeline = None

        try:
            # 1. Initialize VDevice
            params = self._hailo.VDevice.create_params()
            self._target = self._hailo.VDevice(params)
            if hasattr(self._target, "__enter__"):
                self._target.__enter__()

            # 2. Load HEF
            self.hef = self._hailo.HEF(str(self.hef_path))

            # Retrieve stream metadata from HEF
            input_infos = []
            output_infos = []
            if hasattr(self.hef, "get_input_vstream_infos"):
                try:
                    input_infos = self.hef.get_input_vstream_infos()
                    output_infos = self.hef.get_output_vstream_infos()
                except Exception:
                    pass

            # 3. Configure PCIe Stream Interface
            configure_params = self._hailo.ConfigureParams.create_from_hef(
                self.hef, interface=self._hailo.HailoStreamInterface.PCIe
            )
            network_groups = self._target.configure(self.hef, configure_params)
            self._network_group = network_groups[0]
            network_group_params = self._network_group.create_params()

            # Fallback to network_group if HEF did not provide stream infos
            if not input_infos and hasattr(self._network_group, "get_input_vstream_infos"):
                try:
                    input_infos = self._network_group.get_input_vstream_infos()
                    output_infos = self._network_group.get_output_vstream_infos()
                except Exception:
                    pass

            self._input_name = input_infos[0].name if input_infos else "input"
            self._output_names = [info.name for info in output_infos]

            # 4. Configure Virtual Streams
            if hasattr(self._hailo.InputVStreamParams, "make"):
                input_vstreams_params = self._hailo.InputVStreamParams.make(
                    self._network_group, quantized=False, format_type=self._hailo.FormatType.UINT8
                )
                output_vstreams_params = self._hailo.OutputVStreamParams.make(
                    self._network_group, quantized=False, format_type=self._hailo.FormatType.FLOAT32
                )
            elif hasattr(self._hailo.InputVStreamParams, "make_from_network_group"):
                input_vstreams_params = self._hailo.InputVStreamParams.make_from_network_group(
                    self._network_group, quantized=False, format_type=self._hailo.FormatType.UINT8
                )
                output_vstreams_params = self._hailo.OutputVStreamParams.make_from_network_group(
                    self._network_group, quantized=False, format_type=self._hailo.FormatType.FLOAT32
                )
            else:
                input_vstreams_params = self._hailo.InputVStreamParams(self._network_group)
                output_vstreams_params = self._hailo.OutputVStreamParams(self._network_group)

            # 5. Activate network and instantiate InferVStreams for low-latency streaming
            self._activated_network = self._network_group.activate(network_group_params)
            self._activated_network.__enter__()

            self._infer_pipeline = self._hailo.InferVStreams(
                self._network_group, input_vstreams_params, output_vstreams_params
            )
            self._infer_pipeline.__enter__()

        except Exception as e:
            self.close()
            raise DeploymentError(
                f"Failed to initialize Hailo-8L VDevice and configure streams: {e}",
                code="E-DEPLOY-INIT-001",
                artifact=str(self.hef_path),
            ) from e

    @property
    def backend_name(self) -> str:
        return "Hailo-8L NPU (Raspberry Pi AI HAT+ PCIe)"

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
                if hasattr(self._target, "__exit__"):
                    self._target.__exit__(None, None, None)
                elif hasattr(self._target, "release"):
                    self._target.release()
            except Exception:
                pass
            self._target = None


class ONNXInferenceEngine(BaseInferenceEngine):
    """CPU/GPU inference engine using ONNX Runtime."""

    def __init__(self, onnx_path: Union[str, Path], threads: Optional[int] = None) -> None:
        self.onnx_path = Path(onnx_path).resolve()
        self.threads = threads
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

        from yolo_hailo_mlops.operations import get_multithreaded_ort_session_options
        sess_opts = get_multithreaded_ort_session_options(threads=self.threads)

        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def backend_name(self) -> str:
        return "Host CPU (ONNX Runtime)"

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

    def __init__(self, checkpoint_path: Union[str, Path], threads: Optional[int] = None) -> None:
        self.ckpt_path = Path(checkpoint_path).resolve()
        self.threads = threads
        if not self.ckpt_path.is_file():
            raise DeploymentError(
                f"PyTorch checkpoint file not found: {self.ckpt_path}",
                code="E-DEPLOY-PT-001",
                artifact=str(self.ckpt_path),
            )

        from yolo_hailo_mlops.operations import configure_cpu_multithreading
        configure_cpu_multithreading(self.threads)

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
        return "Host CPU (PyTorch)"

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
    threads: Optional[int] = None,
) -> BaseInferenceEngine:
    """Create inference engine according to target availability and backend preferences."""
    root = Path(root_artifacts) if root_artifacts else Path("artifacts")

    norm_backend = (backend or "auto").lower()
    if norm_backend in ("hailo-hat", "npu"):
        norm_backend = "hailo"

    # Resolve candidate model paths if not specified
    if not model_path:
        # Filter candidate search list based on requested compute target
        if norm_backend in ("cpu", "pytorch", "onnx"):
            candidates = [
                root / "latest" / "onnx" / "model.onnx",
                root / "onnx" / "model.onnx",
                root / "runs" / "latest" / "onnx" / "model.onnx",
                root / "latest" / "pytorch" / "best.pt",
                root / "pytorch" / "best.pt",
                root / "runs" / "latest" / "pytorch" / "best.pt",
                Path("yolo11n.pt"),
            ]
        elif norm_backend == "hailo":
            candidates = [
                root / "latest" / "hailo" / "model.hef",
                root / "hailo" / "model.hef",
                root / "runs" / "latest" / "hailo" / "model.hef",
            ]
        else:
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
            if norm_backend != "cpu":
                hef_runs = sorted(root.glob("runs/**/hailo/*.hef"), key=lambda p: p.stat().st_mtime, reverse=True)
                if hef_runs:
                    resolved_path = hef_runs[0]
            if not resolved_path:
                onnx_runs = sorted(root.glob("runs/**/onnx/*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
                if onnx_runs:
                    resolved_path = onnx_runs[0]
            if not resolved_path:
                pt_runs = sorted(root.glob("runs/**/pytorch/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if pt_runs:
                    resolved_path = pt_runs[0]

        if not resolved_path:
            raise DeploymentError(
                f"No suitable model artifact found for target '{backend}'. Specify '--model <path>' explicitly.",
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
    if norm_backend == "hailo":
        if ext != ".hef":
            # Search for compiled .hef in canonical artifact locations
            hef_alt = None
            for cand in [root / "latest" / "hailo" / "model.hef", root / "hailo" / "model.hef", root / "runs" / "latest" / "hailo" / "model.hef"]:
                if cand.is_file():
                    hef_alt = cand
                    break
            if hef_alt:
                model_p = hef_alt
                ext = ".hef"
            else:
                raise DeploymentError(
                    f"Hailo Hat NPU backend requires a compiled .hef binary, got '{model_p.name}'",
                    code="E-DEPLOY-BCK-001",
                    artifact=str(model_p),
                    remediation="Compile model to HEF with 'python train_and_compile.py compile' or use '--target cpu'.",
                )
        return HailoInferenceEngine(model_p)

    elif norm_backend == "cpu":
        if ext == ".hef":
            # Locate corresponding .onnx or .pt for CPU inference
            alt_cpu = None
            for cand in [root / "latest" / "onnx" / "model.onnx", root / "onnx" / "model.onnx", root / "latest" / "pytorch" / "best.pt", root / "pytorch" / "best.pt", Path("yolo11n.pt")]:
                if cand.is_file():
                    alt_cpu = cand
                    break
            if alt_cpu:
                model_p = alt_cpu
                ext = model_p.suffix.lower()
            else:
                raise DeploymentError(
                    "CPU deployment requires an .onnx or .pt model, but only .hef was provided.",
                    code="E-DEPLOY-BCK-003",
                    remediation="Export to ONNX ('python train_and_compile.py export') or specify '--model yolo11n.pt'.",
                )
        def _make_onnx(p):
            return ONNXInferenceEngine(p, threads=threads) if threads is not None else ONNXInferenceEngine(p)

        def _make_pytorch(p):
            return PyTorchInferenceEngine(p, threads=threads) if threads is not None else PyTorchInferenceEngine(p)

        if ext == ".onnx":
            return _make_onnx(model_p)
        elif ext == ".pt":
            return _make_pytorch(model_p)
        else:
            return _make_pytorch(model_p)

    elif norm_backend == "onnx":
        if ext != ".onnx":
            onnx_alt = root / "onnx" / "model.onnx"
            if onnx_alt.is_file():
                model_p = onnx_alt
            else:
                raise DeploymentError(
                    f"ONNX backend requires an .onnx model, got '{model_p.name}'",
                    code="E-DEPLOY-BCK-002",
                )
        return ONNXInferenceEngine(model_p, threads=threads) if threads is not None else ONNXInferenceEngine(model_p)

    elif norm_backend == "pytorch":
        return PyTorchInferenceEngine(model_p, threads=threads) if threads is not None else PyTorchInferenceEngine(model_p)

    # Auto resolution
    if ext == ".hef":
        env = detect_hailo_environment()
        if env.has_hardware:
            try:
                return HailoInferenceEngine(model_p)
            except DeploymentError as e:
                if norm_backend != "hailo":
                    print(f"[NOTICE] Hailo accelerator engine initialization note ({e.message}).")
                    # Fallback to ONNX or PyTorch if available
                    onnx_cand = root / "onnx" / "model.onnx"
                    if onnx_cand.is_file():
                        print(f"[INFO] Gracefully utilizing ONNX runtime engine ({onnx_cand})...")
                        return ONNXInferenceEngine(onnx_cand, threads=threads) if threads is not None else ONNXInferenceEngine(onnx_cand)
                    pt_cand = root / "pytorch" / "best.pt"
                    if pt_cand.is_file():
                        print(f"[INFO] Gracefully utilizing PyTorch CPU engine ({pt_cand})...")
                        return PyTorchInferenceEngine(pt_cand, threads=threads) if threads is not None else PyTorchInferenceEngine(pt_cand)
                    pt_base = Path("yolo11n.pt")
                    if pt_base.is_file():
                        print(f"[INFO] Gracefully utilizing base PyTorch engine ({pt_base})...")
                        return PyTorchInferenceEngine(pt_base, threads=threads) if threads is not None else PyTorchInferenceEngine(pt_base)
                raise
        else:
            # Fallback to ONNX if available
            onnx_cand = root / "onnx" / "model.onnx"
            if onnx_cand.is_file():
                return ONNXInferenceEngine(onnx_cand, threads=threads) if threads is not None else ONNXInferenceEngine(onnx_cand)
            pt_cand = root / "pytorch" / "best.pt"
            if pt_cand.is_file():
                return PyTorchInferenceEngine(pt_cand, threads=threads) if threads is not None else PyTorchInferenceEngine(pt_cand)
            # Re-raise informative error from Hailo engine
            return HailoInferenceEngine(model_p)

    elif ext == ".onnx":
        return ONNXInferenceEngine(model_p, threads=threads) if threads is not None else ONNXInferenceEngine(model_p)
    elif ext == ".pt":
        return PyTorchInferenceEngine(model_p, threads=threads) if threads is not None else PyTorchInferenceEngine(model_p)
    else:
        raise DeploymentError(
            f"Unsupported model extension '{ext}'. Expected .hef, .onnx, or .pt.",
            code="E-DEPLOY-MDL-003",
            artifact=str(model_p),
        )
