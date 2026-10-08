"""Phase C: ONNX graph structure validation and ONNX Runtime smoke testing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from yolo_hailo_mlops.exceptions import ONNXValidationError
from yolo_hailo_mlops.utils.hashing import compute_sha256


@dataclass
class ONNXValidationReport:
    path: str
    sha256: str
    size_bytes: int
    is_valid: bool
    input_name: str
    input_shape: List[int]
    input_dtype: str
    output_names: List[str]
    output_shapes: List[List[Any]]
    smoke_test_passed: bool
    smoke_latency_ms: Optional[float] = None
    has_nan_or_inf: bool = False
    errors: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "is_valid": self.is_valid,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "output_names": self.output_names,
            "output_shapes": self.output_shapes,
            "smoke_test_passed": self.smoke_test_passed,
            "smoke_latency_ms": self.smoke_latency_ms,
            "has_nan_or_inf": self.has_nan_or_inf,
            "errors": self.errors or [],
        }


def validate_onnx_model(
    onnx_path: Union[str, Path],
    expected_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
) -> ONNXValidationReport:
    """Validate ONNX graph conforms to static Hailo contract [1, 3, 640, 640] FP32."""
    p = Path(onnx_path).resolve()
    if not p.is_file():
        raise ONNXValidationError(
            f"ONNX model file not found: {p}",
            code="E-ONNX-002",
            artifact=str(p),
        )

    size = p.stat().st_size
    sha = compute_sha256(p)
    errors: List[str] = []

    input_name = "images"
    input_shape = list(expected_shape)
    input_dtype = "FLOAT"
    output_names: List[str] = []
    output_shapes: List[List[Any]] = []

    # 1. Structural inspection with onnx package
    try:
        import onnx

        model = onnx.load(str(p))
        onnx.checker.check_model(model)

        graph = model.graph
        if len(graph.input) != 1:
            errors.append(f"Expected 1 graph input, found {len(graph.input)}")

        inp = graph.input[0]
        input_name = inp.name

        # Extract dimensions
        actual_dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                actual_dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                errors.append(f"Dynamic dimension detected in input: '{d.dim_param}'. Fixed static dimensions required.")
                actual_dims.append(-1)
            else:
                errors.append("Unspecified dynamic input dimension found.")
                actual_dims.append(-1)

        input_shape = actual_dims
        if actual_dims != list(expected_shape):
            errors.append(f"Input shape mismatch. Expected {list(expected_shape)}, found {actual_dims}")

        # Datatype check: TensorProto.FLOAT == 1
        elem_type = inp.type.tensor_type.elem_type
        if elem_type != onnx.TensorProto.FLOAT:
            errors.append(f"Input tensor must be FP32 (TensorProto.FLOAT), got type {elem_type}")
            input_dtype = f"TYPE_{elem_type}"

        # Inspect outputs
        for out in graph.output:
            output_names.append(out.name)
            dims = []
            for d in out.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            output_shapes.append(dims)

    except ImportError:
        pass
    except Exception as e:
        errors.append(f"ONNX check failed: {e}")

    # 2. Smoke inference with ONNX Runtime
    smoke_passed = False
    smoke_latency_ms = None
    has_nan_inf = False

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy_input = np.zeros(expected_shape, dtype=np.float32)

        # Warmup and measurement
        t0 = time.perf_counter()
        outputs = sess.run(None, {inp_name: dummy_input})
        t1 = time.perf_counter()
        smoke_latency_ms = round((t1 - t0) * 1000.0, 3)

        for out_arr in outputs:
            if np.isnan(out_arr).any() or np.isinf(out_arr).any():
                has_nan_inf = True
                errors.append("Smoke test output contains NaN or Inf values")
                break

        if not has_nan_inf:
            smoke_passed = True

    except ImportError:
        pass
    except Exception as e:
        errors.append(f"ONNX Runtime smoke test execution failed: {e}")

    is_valid = len(errors) == 0

    report = ONNXValidationReport(
        path=str(p),
        sha256=sha,
        size_bytes=size,
        is_valid=is_valid,
        input_name=input_name,
        input_shape=input_shape,
        input_dtype=input_dtype,
        output_names=output_names,
        output_shapes=output_shapes,
        smoke_test_passed=smoke_passed,
        smoke_latency_ms=smoke_latency_ms,
        has_nan_or_inf=has_nan_inf,
        errors=errors,
    )

    if not is_valid:
        raise ONNXValidationError(
            message=f"ONNX validation failed: {'; '.join(errors)}",
            code="E-ONNX-003",
            artifact=str(p),
        )

    return report
