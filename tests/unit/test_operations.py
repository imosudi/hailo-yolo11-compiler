"""Unit tests for operations.py and CPU multithreading management."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from yolo_hailo_mlops.operations import (
    configure_cpu_multithreading,
    get_cpu_capacity,
    get_multithreaded_ort_session_options,
    get_optimal_cpu_threads,
    parallel_map,
)


def test_get_cpu_capacity() -> None:
    """Verify get_cpu_capacity returns positive integer count."""
    capacity = get_cpu_capacity()
    assert isinstance(capacity, int)
    assert capacity >= 1


def test_get_optimal_cpu_threads_explicit() -> None:
    """Verify explicit requested thread count is respected."""
    assert get_optimal_cpu_threads(2) == 2
    assert get_optimal_cpu_threads(8) == 8


def test_get_optimal_cpu_threads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables override default thread count."""
    monkeypatch.setenv("YOLO_HAILO_CPU_THREADS", "6")
    assert get_optimal_cpu_threads(None) == 6

    monkeypatch.delenv("YOLO_HAILO_CPU_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert get_optimal_cpu_threads(None) == 5


def test_configure_cpu_multithreading() -> None:
    """Verify CPU multithreading configuration across environment, PyTorch, and OpenCV."""
    status = configure_cpu_multithreading(threads=4)
    assert status["threads"] == 4
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
    assert os.environ.get("MKL_NUM_THREADS") == "4"

    try:
        import torch
        assert torch.get_num_threads() == 4
    except ImportError:
        pass

    try:
        import cv2
        assert cv2.getNumThreads() == 4
    except ImportError:
        pass


def test_get_multithreaded_ort_session_options() -> None:
    """Verify ONNX Runtime session options configured for parallel CPU execution."""
    opts = get_multithreaded_ort_session_options(threads=4)
    if opts is not None:
        import onnxruntime as ort
        assert opts.intra_op_num_threads == 4
        assert opts.inter_op_num_threads >= 1
        assert opts.execution_mode == ort.ExecutionMode.ORT_PARALLEL


def test_parallel_map() -> None:
    """Verify parallel_map correctly processes items across threads."""
    items = [1, 2, 3, 4, 5]
    results = parallel_map(lambda x: x * 2, items, max_workers=2)
    assert results == [2, 4, 6, 8, 10]


def test_parallel_map_empty() -> None:
    """Verify parallel_map handles empty collections cleanly."""
    assert parallel_map(lambda x: x * 2, []) == []
