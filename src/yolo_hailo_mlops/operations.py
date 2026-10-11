"""yolo_hailo_mlops.operations - Hardware CPU multithreading and parallel execution engine."""

from __future__ import annotations

import sys
from pathlib import Path

# Load core operations functions from operations.py
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from operations import (
    configure_cpu_multithreading,
    get_cpu_capacity,
    get_multithreaded_ort_session_options,
    get_optimal_cpu_threads,
    parallel_map,
    run_cpu_benchmark,
)

__all__ = [
    "configure_cpu_multithreading",
    "get_cpu_capacity",
    "get_multithreaded_ort_session_options",
    "get_optimal_cpu_threads",
    "parallel_map",
    "run_cpu_benchmark",
]
