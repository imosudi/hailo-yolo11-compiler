#!/usr/bin/env python3
"""operations.py - Hardware CPU multithreading and parallel execution engine.

Provides unified management for CPU parallelism across all compute operations:
  - PyTorch CPU operations (intra-op and inter-op threads)
  - ONNX Runtime CPU operations (intra-op, inter-op, ORT_PARALLEL execution mode)
  - OpenCV CPU operations (multithreaded IP / decode pipeline)
  - BLAS / OpenMP libraries (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS)
  - Multithreaded image preprocessing and dataset calibration worker pools
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def get_cpu_capacity() -> int:
    """Return available hardware CPU execution capacity (accounting for affinity/cgroups)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                return affinity
        except Exception:
            pass

    cores = os.cpu_count()
    return max(1, cores if cores is not None else 4)


def get_optimal_cpu_threads(requested: Optional[int] = None) -> int:
    """Determine optimal CPU thread count based on hardware topology and explicit overrides."""
    if requested is not None and requested > 0:
        return requested

    # Check environment variable overrides
    for env_var in ("YOLO_HAILO_CPU_THREADS", "OMP_NUM_THREADS", "TORCH_NUM_THREADS"):
        val = os.environ.get(env_var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)

    return get_cpu_capacity()


def configure_cpu_multithreading(threads: Optional[int] = None) -> Dict[str, Any]:
    """Configure intra-op and inter-op multithreading across CPU frameworks and math runtimes."""
    n_threads = get_optimal_cpu_threads(threads)

    # 1. Environment variables for native math libraries (OpenMP, MKL, BLAS)
    env_vars = {
        "OMP_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
        "MKL_NUM_THREADS": str(n_threads),
        "VECLIB_MAXIMUM_THREADS": str(n_threads),
        "NUMEXPR_NUM_THREADS": str(n_threads),
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    status: Dict[str, Any] = {
        "threads": n_threads,
        "cpu_capacity": get_cpu_capacity(),
        "env": env_vars,
        "pytorch": None,
        "opencv": None,
        "onnxruntime": None,
    }

    # 2. PyTorch intra-op and inter-op threads
    try:
        import torch
        torch.set_num_threads(n_threads)
        interop = max(1, min(4, n_threads // 2))
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            # set_num_interop_threads can only be called once before execution begins
            pass
        status["pytorch"] = {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }
    except Exception as e:
        status["pytorch"] = f"Unavailable ({e})"

    # 3. OpenCV multithreading
    try:
        import cv2
        cv2.setNumThreads(n_threads)
        if hasattr(cv2, "setUseOptimized"):
            cv2.setUseOptimized(True)
        status["opencv"] = {
            "num_threads": cv2.getNumThreads(),
            "use_optimized": cv2.useOptimized() if hasattr(cv2, "useOptimized") else True,
        }
    except Exception as e:
        status["opencv"] = f"Unavailable ({e})"

    # 4. ONNX Runtime availability status
    try:
        import onnxruntime as ort
        status["onnxruntime"] = {
            "available": True,
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
    except Exception as e:
        status["onnxruntime"] = f"Unavailable ({e})"

    return status


def get_multithreaded_ort_session_options(threads: Optional[int] = None) -> Any:
    """Create and configure an onnxruntime.SessionOptions instance for parallel CPU execution."""
    try:
        import onnxruntime as ort
    except (ImportError, ModuleNotFoundError):
        return None

    n_threads = get_optimal_cpu_threads(threads)
    interop = max(1, min(4, n_threads // 2))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = interop
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return opts


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
) -> List[R]:
    """Execute func across items using a ThreadPoolExecutor scaled to CPU cores."""
    items_list = list(items)
    if not items_list:
        return []

    workers = get_optimal_cpu_threads(max_workers)
    if len(items_list) == 1 or workers <= 1:
        return [func(item) for item in items_list]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items_list))


def run_cpu_benchmark(threads: Optional[int] = None, iterations: int = 100) -> Dict[str, Any]:
    """Run synthetic CPU benchmark comparing single-threaded vs multithreaded performance."""
    import numpy as np

    n_threads = get_optimal_cpu_threads(threads)
    configure_cpu_multithreading(n_threads)

    matrix_size = 512
    a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = np.dot(a, b)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np.dot(a, b)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0.0

    return {
        "threads_configured": n_threads,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 4),
        "matmul_512x512_per_sec": round(ops_per_sec, 2),
    }


def main() -> None:
    """CLI diagnostics and benchmarking for CPU operations."""
    parser = argparse.ArgumentParser(description="CPU Multithreading & Operations Diagnostics")
    parser.add_argument("--status", action="store_true", help="Print CPU thread configuration status")
    parser.add_argument("--threads", type=int, default=None, help="Configure explicit thread count")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU matrix operations benchmark")
    args = parser.parse_args()

    cfg = configure_cpu_multithreading(args.threads)

    print("================================================================================")
    print(" hailo-yolo11-compiler CPU Multithreading & Operations Status")
    print("================================================================================")
    print(f"  Host Architecture:   {platform.machine()} ({platform.system()})")
    print(f"  CPU Hardware Cores:  {cfg['cpu_capacity']}")
    print(f"  Active CPU Threads:  {cfg['threads']}")
    print("--------------------------------------------------------------------------------")
    print(f"  PyTorch Threads:     {cfg['pytorch']}")
    print(f"  OpenCV Threads:      {cfg['opencv']}")
    print(f"  ONNX Runtime:        {cfg['onnxruntime']}")
    print("  Environment Vars:")
    for k, v in cfg["env"].items():
        print(f"    {k:<24} = {v}")
    print("================================================================================")

    if args.benchmark:
        print("\nRunning CPU Multithreaded Operations Benchmark...")
        bench = run_cpu_benchmark(threads=args.threads)
        print(f"  Configured Threads:  {bench['threads_configured']}")
        print(f"  Total Duration:      {bench['elapsed_seconds']}s ({bench['iterations']} iterations)")
        print(f"  Throughput:          {bench['matmul_512x512_per_sec']} matmul(512x512) ops/sec")
        print("================================================================================")


if __name__ == "__main__":
    main()
