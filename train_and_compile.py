#!/usr/bin/env python3
"""train_and_compile.py - Principal orchestration entry point for hailo-yolo11-compiler.

Provides unified interface for dataset pre-flight validation, YOLO11 training,
FP32 ONNX export, deterministic INT8 calibration, Hailo DFC compilation,
accuracy gates, and Raspberry Pi 5 Hailo-8L runtime execution.
"""

import sys
from pathlib import Path

# Ensure root and src/ are on Python search path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Initialize optimized CPU multithreading for all operations
try:
    from operations import configure_cpu_multithreading
    configure_cpu_multithreading()
except Exception:
    pass

from yolo_hailo_mlops.cli import main

if __name__ == "__main__":
    main()
