#!/usr/bin/env python3
"""train_and_compile.py - Principal orchestration entry point for hailo-yolo11-compiler.

Provides unified interface for dataset pre-flight validation, YOLO11 training,
FP32 ONNX export, deterministic INT8 calibration, Hailo DFC compilation,
accuracy gates, and Raspberry Pi 5 Hailo-8L runtime execution.
"""

import sys
from pathlib import Path

# Ensure src/ is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo_hailo_mlops.cli import main

if __name__ == "__main__":
    main()
