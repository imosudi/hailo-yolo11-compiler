# Documentation Hub & Sitemap

Welcome to the comprehensive technical documentation for **`hailo-yolo11-compiler`**. This directory contains user manuals, architectural specifications, deployment runbooks, and operational references for compiling and deploying Ultralytics YOLO11 models to the **Hailo-8L AI HAT+ on Raspberry Pi 5**.

---

## Documentation Structure

```text
docs/
├── README.md                      # Documentation hub & sitemap (this document)
│
├── User Guides & Manuals:
│   ├── user_guide.md              # Complete end-to-end user guide and CLI reference
│   ├── dataset_guide.md           # Dataset preparation, YOLO format, and pre-flight validation manual
│   ├── configuration_guide.md     # Configuration system, YAML schema, and environment variables
│   ├── hardware_setup_guide.md    # Raspberry Pi 5 & Hailo-8L AI HAT+ physical assembly & driver guide
│   └── benchmarking_and_evaluation.md # Accuracy gates, telemetry provenance, and latency benchmarks
│
├── Architectural & Technical Specifications:
│   ├── architecture.md            # System architecture, state machine, and execution boundaries
│   ├── pipeline.md                # Phase-by-phase execution lifecycle and data flow
│   ├── hailo.md                   # Hailo Dataflow Compiler (DFC) & HailoRT integration details
│   ├── calibration.md             # Deterministic INT8 calibration & preprocessing contract
│   ├── reproducibility.md         # Experiment reproducibility, manifests, and atomic storage
│   └── troubleshooting.md         # Diagnostic error taxonomy (E-*) and remediation procedures
```

---

## Role-Based Reading Paths

Depending on your engineering role, follow these recommended reading sequences:

### 1. Machine Learning & Vision Engineers
*Goal: Prepare custom datasets, train YOLO11 models, and export static FP32 ONNX graphs.*
1. [User Guide: Quickstart & Workflows](user_guide.md)
2. [Dataset Guide: Formatting & Pre-flight Validation](dataset_guide.md)
3. [Configuration Guide: Training & Export Parameters](configuration_guide.md)
4. [Calibration Guide: Deterministic INT8 Sampling](calibration.md)

### 2. Edge AI & Compiler Engineers
*Goal: Translate ONNX models into Hailo Archives (HAR), perform INT8 PTQ, and compile HEF binaries.*
1. [Architecture Specification: Execution Boundaries](architecture.md)
2. [Hailo Compilation Guide: DFC & Model Scripts](hailo.md)
3. [Calibration Guide: Preprocessing Contract](calibration.md)
4. [Troubleshooting Guide: Compilation & Routing Remediation](troubleshooting.md)

### 3. Embedded Systems & Deployment Engineers
*Goal: Assemble Raspberry Pi 5 hardware, install HailoRT drivers, and run physical benchmarks.*
1. [Hardware Setup Guide: Raspberry Pi 5 & AI HAT+](hardware_setup_guide.md)
2. [Benchmarking & Evaluation Guide: Latency, FPS & Telemetry](benchmarking_and_evaluation.md)
3. [Reproducibility Guide: Cryptographic Manifests](reproducibility.md)
4. [Troubleshooting Guide: Driver & Device Node Diagnostics](troubleshooting.md)

---

## Quick Reference to Primary Commands

| Goal | CLI Command | Documented In |
| :--- | :--- | :--- |
| **System Diagnostics** | `python train_and_compile.py doctor` | [user_guide.md](user_guide.md#command-reference) |
| **Inspect Execution Plan** | `python train_and_compile.py --dry-run all` | [user_guide.md](user_guide.md#dry-run-planning) |
| **Validate Dataset** | `python train_and_compile.py dataset` | [dataset_guide.md](dataset_guide.md) |
| **Train YOLO11** | `python train_and_compile.py train --epochs 100` | [user_guide.md](user_guide.md#phase-b-training) |
| **Export Static ONNX** | `python train_and_compile.py export` | [user_guide.md](user_guide.md#phase-c-onnx-export) |
| **Extract Calibration Set** | `python train_and_compile.py calibrate --count 200` | [calibration.md](calibration.md) |
| **Compile Hailo-8L HEF** | `python train_and_compile.py compile` | [hailo.md](hailo.md) |
| **Validate Accuracy & Speed** | `python train_and_compile.py validate` | [benchmarking_and_evaluation.md](benchmarking_and_evaluation.md) |
| **End-to-End Orchestration** | `python train_and_compile.py all` | [user_guide.md](user_guide.md#end-to-end-execution) |
| **Execute Test Suite** | `pytest -v` | [architecture.md](architecture.md#testing-strategy) |

---

## Conventions & Standards

All documentation in this repository is authored in **standard British English** (e.g., *artefact*, *quantisation*, *optimisation*, *normalisation*, *colour*, *centre*, *behaviour*). Technical parameter names that originate from third-party tools or external libraries (such as Hailo script commands or PyTorch function arguments) retain their required syntactic identifiers.
