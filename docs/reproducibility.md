# Experiment Reproducibility & Result Integrity Manual

This manual provides an in-depth reference for cryptographic provenance, deterministic seed propagation, immutable experiment ledgers, and atomic filesystem safety within the **`hailo-yolo11-compiler`** pipeline.

---

## Table of Contents

1. [Principles of Defensible Edge MLOps](#1-principles-of-defensible-edge-mlops)
2. [Deterministic Random Seed Propagation](#2-deterministic-random-seed-propagation)
3. [Cryptographic SHA-256 Provenance](#3-cryptographic-sha-256-provenance)
4. [The Immutable Experiment Ledger (`run_manifest.json`)](#4-the-immutable-experiment-ledger-run_manifestjson)
5. [Atomic Filesystem Storage Contract](#5-atomic-filesystem-storage-contract)
6. [Telemetry Provenance Taxonomy](#6-telemetry-provenance-taxonomy)
7. [Detecting Artefact Corruption & Tampering](#7-detecting-artefact-corruption--tampering)

---

## 1. Principles of Defensible Edge MLOps

In edge AI systems, unverified performance claims and unreproducible compilation runs lead to deployment failures and silent regressions. `hailo-yolo11-compiler` enforces three fundamental research tenets:

1. **Defensible Chain of Provenance**: Every output file records its upstream source files, producing phase, execution duration, system commit hash, and SHA-256 cryptographic digest.
2. **Zero Hidden Fallbacks**: The pipeline never silently executes Hailo operations on CPU or GPU. If physical Hailo hardware is missing during runtime validation, the state machine records `HARDWARE_VALIDATION = NOT_EXECUTED`.
3. **Zero Fabricated Telemetry**: Hardware metrics that cannot be directly queried through physical sensors are recorded as `UNAVAILABLE` rather than faked with zero values.

---

## 2. Deterministic Random Seed Propagation

Randomness in neural network training, data splitting, and calibration sampling can introduce non-deterministic deviations. The pipeline accepts a master random seed (default `seed: 42`) and propagates it across all execution subsystems:

```python
# Automatic seed propagation in yolo_hailo_mlops
import os, random, numpy as np, torch

def enforce_determinism(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Deterministic Calibration Selection
When sampling 200 calibration images from the dataset, the selector initializes a pseudo-random permutation with `project.seed`. Re-running the calibration phase on the same dataset produces an identical sequence of calibration images and an identical `calib_data.npy` tensor.

---

## 3. Cryptographic SHA-256 Provenance

Every model weight file (`.pt`), ONNX graph (`.onnx`), intermediate archive (`.har`), calibration tensor (`.npy`), and compiled bitstream (`.hef`) is hashed immediately upon generation using chunked SHA-256:

```python
import hashlib
from pathlib import Path

def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
```

These checksums are recorded in `run_manifest.json`. If an artefact is modified or corrupted after generation, subsequent stages detect the mismatch and raise `E-ART-003`.

---

## 4. The Immutable Experiment Ledger (`run_manifest.json`)

At the conclusion of each pipeline execution, an immutable ledger is written to `artifacts/runs/<run_id>/run_manifest.json`:

```json
{
  "run_id": "20261008-201422-a31f",
  "status": "SUCCESS",
  "started_at": "2026-10-08T20:14:22.124510+00:00",
  "completed_at": "2026-10-08T20:18:45.312940+00:00",
  "duration_seconds": 263.19,
  "execution_mode": "portable",
  "config": {
    "project": { "name": "yolo11-hailo", "seed": 42 },
    "model": { "source": "yolo11n.pt", "variant": "n" },
    "hailo": { "target": "hailo8l" }
  },
  "environment": {
    "os": "Linux 6.6.20+rpt-rpi-2712",
    "architecture": "aarch64",
    "python_version": "3.11.2",
    "git": {
      "commit": "7a7d2b5e13d9a1f28b490a",
      "branch": "main",
      "dirty": false
    }
  },
  "phases": {
    "dataset_validation": { "state": "SUCCESS", "duration_seconds": 1.42 },
    "training": { "state": "SKIPPED", "duration_seconds": 0.05 },
    "onnx_export": { "state": "SUCCESS", "duration_seconds": 12.18 },
    "calibration": { "state": "SUCCESS", "duration_seconds": 4.81 },
    "hailo_compile": { "state": "SUCCESS", "duration_seconds": 142.30 },
    "accuracy_validation": { "state": "SUCCESS", "duration_seconds": 18.52 }
  },
  "artifacts": {
    "model_onnx": {
      "path": "/workspace/artifacts/runs/20261008-201422-a31f/onnx/model.onnx",
      "type": "onnx_model",
      "size_bytes": 11245890,
      "sha256": "4b281f692ad2459ab12...e58",
      "created_at": "2026-10-08T20:15:02+00:00",
      "producer_phase": "onnx_export",
      "status": "valid"
    },
    "model_hef": {
      "path": "/workspace/artifacts/runs/20261008-201422-a31f/hailo/model.hef",
      "type": "hef_binary",
      "size_bytes": 4519820,
      "sha256": "8f3b14092bba4...19c2",
      "created_at": "2026-10-08T20:17:34+00:00",
      "producer_phase": "hailo_compile",
      "status": "valid"
    }
  },
  "metrics": {
    "pytorch_map50": 0.824,
    "hailo_map50": 0.816,
    "delta_map50": 0.008,
    "accuracy_gate_passed": true
  },
  "telemetry": [
    { "name": "median_latency_ms", "value": 11.8, "provenance": "MEASURED" },
    { "name": "throughput_fps", "value": 84.7, "provenance": "DERIVED" },
    { "name": "soc_temperature_c", "value": 51.3, "provenance": "MEASURED" }
  ]
}
```

---

## 5. Atomic Filesystem Storage Contract

To prevent partially written or truncated files caused by out-of-disk errors, process interrupts, or system power loss, all artefacts are committed atomically:

```text
Staging:     artifacts/runs/<run_id>/hailo/.tmp_model_hef_xyz123
Flush:       os.fsync(fd)
Atomic Move: Path.replace(target_path)
Canonical:   os.symlink(relative_target, "artifacts/hailo/model.hef")
```

```python
import os, tempfile
from pathlib import Path

def atomic_write(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_path.parent
    with tempfile.NamedTemporaryFile("wb", dir=temp_dir, delete=False, prefix=f".tmp_{target_path.name}_") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        temp_name = f.name
    Path(temp_name).replace(target_path)
```

---

## 6. Telemetry Provenance Taxonomy

Every reported benchmark value is accompanied by one of four provenance classifications:

| Classification | Meaning | Safeguard Against |
| :--- | :--- | :--- |
| **`MEASURED`** | Directly observed from physical sensors or system calls. | Prevents synthetic or fabricated timing values. |
| **`DERIVED`** | Calculated mathematically from measured primitives. | Ensures transparent arithmetic derivations. |
| **`CONFIGURED`** | Fixed parameter specified in user configuration. | Distinguishes fixed parameters from observed data. |
| **`UNAVAILABLE`** | Requested metric cannot be provided by host/hardware. | Eliminates misleading zeros for absent sensors. |

---

## 7. Detecting Artefact Corruption & Tampering

If any file in `artifacts/runs/<run_id>/` is modified post-execution:
1. Re-running validation or inspection reads the recorded `sha256` from `run_manifest.json`.
2. Computes the real-time SHA-256 of the file on disk.
3. If hashes differ, execution halts with:
   ```text
   [E-ART-003] artifacts: Cryptographic SHA-256 checksum mismatch for model.hef
   Expected: 8f3b14092bba4...19c2
   Observed: a591ce02b0c41...98ff
   Remediation: Re-run compilation phase with --force to regenerate verified artefacts.
   ```
