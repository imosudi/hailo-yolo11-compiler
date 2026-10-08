# Experiment Reproducibility & Result Integrity

## Principles of Result Integrity

`hailo-yolo11-compiler` treats ML compiler operations and hardware benchmarking with research-grade rigor:

1. **Defensible Chain of Provenance**: Every artifact records its source files, generating phase, runtime duration, git commit, and SHA-256 cryptographic hash.
2. **No Hidden Fallbacks**: The pipeline will never silently emulate missing Hailo hardware on CPU or GPU. Missing physical devices are explicitly marked `NOT_EXECUTED`.
3. **No Fabricated Telemetry**: Hardware metrics that cannot be directly measured are classified as `UNAVAILABLE` with an explicit reason string.

## Telemetry Classification Taxonomy

Every captured metric is tagged with one of four classifications:

| Classification | Meaning | Example |
| :--- | :--- | :--- |
| `MEASURED` | Directly observed from physical sensors or system calls | `cpu_utilization`, `soc_temperature` |
| `DERIVED` | Statistically calculated from measured primitives | `throughput_fps`, `median_latency_ms` |
| `CONFIGURED` | Fixed parameter specified in configuration | `warmup_iterations`, `imgsz` |
| `UNAVAILABLE` | Metric requested but hardware/driver cannot expose it | `hailo_power: null` (Sensor missing) |

## The Experiment Manifest (`run_manifest.json`)

At the conclusion of each pipeline run, an immutable JSON ledger is generated:

```json
{
  "run_id": "20261008-201422-a31f",
  "status": "SUCCESS",
  "started_at": "2026-10-08T20:14:22.124510+00:00",
  "completed_at": "2026-10-08T20:18:45.312940+00:00",
  "duration_seconds": 263.19,
  "execution_mode": "portable",
  "config": { ... },
  "environment": {
    "os": "Linux",
    "architecture": "x86_64",
    "python_version": "3.12.3",
    "dependencies": { ... },
    "git": {
      "commit": "766f4a7e127a2da4eaebcf226332d6465da0c989",
      "branch": "main",
      "dirty": false
    }
  },
  "phases": { ... },
  "artifacts": {
    "model_onnx": {
      "path": "/workspace/artifacts/runs/20261008-201422-a31f/onnx/model.onnx",
      "type": "onnx_model",
      "size_bytes": 11245890,
      "sha256": "4b281f692ad2...e58",
      "created_at": "2026-10-08T20:15:02+00:00",
      "producer_phase": "onnx_export",
      "status": "valid"
    }
  },
  "metrics": { ... },
  "telemetry": [ ... ]
}
```

## Atomic Artifact Storage

To prevent partially written, corrupted files resulting from process interruptions or out-of-disk conditions:
- All artifacts are written to temporary files on the same filesystem (`.tmp_<name>_`).
- Synchronized to persistent media via `os.fsync()`.
- Atomically renamed to their final path via `Path.replace()`.
