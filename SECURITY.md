# Security Policy

## Supported Versions

Only the latest release of `hailo-yolo11-compiler` receives active security updates and vulnerability patches.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Edge AI Threat Model & Security Posture

Deploying computer vision models on edge platforms such as the Raspberry Pi 5 with Hailo-8L accelerators introduces specific security considerations:

### 1. Model Artifact Integrity
- **Checksum Verification**: All models (`best.pt`, `model.onnx`, `model.har`, `model.hef`) are SHA-256 hashed and recorded in `run_manifest.json`.
- **Tamper Evidence**: The compiler verifies SHA-256 signatures prior to loading weights or deploying HEF files to hardware to prevent model substitution or backdoor injection attacks.

### 2. Subprocess & Command Execution
- The pipeline executes external utilities (such as Hailo CLI commands or system telemetry scripts) strictly via structured argument lists (`subprocess.run(["cmd", "arg"])`).
- Shell expansion (`shell=True`) is strictly prohibited across the entire codebase to prevent arbitrary command injection.
- Secret and sensitive environment variables are automatically redacted from execution logs.

### 3. Supply Chain Security
- All dependencies are separated into distinct requirement layers (`core`, `training`, `onnx`, `hailo`, `dev`) and pinned to prevent unverified upstream dependency changes during automated builds.
- Serialization utilizes standard ONNX formats and PyTorch safe loading protocols.

### 4. Edge Host Hardening
- Do not run pipeline scripts as `root` unless hardware device nodes (`/dev/hailo0`) strictly require group access. We recommend adding the runtime user to the `hailo` or `plugdev` user group:
  ```bash
  sudo usermod -aG hailo $USER
  ```

## Reporting a Vulnerability

If you discover a security vulnerability in this project:

1. **Do not** disclose the vulnerability publicly in an open GitHub issue or pull request.
2. Send a detailed report to `imosudi@gmail.com` with:
   - A description of the issue.
   - Proof of concept or reproduction steps.
   - Affected system components or hardware targets.
   - Any proposed remediation.
3. You will receive an acknowledgment within 48 hours, followed by updates on triage and patching timelines.
