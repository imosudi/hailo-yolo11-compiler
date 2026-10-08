"""Subprocess execution wrapper with strict safety, secret redaction, and error reporting."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

from yolo_hailo_mlops.exceptions import YoloHailoError

SENSITIVE_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASS", "CREDENTIAL", "AUTH")


@dataclass
class CommandResult:
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def redact_command(cmd: Sequence[str]) -> List[str]:
    """Redact potentially sensitive argument values in logged commands."""
    redacted = []
    redact_next = False
    for arg in cmd:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        if any(pat in arg.upper() for pat in ("--PASSWORD", "--TOKEN", "--KEY", "--SECRET")):
            redacted.append(arg)
            redact_next = True
        elif "=" in arg and any(pat in arg.upper().split("=")[0] for pat in SENSITIVE_PATTERNS):
            k, _ = arg.split("=", 1)
            redacted.append(f"{k}=[REDACTED]")
        else:
            redacted.append(arg)
    return redacted


def safe_run(
    cmd: Sequence[str],
    cwd: Optional[Union[str, os.PathLike]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    error_code: str = "E-SUBPROC-001",
    phase: str = "subprocess",
) -> CommandResult:
    """Execute command safely without shell expansion and capture output."""
    if isinstance(cmd, str):
        raise TypeError("safe_run requires a list or tuple of string arguments, not a string.")

    cmd_list = [str(c) for c in cmd]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.run(
            cmd_list,
            cwd=cwd,
            env=merged_env,
            timeout=timeout,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        safe_cmd_str = " ".join(redact_command(cmd_list))
        raise YoloHailoError(
            message=f"Command timed out after {timeout}s: {safe_cmd_str}",
            code=error_code,
            phase=phase,
            remediation="Increase the command timeout or check system load.",
        ) from e
    except FileNotFoundError as e:
        safe_cmd_str = " ".join(redact_command(cmd_list))
        raise YoloHailoError(
            message=f"Command executable not found: {cmd_list[0]}",
            code=error_code,
            phase=phase,
            remediation=f"Ensure '{cmd_list[0]}' is installed and present in PATH.",
        ) from e

    res = CommandResult(
        command=cmd_list,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_seconds=0.0,
    )

    if check and proc.returncode != 0:
        safe_cmd_str = " ".join(redact_command(cmd_list))
        err_excerpt = (proc.stderr or proc.stdout or "").strip()
        if len(err_excerpt) > 500:
            err_excerpt = err_excerpt[-500:]
        raise YoloHailoError(
            message=f"Command failed (exit {proc.returncode}): {safe_cmd_str}\n{err_excerpt}",
            code=error_code,
            phase=phase,
            remediation="Check the command stderr output above for details.",
        )

    return res
