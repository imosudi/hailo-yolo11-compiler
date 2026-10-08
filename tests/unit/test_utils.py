"""Unit tests for filesystem, hashing, and subprocess utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_hailo_mlops.exceptions import YoloHailoError
from yolo_hailo_mlops.utils.filesystem import atomic_write, ensure_dir, safe_symlink_or_copy
from yolo_hailo_mlops.utils.hashing import compute_dir_sha256, compute_sha256, compute_string_sha256
from yolo_hailo_mlops.utils.subprocess import redact_command, safe_run


def test_atomic_write_and_hashing(temp_workspace: Path) -> None:
    """Verify atomic write creates the file and hashing is consistent."""
    target = temp_workspace / "test.txt"
    atomic_write(target, "Hello World\n")

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "Hello World\n"

    sha = compute_sha256(target)
    assert sha == compute_string_sha256("Hello World\n")


def test_safe_symlink_or_copy(temp_workspace: Path) -> None:
    """Verify symlink creation or copy fallback."""
    src = temp_workspace / "src.txt"
    src.write_text("content", encoding="utf-8")
    dst = temp_workspace / "dst.txt"

    safe_symlink_or_copy(src, dst)
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "content"


def test_subprocess_secret_redaction() -> None:
    """Verify sensitive tokens are redacted from logged command lists."""
    cmd = ["hailo", "login", "--token", "super_secret_jwt", "--user=admin"]
    redacted = redact_command(cmd)
    assert "super_secret_jwt" not in redacted
    assert "[REDACTED]" in redacted


def test_subprocess_safe_run() -> None:
    """Verify safe subprocess execution captures stdout."""
    res = safe_run(["echo", "mlops_ready"])
    assert res.exit_code == 0
    assert "mlops_ready" in res.stdout


def test_subprocess_safe_run_failure() -> None:
    """Verify non-zero return code raises typed error when check=True."""
    with pytest.raises(YoloHailoError):
        safe_run(["false"])
