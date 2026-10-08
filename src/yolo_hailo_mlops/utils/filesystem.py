"""Safe atomic file operations, directory creation, and symlink management."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write(
    target_path: Union[str, Path],
    content: Union[str, bytes],
    binary: bool = False,
) -> Path:
    """Atomically write content to target path using a tempfile and rename."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"

    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=str(target.parent),
        prefix=f".tmp_{target.name}_",
        delete=False,
        encoding=encoding,
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)

    # Atomic replace
    temp_path.replace(target)
    return target


def safe_symlink_or_copy(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
) -> Path:
    """Create a relative symlink to source at target, falling back to copy."""
    src = Path(source_path).resolve()
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)

    try:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
    except (OSError, NotImplementedError):
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return dst
