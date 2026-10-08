"""Cryptographic hashing utilities for artifact verification and reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 digest of a file in streaming chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_sha256(content: str) -> str:
    """Compute the SHA-256 digest of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_dir_sha256(dir_path: Union[str, Path]) -> str:
    """Compute a deterministic SHA-256 digest of a directory tree."""
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Cannot hash non-existent directory: {path}")

    hasher = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            rel_path = child.relative_to(path).as_posix()
            hasher.update(rel_path.encode("utf-8"))
            with open(child, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
    return hasher.hexdigest()
