"""Structured multi-target logging supporting Console, Plain-text, and JSON Lines."""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class JSONLineFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", self.run_id),
            "phase": getattr(record, "phase", None),
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "duration_seconds"):
            log_entry["duration_seconds"] = record.duration_seconds
        if hasattr(record, "artifact"):
            log_entry["artifact"] = record.artifact
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class StructuredConsoleFormatter(logging.Formatter):
    """Formats human-readable coloured console logs with RUN and PHASE tags."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_tag = f"[RUN={getattr(record, 'run_id', self.run_id or 'none')[:18]}]"
        phase = getattr(record, "phase", None)
        phase_tag = f"[{phase}]" if phase else ""
        level = f"{record.levelname:<5}"

        msg = record.getMessage()
        return f"{t} {level} {run_tag} {phase_tag} {msg}".strip()


class PipelineLogger:
    """Central logging controller for the MLOps pipeline."""

    def __init__(
        self,
        name: str = "yolo_hailo_mlops",
        run_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        self.run_id = run_id
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.logger.handlers.clear()

        # 1. Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(StructuredConsoleFormatter(run_id=self.run_id))
        self.logger.addHandler(ch)

        # 2. File handlers if log_dir provided
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Text file handler
            fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(StructuredConsoleFormatter(run_id=self.run_id))
            self.logger.addHandler(fh)

            # JSONL handler
            jh = logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8")
            jh.setLevel(logging.DEBUG)
            jh.setFormatter(JSONLineFormatter(run_id=self.run_id))
            self.logger.addHandler(jh)

    def log(
        self,
        level: int,
        message: str,
        phase: Optional[str] = None,
        event: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        artifact: Optional[str] = None,
    ) -> None:
        extra = {
            "run_id": self.run_id,
            "phase": phase,
            "event": event,
        }
        if duration_seconds is not None:
            extra["duration_seconds"] = duration_seconds
        if artifact is not None:
            extra["artifact"] = artifact

        self.logger.log(level, message, extra=extra)

    def info(self, message: str, phase: Optional[str] = None, **kwargs: Any) -> None:
        self.log(logging.INFO, message, phase=phase, **kwargs)

    def warning(self, message: str, phase: Optional[str] = None, **kwargs: Any) -> None:
        self.log(logging.WARNING, message, phase=phase, **kwargs)

    def error(self, message: str, phase: Optional[str] = None, **kwargs: Any) -> None:
        self.log(logging.ERROR, message, phase=phase, **kwargs)

    def debug(self, message: str, phase: Optional[str] = None, **kwargs: Any) -> None:
        self.log(logging.DEBUG, message, phase=phase, **kwargs)


_DEFAULT_LOGGER: Optional[PipelineLogger] = None


def get_logger(
    run_id: Optional[str] = None,
    log_dir: Optional[Path] = None,
    verbose: bool = False,
) -> PipelineLogger:
    """Get or initialise the shared pipeline logger instance."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None or run_id or log_dir:
        _DEFAULT_LOGGER = PipelineLogger(run_id=run_id, log_dir=log_dir, verbose=verbose)
    return _DEFAULT_LOGGER
