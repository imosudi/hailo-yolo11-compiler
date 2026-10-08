"""Dataset statistics models and calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SplitStatistics:
    name: str
    num_images: int = 0
    num_labels: int = 0
    num_boxes: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    missing_labels: int = 0
    corrupt_images: int = 0
    empty_images: int = 0
    widths: List[int] = field(default_factory=list, repr=False)
    heights: List[int] = field(default_factory=list, repr=False)

    def summary(self) -> Dict[str, Any]:
        avg_w = round(sum(self.widths) / len(self.widths), 1) if self.widths else 0
        avg_h = round(sum(self.heights) / len(self.heights), 1) if self.heights else 0
        return {
            "num_images": self.num_images,
            "num_labels": self.num_labels,
            "num_boxes": self.num_boxes,
            "class_counts": self.class_counts,
            "missing_labels": self.missing_labels,
            "corrupt_images": self.corrupt_images,
            "empty_images": self.empty_images,
            "avg_width": avg_w,
            "avg_height": avg_h,
        }


@dataclass
class DatasetValidationReport:
    is_valid: bool
    status: str  # "PASSED" or "FAILED"
    train_stats: SplitStatistics
    val_stats: SplitStatistics
    test_stats: Optional[SplitStatistics] = None
    duplicate_images: int = 0
    train_val_leakage: int = 0
    calibration_candidates: int = 0
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "duplicate_images": self.duplicate_images,
            "train_val_leakage": self.train_val_leakage,
            "calibration_candidates": self.calibration_candidates,
            "train": self.train_stats.summary(),
            "val": self.val_stats.summary(),
            "test": self.test_stats.summary() if self.test_stats else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def format_terminal_summary(self) -> str:
        total_images = self.train_stats.num_images + self.val_stats.num_images
        total_labels = self.train_stats.num_labels + self.val_stats.num_labels
        lines = [
            "DATASET VALIDATION",
            "------------------",
            f"Images:                 {total_images:,}",
            f"Labels:                 {total_labels:,}",
            f"Missing labels:         {self.train_stats.missing_labels + self.val_stats.missing_labels:,}",
            f"Duplicate images:       {self.duplicate_images:,}",
            f"Train/val leakage:      {self.train_val_leakage:,}",
            f"Classes:                {self.num_classes}",
            f"Calibration candidates: {self.calibration_candidates:,}",
            "",
            f"STATUS: {self.status}",
        ]
        if self.errors:
            lines.append(f"Reason: {'; '.join(self.errors)}")
        return "\n".join(lines)
