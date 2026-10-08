"""Manifest serialisation for calibration dataset provenance."""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

from yolo_hailo_mlops.utils.filesystem import atomic_write


@dataclass
class CalibrationSampleRecord:
    source_path: str
    calib_filename: str
    sha256: str
    original_size: List[int]
    preprocessed_shape: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationManifest:
    count: int
    seed: int
    source: str
    preprocessing_contract: Dict[str, Any]
    samples: List[CalibrationSampleRecord] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "seed": self.seed,
            "source": self.source,
            "created_at": self.created_at,
            "preprocessing_contract": self.preprocessing_contract,
            "samples": [s.to_dict() for s in self.samples],
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        json_data = json.dumps(self.to_dict(), indent=2)
        return atomic_write(output_path, json_data)
