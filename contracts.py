from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass
class Observation:
    name: str
    rgb: np.ndarray
    depth_mm: np.ndarray
    valid: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class Segment:
    id: int
    points_yx: np.ndarray
    closed: bool = False
    geometry: dict = field(default_factory=dict)


@dataclass
class Quantification:
    skeleton: np.ndarray
    segments: list[Segment]
    features: dict
    physical_available: bool
    missing: list[str]


def serializable(value):
    if hasattr(value, '__dataclass_fields__'):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
