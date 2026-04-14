from dataclasses import dataclass
from typing import Optional

from yolo.config.schemas.data import DataConfig  # noqa: F401


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float
    max_bbox: int


@dataclass
class InferenceConfig:
    task: str
    nms: NMSConfig
    data: DataConfig
    fast_inference: Optional[None]
    save_predict: bool


@dataclass
class ValidationConfig:
    task: str
    nms: NMSConfig
    data: DataConfig
