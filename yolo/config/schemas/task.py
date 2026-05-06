from dataclasses import dataclass
from typing import List, Optional

from yolo.config.schemas.data import DataConfig  # noqa: F401
from yolo.schema import TrainerTaskType


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float
    max_bbox: int


@dataclass
class InferenceConfig:
    task: TrainerTaskType
    nms: NMSConfig
    data: DataConfig
    backend: str
    save_predict: bool


@dataclass
class ValidationConfig:
    task: TrainerTaskType
    nms: NMSConfig
    data: DataConfig


@dataclass
class ExportConfig:
    task: TrainerTaskType
    formats: List[str]
    output_dir: str
