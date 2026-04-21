from dataclasses import dataclass
from typing import Any, Dict, Union

from yolo.config.schemas.data import DataConfig
from yolo.config.schemas.task import ValidationConfig


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float
    momentum: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class MatcherConfig:
    iou: str
    topk: int
    factor: Dict[str, int]


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    device: Union[str, int] = "auto"
    precision: str = "32-true"
    sync_batchnorm: bool = True
    log_every_n_steps: int = 1
    gradient_clip_val: float = 10.0
    gradient_clip_algorithm: str = "norm"
    deterministic: bool = True


@dataclass
class LossConfig:
    objective: Dict[str, int]
    aux: Union[bool, float]
    matcher: MatcherConfig


@dataclass
class SchedulerConfig:
    type: str
    warmup: Dict[str, Union[int, float]]
    args: Dict[str, Any]


@dataclass
class EMAConfig:
    enable: bool
    decay: float


@dataclass
class TrainConfig:
    task: str
    epoch: int
    save_all_checkpoints: bool
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    ema: EMAConfig
    validation: ValidationConfig
