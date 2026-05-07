# Thin re-export — all dataclasses now live in yolo/config/schemas/
from dataclasses import dataclass
from typing import List, Optional, Union

from omegaconf import OmegaConf

from yolo.config.schemas.data import (
    DataConfig,
    DatasetConfig,
    DownloadDetail,
    DownloadOptions,
)
from yolo.config.schemas.model import (
    AnchorConfig,
    BlockConfig,
    CompileConfig,
    LayerConfg,
    ModelConfig,
    YOLOLayer,
)
from yolo.config.schemas.task import (
    ExportConfig,
    InferenceConfig,
    NMSConfig,
    ValidationConfig,
)
from yolo.config.schemas.training import (
    EMAConfig,
    LossConfig,
    MatcherConfig,
    OptimizerArgs,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
)
from yolo.schema import TrainerTaskType


@dataclass
class Config:
    task: Union[TrainConfig, InferenceConfig, ValidationConfig, ExportConfig]
    dataset: DatasetConfig
    model: ModelConfig
    name: str

    trainer: TrainerConfig

    image_size: List[int]

    out_path: str
    exist_ok: bool

    seed: int
    use_wandb: bool
    use_tensorboard: bool

    task_type: TrainerTaskType
    weight: Optional[str]
    quiet: bool = False


def resolve_config(cfg: Config) -> Config:
    from yolo.schema import TaskMode, TrainerTaskType

    cfg.task_type = TrainerTaskType(cfg.task_type)
    # The 'task' field inside the task config represents the action mode
    if hasattr(cfg.task, "task"):
        cfg.task.task = TaskMode(cfg.task.task)
    return cfg


IDX_TO_ID = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    154,
    155,
    156,
    157,
    158,
    159,
]

__all__ = [
    "AnchorConfig",
    "LayerConfg",
    "BlockConfig",
    "ModelConfig",
    "CompileConfig",
    "YOLOLayer",
    "DownloadDetail",
    "DownloadOptions",
    "DatasetConfig",
    "DataConfig",
    "OptimizerArgs",
    "OptimizerConfig",
    "MatcherConfig",
    "TrainerConfig",
    "LossConfig",
    "SchedulerConfig",
    "EMAConfig",
    "TrainConfig",
    "NMSConfig",
    "InferenceConfig",
    "ValidationConfig",
    "ExportConfig",
    "Config",
    "IDX_TO_ID",
]
