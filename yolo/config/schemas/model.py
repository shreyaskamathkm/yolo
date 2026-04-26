from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch import nn


@dataclass
class AnchorConfig:
    strides: List[int]
    reg_max: Optional[int]
    anchor_num: Optional[int]
    anchor: List[List[int]]


@dataclass
class LayerConfg:
    args: Dict
    source: Union[int, str, List[int]]
    tags: str


@dataclass
class BlockConfig:
    block: List[Dict[str, LayerConfg]]


@dataclass
class CompileConfig:
    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    backend: str = "inductor"


@dataclass
class ModelConfig:
    name: Optional[str]
    anchor: AnchorConfig
    model: Dict[str, BlockConfig]
    compile: Optional[CompileConfig] = None


@dataclass
class YOLOLayer(nn.Module):
    source: Union[int, str, List[int]]
    output: bool
    tags: str
    layer_type: str
    usable: bool
    external: Optional[dict]
