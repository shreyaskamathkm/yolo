from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class DownloadDetail:
    url: str
    file_size: int


@dataclass
class DownloadOptions:
    details: Dict[str, DownloadDetail]


@dataclass
class DatasetConfig:
    path: str
    class_num: int
    class_list: List[str]
    auto_download: Optional[DownloadOptions]


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    dataloader_workers: int
    image_size: List[int]
    data_augment: Dict[str, int]
    source: Optional[Union[str, int]]
    dynamic_shape: Optional[bool]
    equivalent_batch_size: Optional[int] = 64
    drop_last: bool = True
