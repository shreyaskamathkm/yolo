from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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
    type: str = "coco"
    auto_download: Optional[Dict[str, Any]] = None
    train: Optional[str] = None
    validation: Optional[str] = None
    test: Optional[str] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    dataloader_workers: int
    image_size: List[int]
    data_augment: List[Dict[str, Any]]
    source: Optional[Union[str, int]] = None
    dynamic_shape: Optional[bool] = False
    equivalent_batch_size: Optional[int] = 64
    drop_last: bool = True
    redo_cache: bool = False
