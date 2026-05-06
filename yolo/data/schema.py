from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch import Tensor

from yolo.schema import DataSplitType, TaskMode, TrainerTaskType


@dataclass
class Sample:
    """A structured data sample from the dataset."""

    image: Tensor
    bboxes: Optional[Tensor] = None
    masks: Optional[Tensor] = None
    keypoints: Optional[Tensor] = None
    reverse_transforms: Optional[Tensor] = None
    info: Optional[Dict[str, Any]] = None


@dataclass
class Batch:
    """A structured batch of samples."""

    images: Tensor
    targets: Optional[Tensor] = None
    masks: Optional[Any] = None
    keypoints: Optional[Tensor] = None
    reverse_transforms: Optional[Tensor] = None
    paths: Optional[List[str]] = None
    batch_size: int = 0

    def __iter__(self):
        return iter((self.batch_size, self.images, self.targets, self.reverse_transforms, self.paths))
