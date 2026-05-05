from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from torch import Tensor


class TrainerTaskType(str, Enum):
    # __str__ is overridden for Python 3.10 compatibility (StrEnum is 3.11+)
    # This ensures f-strings and registry lookups use the value instead of the repr
    DETECTION = "detect"
    SEGMENTATION = "segment"
    INFERENCE = "inference"

    def __str__(self):
        return self.value


class DataSplitType(str, Enum):
    # __str__ is overridden for Python 3.10 compatibility
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    INFERENCE = "inference"

    def __str__(self):
        return self.value


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
