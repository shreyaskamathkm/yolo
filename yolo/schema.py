from enum import Enum


class TrainerTaskType(str, Enum):
    """Supported computer vision tasks."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"

    def __str__(self):
        return self.value


class TaskMode(str, Enum):
    """Operation modes for the YOLO framework."""

    TRAIN = "train"
    VAL = "validation"
    INFERENCE = "inference"
    EXPORT = "export"

    def __str__(self):
        return self.value


class DataSplitType(str, Enum):
    """Standard dataset splits."""

    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

    def __str__(self):
        return self.value
