from yolo.utils.registry import Registry

DATASETS = Registry("datasets")

from yolo.data.datasets.coco import COCODetectionDataset, COCOSegmentationDataset
from yolo.data.datasets.yolo import YOLODetectionDataset, YOLOSegmentationDataset

__all__ = [
    "DATASETS",
    "COCODetectionDataset",
    "COCOSegmentationDataset",
    "YOLODetectionDataset",
    "YOLOSegmentationDataset",
]
