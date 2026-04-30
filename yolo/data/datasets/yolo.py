from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from rich.progress import track
from torch import Tensor

from yolo.data.datasets import DATASETS
from yolo.data.datasets.base import DetectionDataset, SegmentationDataset


@DATASETS.register_module(name="detect_txt")
class YOLODetectionDataset(DetectionDataset):
    """Dataset for YOLO-format detection labels (.txt)."""

    def load_valid_labels(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, Tensor, float]]:
        data = []
        image_paths = sorted((dataset_path / "images" / phase_name).iterdir())
        labels_path = dataset_path / "labels" / phase_name
        for img_path in track(image_paths, description="Filtering"):
            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            labels = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Normalized to pixel coordinates in __getitem__ or transform
            data.append((img_path, labels, 1.0))  # Placeholder for ratio
        return data


@DATASETS.register_module(name="segment_txt")
class YOLOSegmentationDataset(SegmentationDataset):
    """Dataset for YOLO-format segmentation labels (.txt)."""

    def load_valid_labels(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, List[Tensor], float]]:
        data = []
        image_paths = sorted((dataset_path / "images" / phase_name).iterdir())
        labels_path = dataset_path / "labels" / phase_name
        for img_path in track(image_paths, description="Filtering"):
            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            # YOLO segments: class x1 y1 x2 y2 ...
            with open(label_path, "r") as f:
                segments = [torch.tensor([float(x) for x in line.split()]) for line in f.read().splitlines()]
            data.append((img_path, segments, 1.0))
        return data
