from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from rich.progress import track
from torch import Tensor

from yolo.data.base.detect import DetectionDataset
from yolo.data.base.segment import SegmentationDataset
from yolo.data.schema import TrainerTaskType
from yolo.registry import DATASETS


@DATASETS.register_module(name=(TrainerTaskType.DETECTION, "yolo"))
class YOLODetectionDataset(DetectionDataset):
    """Dataset for YOLO-format detection labels (.txt)."""

    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, Tensor, float]]:
        data = []
        # Determine image and label directories
        phase_name = Path(phase_path).name
        image_dir = dataset_path / "images" / phase_name
        labels_dir = dataset_path / "labels" / phase_name

        if not image_dir.exists():
            # Try if phase_path itself is the image directory
            if Path(phase_path).is_dir():
                image_dir = Path(phase_path)
                labels_dir = Path(str(image_dir).replace("images", "labels"))
            else:
                raise FileNotFoundError(f"Could not find image directory for phase '{phase_path}' in '{dataset_path}'")

        image_paths = sorted(image_dir.iterdir())
        for img_path in track(image_paths, description="Filtering"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            labels = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Normalized to pixel coordinates in __getitem__ or transform
            data.append((img_path, labels, 1.0))  # Placeholder for ratio
        return data


@DATASETS.register_module(name=(TrainerTaskType.SEGMENTATION, "yolo"))
class YOLOSegmentationDataset(SegmentationDataset):
    """Dataset for YOLO-format segmentation labels (.txt)."""

    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, List[Tensor], float]]:
        data = []
        # Determine image and label directories
        phase_name = Path(phase_path).name
        image_dir = dataset_path / "images" / phase_name
        labels_dir = dataset_path / "labels" / phase_name

        if not image_dir.exists():
            # Try if phase_path itself is the image directory
            if Path(phase_path).is_dir():
                image_dir = Path(phase_path)
                labels_dir = Path(str(image_dir).replace("images", "labels"))
            else:
                raise FileNotFoundError(f"Could not find image directory for phase '{phase_path}' in '{dataset_path}'")

        image_paths = sorted(image_dir.iterdir())
        for img_path in track(image_paths, description="Filtering"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            # YOLO segments: class x1 y1 x2 y2 ...
            with open(label_path, "r") as f:
                segments = [torch.tensor([float(x) for x in line.split()]) for line in f.read().splitlines()]
            data.append((img_path, segments, 1.0))
        return data
