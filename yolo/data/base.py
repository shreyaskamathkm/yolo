from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.data.augmentation import (
    AugmentationComposer,
    HorizontalFlip,
    MixUp,
    Mosaic,
    PadAndResize,
    RandomCrop,
    RemoveOutliers,
    VerticalFlip,
)
from yolo.utils.logger import logger


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


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    def __init__(self, dataset_path: Path, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        self.dataset_path = dataset_path
        self.data_cfg = data_cfg
        self.dataset_cfg = dataset_cfg
        self.phase = phase
        self.phase_name = dataset_cfg.get(phase, phase)

        self.image_size = data_cfg.image_size
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        augment_cfg = data_cfg.data_augment
        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size)
        self.transform.get_more_data = self.get_more_data

        self.data_list = self.load_data(Path(dataset_cfg.path), self.phase_name)

    def load_data(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, Any, float]]:
        """Loads or generates dataset cache."""
        clean_name = Path(phase_name).stem
        cache_path = dataset_path / f"{clean_name}.pache"

        if not cache_path.exists():
            logger.info(f":factory: Generating {phase_name} cache")
            data = self.load_valid_labels(dataset_path, phase_name)
            torch.save(data, cache_path)
        else:
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception as e:
                logger.error(f":rotating_light: Failed to load the cache at '{cache_path}'.")
                raise e
            logger.info(f":package: Loaded {phase_name} cache, {len(data)} items found.")
        return data

    @abstractmethod
    def load_valid_labels(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, Any, float]]:
        """Finds and pairs images with labels."""
        pass

    def get_image(self, idx: int) -> Tuple[Image.Image, Path]:
        img_path, _, _ = self.data_list[idx]
        return Image.open(img_path).convert("RGB"), img_path

    @abstractmethod
    def get_labels(self, idx: int) -> Any:
        """Returns the labels for the given index."""
        pass

    def get_more_data(self, num: int = 1) -> List[Tuple[Image.Image, Any]]:
        indices = torch.randint(0, len(self), (num,))
        results = []
        for idx in indices:
            img, _ = self.get_image(idx.item())
            labels = self.get_labels(idx.item())
            results.append((img, labels))
        return results

    def _update_image_size(self, idx: int) -> None:
        batch_start_idx = (idx // self.batch_size) * self.batch_size
        _, _, ratio = self.data_list[batch_start_idx]
        image_ratio = np.clip(ratio, 1 / 3, 3)
        shift = ((self.base_size / 32 * (image_ratio - 1)) // (image_ratio + 1)) * 32
        self.image_size = [int(self.base_size + shift), int(self.base_size - shift)]
        self.transform.pad_resize.set_size(self.image_size)

    def __len__(self) -> int:
        return len(self.data_list)

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        pass
