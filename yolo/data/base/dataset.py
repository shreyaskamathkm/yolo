import logging
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.data.augmentation import Compose
from yolo.data.schema import Sample
from yolo.utils.distributed import rank_zero_first
from yolo.data.schema import TrainerTaskType

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    def __init__(
        self,
        dataset_path: Path,
        data_cfg: DataConfig,
        dataset_cfg: DatasetConfig,
        task: str = "detect",
        phase: str = "train2017",
    ):
        self.dataset_path = dataset_path
        self.data_cfg = data_cfg
        self.dataset_cfg = dataset_cfg
        self.task = task
        self.phase = phase
        self.phase_path = dataset_cfg.get(phase, phase)

        self.image_size = data_cfg.image_size
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = int(mean(self.image_size))

        self.transform = Compose(data_cfg.data_augment, tuple(self.image_size))
        self.transform.get_more_data = self.get_more_data

        # Robust path handling: if path doesn't exist, try relative to project root
        actual_dataset_path = Path(dataset_cfg.path)
        if not actual_dataset_path.exists():
            # Try looking one level up (common for notebooks in subfolders)
            parent_path = Path("..") / actual_dataset_path
            if parent_path.exists():
                actual_dataset_path = parent_path
            else:
                logger.warning(f"Dataset path '{actual_dataset_path}' not found.")

        self.data_list = self.load_data(actual_dataset_path, self.phase_path)

        if len(self.data_list) == 0:
            raise RuntimeError(
                f"Dataset is empty! No valid images/labels found in '{actual_dataset_path}' for phase '{self.phase_path}'. "
                "Please check your dataset paths and annotation files."
            )

    def load_data(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, Any, float]]:
        """Loads or generates dataset cache."""
        clean_name = Path(phase_path).stem
        cache_path = dataset_path / f"{clean_name}_{self.task}_{self.dataset_cfg.type}.cache"

        with rank_zero_first():
            if cache_path.exists() and self.data_cfg.redo_cache:
                logger.info(f":wastebasket: Removing existing cache: {cache_path}")
                cache_path.unlink()

            if not cache_path.exists():
                logger.info(f":factory: Generating {phase_path} cache")
                data = self.load_valid_labels(dataset_path, phase_path)
                temp_path = cache_path.with_suffix(".tmp")
                torch.save(data, temp_path)
                temp_path.rename(cache_path)

        try:
            data = torch.load(cache_path, weights_only=False)
        except Exception as e:
            logger.error(f":rotating_light: Failed to load the cache at '{cache_path}'.")
            raise e
        logger.info(f":package: Loaded {phase_path} cache, {len(data)} items found.")
        return data

    @abstractmethod
    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, Any, float]]:
        """Finds and pairs images with labels."""
        pass

    def get_image(self, idx: int) -> Tuple[Image.Image, Path]:
        img_path, _, _ = self.data_list[idx]
        return Image.open(img_path).convert("RGB"), img_path

    @abstractmethod
    def get_labels(self, idx: int) -> Any:
        """Returns the labels for the given index."""
        pass

    def get_more_data(self, num: int = 1) -> List[Tuple[Image.Image, Any, Optional[List[Tensor]]]]:
        indices = torch.randint(0, len(self), (num,))
        results = []
        for idx in indices:
            img, _ = self.get_image(idx.item())
            labels = self.get_labels(idx.item())
            # Detection labels are bboxes [N, 5], Segmentation labels are polygons [List[Tensor]]
            if self.task == TrainerTaskType.DETECTION:
                results.append((img, labels, None))
            elif self.task == TrainerTaskType.SEGMENTATION:
                # For segmentation, we need to derive bboxes for transforms that use them (like Mosaic)
                bboxes = []
                for poly in labels:
                    cls = poly[0]
                    pts = torch.as_tensor(poly[1:]).reshape(-1, 2)
                    bboxes.append([cls, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
                bboxes = torch.tensor(bboxes).reshape(-1, 5)
                results.append((img, bboxes, labels))
            else:
                results.append((img, labels, None))
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
