from typing import List

import numpy as np
import torch
from torch import Tensor

from yolo.data.base.dataset import BaseDataset
from yolo.data.schema import Sample


class DetectionDataset(BaseDataset):
    """Base class for detection-specific datasets."""

    def get_labels(self, idx: int) -> Tensor:
        _, annotations, _ = self.data_list[idx]
        return self.parse_labels(annotations)

    def parse_labels(self, annotations: List[List[float]]) -> Tensor:
        bboxes = []
        for seg_data in annotations:
            cls = seg_data[0]
            points = np.array(seg_data[1:]).reshape(-1, 2).clip(0, 1)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)
        if bboxes:
            return torch.stack(bboxes)
        return torch.zeros((0, 5))

    def __getitem__(self, idx: int) -> Sample:
        img, img_path = self.get_image(idx)
        labels = self.get_labels(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, labels, masks, rev_tensor = self.transform(img, labels)

        # Post-process labels based on current image size (e.g. scale bboxes)
        if isinstance(labels, Tensor) and labels.dim() == 2 and labels.size(1) == 5:
            labels[:, [1, 3]] *= self.image_size[0]
            labels[:, [2, 4]] *= self.image_size[1]

        return Sample(
            image=img,
            bboxes=labels,
            masks=masks,
            reverse_transforms=rev_tensor,
            info={"path": str(img_path), "rev_tensor": rev_tensor},
        )
