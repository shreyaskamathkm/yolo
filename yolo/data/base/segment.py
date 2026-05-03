from pathlib import Path
from typing import Any, List, Tuple

import torch

from yolo.data.base.dataset import BaseDataset
from yolo.data.schema import Sample


class SegmentationDataset(BaseDataset):
    """Dataset class for segmentation tasks."""

    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, Any, float]]:
        """Placeholder for task-specific label loading."""
        raise NotImplementedError

    def get_labels(self, idx: int) -> Any:
        _, annotations, _ = self.data_list[idx]
        return annotations

    def __getitem__(self, idx: int) -> Sample:
        img, img_path = self.get_image(idx)
        polygons = self.get_labels(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        # Derive bboxes from polygons for the detect head
        bboxes = []
        for poly in polygons:
            cls = poly[0]
            pts = poly[1:].reshape(-1, 2)
            bboxes.append([cls, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        bboxes = torch.tensor(bboxes).reshape(-1, 5)

        # Clone polygons to avoid in-place modification of the cached data
        polygons = [poly.clone() for poly in polygons]

        # Transform image, boxes, and masks
        img, bboxes, masks, rev_tensor = self.transform(img, bboxes, masks=polygons)

        # Scale bboxes to pixel coordinates for consistency with DetectionDataset
        if bboxes.size(0) > 0:
            bboxes[:, [1, 3]] *= self.image_size[0]
            bboxes[:, [2, 4]] *= self.image_size[1]

        return Sample(
            image=img,
            bboxes=bboxes,
            masks=masks,
            reverse_transforms=rev_tensor,
            info={"path": str(img_path), "rev_tensor": rev_tensor},
        )
