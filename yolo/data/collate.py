from typing import List

import torch

from yolo.data.base import Batch, Sample


def collate_fn(batch: List[Sample]) -> Batch:
    """Collates a list of Samples into a Batch."""
    batch_size = len(batch)
    images = torch.stack([s.image for s in batch])
    paths = [s.info["path"] for s in batch]
    reverse_transforms = torch.stack([s.reverse_transforms for s in batch])

    # Handle bboxes
    target_list = [s.bboxes for s in batch if s.bboxes is not None]
    if target_list:
        max_targets = max([t.size(0) for t in target_list])
        max_targets = max(min(max_targets, 100), 1)
        padded_targets = torch.zeros((batch_size, max_targets, 5))
        padded_targets[:, :, 0] = -1
        for i, t in enumerate(target_list):
            num = min(t.size(0), max_targets)
            padded_targets[i, :num] = t[:num]
    else:
        padded_targets = torch.zeros((batch_size, 0, 5))

    # Handle masks (polygons) - keep as list or pad if needed
    masks = [s.masks for s in batch] if any(s.masks is not None for s in batch) else None

    return Batch(
        images=images,
        targets=padded_targets,
        masks=masks,
        reverse_transforms=reverse_transforms,
        paths=paths,
        batch_size=batch_size,
    )
