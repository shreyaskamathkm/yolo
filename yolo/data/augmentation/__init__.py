from typing import List, Tuple

import torch
from torchvision.transforms import functional as TF

from yolo.data.augmentation.transforms import (
    HorizontalFlip,
    MixUp,
    Mosaic,
    PadAndResize,
    RandomCrop,
    RemoveOutliers,
    VerticalFlip,
)


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: Tuple[int, int] = (640, 640), base_size: int = 640):
        self.transforms = transforms
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5), masks=None):
        for transform in self.transforms:
            image, boxes, masks = transform(image, boxes, masks)

        # Final padding/resize returns (image, boxes, masks, transform_info)
        image, boxes, masks, rev_tensor = self.pad_resize(image, boxes, masks)
        image = TF.to_tensor(image)
        return image, boxes, masks, rev_tensor
