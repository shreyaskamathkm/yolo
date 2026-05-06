# Augmentation

The `yolo` repository provides a robust and flexible image augmentation pipeline designed for object detection and instance segmentation tasks.

## Architecture

The augmentation system is built around a standardized interface that ensures all transformations handle bounding boxes and segmentation masks in sync.

### `BaseTransform`

All augmentation classes inherit from `BaseTransform`. This base class provides:
- A consistent `__call__` signature: `(image, boxes, masks) -> (image, boxes, masks)`.
- Helper methods for cloning tensors to prevent in-place mutation of dataset caches.
- Utility functions for filtering degenerate boxes (zero-area) and valid masks.

### `AugmentationComposer`

The `AugmentationComposer` is responsible for chaining multiple transforms together. It also handles:
- Injecting the "parent" sampler into multi-image transforms like `Mosaic` and `MixUp`.
- Applying a final `PadAndResize` operation to ensure images match the target input size.
- Converting the final PIL image to a PyTorch tensor.

## Supported Transforms

- **`HorizontalFlip` / `VerticalFlip`**: Randomly flips images and coordinates.
- **`RandomCrop`**: Crops the image to half-size, clamping and filtering labels that fall outside.
- **`Mosaic`**: Combines 4 images into a 2x2 grid. Fully supports segmentation masks.
- **`MixUp`**: Blends two images using a beta distribution.
- **`RemoveOutliers`**: Filters out boxes smaller than a specified threshold.
- **`PadAndResize`**: Standard letterbox resizing.

## Multi-Image Sampling

Transforms like `Mosaic` and `MixUp` require additional samples from the dataset. This is handled via a **sampler injection** pattern:

1. The `BaseDataset` provides a `get_more_data(n)` method.
2. This method is attached to the `AugmentationComposer`.
3. The composer injects itself as a `parent` into the transforms.
4. The transforms call `self.parent.get_more_data(n)` at runtime.

This design allows these transforms to be portable while still having access to the dataset when needed.

## Standalone Testing

You can test any transform standalone without a full dataloader loop. This is useful for debugging or visualizing specific augmentations.

```python
from PIL import Image
import torch
from yolo.data.augmentation.transforms import Mosaic

# 1. Create a dummy parent to provide more data
class MockParent:
    def __init__(self, base_size=640):
        self.base_size = base_size
    def get_more_data(self, n):
        # Return list of (Image, boxes, masks)
        return [(Image.new('RGB', (640, 640)), torch.zeros((0, 5)), None)] * n

# 2. Setup transform
mosaic = Mosaic(prob=1.0).set_parent(MockParent())

# 3. Run on your image
img = Image.open("test.jpg")
boxes = torch.tensor([[0, 0.1, 0.1, 0.5, 0.5]]) # [cls, x1, y1, x2, y2]
aug_img, aug_boxes, aug_masks = mosaic(img, boxes)
```

## Interactive Demonstrations

For a hands-on exploration of the augmentation system, refer to the following notebooks:

- **[COCO Dataloader Demo](https://github.com/shreyaskamathkm/yolo/blob/main/notebooks/coco_dataloader_demo.ipynb)**:
  - Demonstrates the end-to-end detection pipeline.
  - Visualizes complex multi-image augmentations like **Mosaic** and **MixUp**.
  - Shows how to use `TrainerTaskType` and `DataSplitType` for configuration-driven data loading.
- **[COCO Segmentation Demo](https://github.com/shreyaskamathkm/yolo/blob/main/notebooks/coco_segmentation_demo.ipynb)**:
  - Focuses on the instance segmentation pipeline.
  - Showcases synchronized transformations of images, bounding boxes, and **segmentation masks** (polygons).
  - Demonstrates how the system handles mask-to-box conversion for hybrid transforms.

These notebooks provide a safe environment to experiment with different augmentation settings before starting a full training run.
