# Data Augmentation

The `yolo` repository provides a robust image augmentation pipeline designed for object detection and instance segmentation. All transformations handle bounding boxes and segmentation masks in sync.

## Architecture

### `BaseTransform`

All augmentation classes inherit from `BaseTransform`. This ensures a consistent interface:
- **Signature**: `__call__(image, boxes, masks) -> (image, boxes, masks)`
- **No Mutation**: Tensors are cloned before modification to prevent dataset cache corruption.
- **Filtering**: Automatic pruning of degenerate boxes (zero-area) and collapsed polygons.

### `AugmentationComposer`

The `AugmentationComposer` chains multiple transforms. It also handles the injection of the dataset sampler for multi-image transforms like `Mosaic` and `MixUp`.

## Built-in Transforms

| Class | Parameters | Mask Support | Description |
|---|---|---|---|
| `HorizontalFlip` | `prob=0.5` | ✅ | Flips image and labels horizontally. |
| `VerticalFlip` | `prob=0.5` | ✅ | Flips image and labels vertically. |
| `RandomCrop` | `prob=0.5` | ✅ | Crops to 50% size; clamps/filters labels outside the crop. |
| `Mosaic` | `prob=0.5` | ✅ | Combines 4 images into a 2x2 grid. Requires a `parent` dataset. |
| `MixUp` | `prob=0.5`, `alpha=1.0` | ✅ | Blends two images using Beta distribution. Keeps labels from both images. |
| `RemoveOutliers` | `min_box_area=1e-8` | ✅ | Drops boxes below a minimum area threshold. |
| `PadAndResize` | `image_size`, `bg=(114,114,114)` | ✅ | Letterbox resizing to target input size. |

## Multi-Image Sampling

Transforms like `Mosaic` and `MixUp` need additional data. This is handled via the `set_parent` mechanism. The `AugmentationComposer` automatically links the transforms to the dataset's sampling method.

## Writing a Custom Transform

A custom transform should inherit from `BaseTransform`:

```python
from yolo.data.augmentation.transforms import BaseTransform

class MyTransform(BaseTransform):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, boxes, masks=None):
        if torch.rand(1).item() > self.prob:
            return image, boxes, masks

        # 1. Clone inputs to avoid cache mutation
        boxes = self._clone_boxes(boxes)
        masks = self._clone_masks(masks)

        # 2. ... your custom augmentation logic here ...
        # (Modify image, update boxes/masks coordinates, etc.)

        # 3. Filter degenerate boxes and collapsed polygons
        boxes, masks = self._filter_boxes_and_masks(boxes, masks)

        return image, boxes, masks
```

## Standalone Testing

You can test any transform without a full dataloader loop. This is useful for debugging.

```python
from PIL import Image
import torch
from yolo.data.augmentation.transforms import Mosaic

# 1. Mock parent for sampling
class MockParent:
    def __init__(self, base_size=640): self.base_size = base_size
    def get_more_data(self, n): return [(Image.new('RGB', (640, 640)), torch.zeros((0, 5)), None)] * n

# 2. Run transform
mosaic = Mosaic(prob=1.0).set_parent(MockParent())
img = Image.open("test.jpg")
aug_img, aug_boxes, aug_masks = mosaic(img, torch.zeros((0, 5)))
```

For interactive examples, see the
- [Detection Demo](../../notebooks/coco_dataloader_demo.ipynb).
- [Segmentation Demo](../../notebooks/coco_segmentation_demo.ipynb)
