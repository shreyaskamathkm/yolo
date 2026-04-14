# Data Augmentation

See [HOWTO](../HOWTO.md#custom-data-augmentation) for a guide on adding custom augmentation transforms.

## Built-in Transforms

All transforms live in `yolo/data/augmentation.py` and follow the same interface: `__call__(image, boxes) -> (image, boxes)`.

| Class | Default prob | Description |
|---|---|---|
| `PadAndResize` | — | Letterbox-pads and resizes image to a target size while adjusting boxes |
| `RemoveOutliers` | — | Drops bounding boxes whose area falls below a minimum threshold |
| `HorizontalFlip` | 0.5 | Randomly flips image and boxes horizontally |
| `VerticalFlip` | 0.5 | Randomly flips image and boxes vertically |
| `Mosaic` | 0.5 | Stitches four dataset images into one; boxes are adjusted accordingly |
| `MixUp` | 0.5 | Alpha-blends two images and merges their box lists |
| `RandomCrop` | 0.5 | Crops image to half its size; clips boxes to the new boundary |

`AugmentationComposer` chains these transforms together and handles image-size updates for transforms like `PadAndResize` that need the target size at runtime:

```python
from yolo.data.augmentation import AugmentationComposer, HorizontalFlip, Mosaic, PadAndResize

transforms = AugmentationComposer(
    [Mosaic(), HorizontalFlip(), PadAndResize(image_size)],
    image_size=640,
)
image, boxes = transforms(image, boxes)
```

## Writing a Custom Transform

A transform is any callable that accepts `(image: PIL.Image, boxes: Tensor)` and returns `(image, boxes)`:

```python
class MyTransform:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        import random
        if random.random() > self.prob:
            return image, boxes
        # ... apply transform ...
        return image, boxes
```

Pass it to `AugmentationComposer` alongside the built-in transforms:

```python
transforms = AugmentationComposer([MyTransform(0.3), HorizontalFlip()], image_size=640)
```

To override the augmentation pipeline for training, pass a custom `AugmentationComposer` to `create_dataloader`.
