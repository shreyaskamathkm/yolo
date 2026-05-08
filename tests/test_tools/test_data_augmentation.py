import torch
from PIL import Image
from torchvision.transforms import functional as TF

from yolo.data.augmentation import (
    Compose,
    HorizontalFlip,
    Mosaic,
    VerticalFlip,
)


def test_horizontal_flip():
    # Create a mock image and bounding boxes
    img = Image.new("RGB", (100, 100), color="red")
    boxes = torch.tensor([[1, 0.05, 0.1, 0.7, 0.9]])  # class, xmin, ymin, xmax, ymax

    flip_transform = HorizontalFlip(prob=1)  # Set probability to 1 to ensure flip
    flipped_img, flipped_boxes, flipped_masks = flip_transform(img, boxes)

    # Assert image is flipped by comparing it to a manually flipped image
    assert TF.hflip(img) == flipped_img

    # Assert bounding boxes are flipped correctly
    expected_boxes = torch.tensor([[1, 0.3, 0.1, 0.95, 0.9]])
    assert torch.allclose(flipped_boxes, expected_boxes), "Bounding boxes were not flipped correctly"


def test_compose():
    # Test with configuration list of dicts
    compose = Compose([{"type": "HorizontalFlip", "prob": 0}, {"type": "VerticalFlip", "prob": 0}])
    img = Image.new("RGB", (640, 640), color="blue")
    boxes = torch.tensor([[0, 0.2, 0.2, 0.8, 0.8]])

    transformed_img, transformed_boxes, transformed_masks, rev_tensor = compose(img, boxes)
    assert transformed_img.shape == (3, 640, 640)
    assert torch.equal(transformed_boxes, boxes)


def test_compose_with_config():
    # Test with configuration list of dicts
    augment_cfg = [
        {"type": "HorizontalFlip", "prob": 0.0},
        {"type": "VerticalFlip", "prob": 0.0},
    ]
    compose = Compose(augment_cfg, image_size=(640, 640))
    img = Image.new("RGB", (640, 640), color="blue")
    boxes = torch.tensor([[0, 0.2, 0.2, 0.8, 0.8]])

    transformed_img, transformed_boxes, transformed_masks, rev_tensor = compose(img, boxes)
    assert transformed_img.shape == (3, 640, 640)
    assert torch.equal(transformed_boxes, boxes)


def test_mosaic():
    img = Image.new("RGB", (100, 100), color="green")
    boxes = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])

    # Mock parent with image_size and get_more_data method
    class MockParent:
        base_size = 100

        def get_more_data(self, num_images):
            return [(img, boxes, None) for _ in range(num_images)]

    mosaic = Mosaic(prob=1)  # Ensure mosaic is applied
    mosaic.set_parent(MockParent())

    mosaic_img, mosaic_boxes, mosaic_masks = mosaic(img, boxes)

    # Checks here would depend on the exact expected behavior of the mosaic function,
    # such as dimensions and content of the output image and boxes.

    assert mosaic_img.size == (100, 100), "Mosaic image size should be same"
    assert len(mosaic_boxes) > 0, "Should have some bounding boxes"
