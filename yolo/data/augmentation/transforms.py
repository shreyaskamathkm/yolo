from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

from yolo.registry import TRANSFORMS


class BaseTransform(abc.ABC):
    """Abstract base for all image and label transformations.

    Provides common utilities for cloning and filtering boxes and masks.
    """

    @abc.abstractmethod
    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        """
        Args:
            image: Input PIL Image.
            boxes: [N, 5] box tensor (class_id, x_min, y_min, x_max, y_max) in [0, 1].
            masks: Optional list of N polygon tensors [class_id, x1, y1, x2, y2, ...].
        """
        ...

    @staticmethod
    def _clone_boxes(boxes: Tensor) -> Tensor:
        """Return a detached clone so we never mutate the caller's tensor."""
        return boxes.clone()

    @staticmethod
    def _clone_masks(masks: Optional[List[Tensor]]) -> Optional[List[Tensor]]:
        if masks is None:
            return None
        return [m.clone() for m in masks]

    @staticmethod
    def _filter_boxes_and_masks(
        boxes: Tensor,
        masks: Optional[List[Tensor]],
        min_area: float = 0.0,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Remove degenerate boxes and their corresponding masks in sync."""
        if boxes.numel() == 0:
            return boxes, ([] if masks is not None else None)

        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        keep_mask = (w > 0) & (h > 0) & (w * h > min_area)
        keep_idx = keep_mask.nonzero(as_tuple=False).view(-1).tolist()

        filtered_boxes = boxes[keep_mask]
        filtered_masks: Optional[List[Tensor]] = None
        if masks is not None:
            if len(masks) == len(boxes):
                filtered_masks = [masks[i] for i in keep_idx]
            else:
                import warnings

                warnings.warn(
                    f"masks length ({len(masks)}) != boxes length ({len(boxes)}); "
                    "masks will not be filtered in sync with boxes.",
                    stacklevel=3,
                )
                filtered_masks = masks

        return filtered_boxes, filtered_masks

    @staticmethod
    def _is_valid_mask(mask: Tensor) -> bool:
        """Return True if the polygon mask spans a non-zero area."""
        if mask.numel() < 7:  # [class, x1, y1, x2, y2, x3, y3]
            return False
        xs, ys = mask[1::2], mask[2::2]
        return bool((xs.max() - xs.min()) > 0 and (ys.max() - ys.min()) > 0)


@TRANSFORMS.register_module()
class RemoveOutliers(BaseTransform):
    """Removes bounding boxes that are smaller than a specified minimum area.

    Args:
        min_box_area: The minimum area a box must have to be kept.
                      Boxes with area <= min_box_area will be removed.
                      Default is 1e-8.
    """

    def __init__(self, min_box_area: float = 1e-8):
        self.min_box_area = min_box_area

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        boxes = self._clone_boxes(boxes)
        masks = self._clone_masks(masks)
        boxes, masks = self._filter_boxes_and_masks(boxes, masks, self.min_box_area)
        return image, boxes, masks


@TRANSFORMS.register_module()
class PadAndResize(BaseTransform):
    """Letterbox-resizes the image to a target size while maintaining aspect ratio.

    The image is scaled to fit within the target dimensions, and any remaining
    space is padded with a background color. Bounding boxes and masks are
    adjusted accordingly.

    Args:
        image_size: Target (width, height) of the resized image.
        background_color: RGB color used for padding (default is gray [114, 114, 114]).
    """

    def __init__(self, image_size: Tuple[int, int], background_color: Tuple[int, int, int] = (114, 114, 114)):
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ):
        boxes = self._clone_boxes(boxes)
        masks = self._clone_masks(masks)

        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        if boxes.numel() > 0:
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
            boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        if masks is not None:
            for mask in masks:
                mask[1::2] = (mask[1::2] * new_width + pad_left) / self.target_width
                mask[2::2] = (mask[2::2] * new_height + pad_top) / self.target_height

        pad_right = self.target_width - new_width - pad_left
        pad_bottom = self.target_height - new_height - pad_top
        transform_info = torch.tensor([scale, pad_left, pad_top, pad_right, pad_bottom], dtype=torch.float32)
        return padded_image, boxes, masks, transform_info


@TRANSFORMS.register_module()
class HorizontalFlip(BaseTransform):
    """Randomly flips the image, bounding boxes, and masks horizontally.

    Args:
        prob: Probability of applying the flip. Default is 0.5.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        if torch.rand(1).item() < self.prob:
            boxes = self._clone_boxes(boxes)
            masks = self._clone_masks(masks)
            image = TF.hflip(image)
            if boxes.numel() > 0:
                x_cols = boxes[:, [3, 1]].clone()
                boxes[:, [1, 3]] = 1.0 - x_cols
            if masks is not None:
                for mask in masks:
                    mask[1::2] = 1.0 - mask[1::2]
        return image, boxes, masks


@TRANSFORMS.register_module()
class VerticalFlip(BaseTransform):
    """Randomly flips the image, bounding boxes, and masks vertically.

    Args:
        prob: Probability of applying the flip. Default is 0.5.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        if torch.rand(1).item() < self.prob:
            boxes = self._clone_boxes(boxes)
            masks = self._clone_masks(masks)
            image = TF.vflip(image)
            if boxes.numel() > 0:
                y_cols = boxes[:, [4, 2]].clone()
                boxes[:, [2, 4]] = 1.0 - y_cols
            if masks is not None:
                for mask in masks:
                    mask[2::2] = 1.0 - mask[2::2]
        return image, boxes, masks


@TRANSFORMS.register_module()
class Mosaic(BaseTransform):
    """Combines four images into a single 2x2 grid (Mosaic augmentation).

    This transform requires a parent dataset to provide additional samples.
    The center of the mosaic grid is randomly selected, and images are
    pasted around it. Bounding boxes and masks are filtered and adjusted.

    Requires a parent dataset with:
        get_more_data(n) -> List[Tuple[PIL.Image, Tensor, Optional[List[Tensor]]]]
        base_size: The target output size for the mosaic components.

    Args:
        prob: Probability of applying Mosaic augmentation. Default is 0.5.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent
        return self

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        if torch.rand(1).item() >= self.prob:
            return image, boxes, masks

        if self.parent is None:
            raise RuntimeError("Parent is not set. Mosaic cannot retrieve data.")

        boxes = self._clone_boxes(boxes)
        masks = self._clone_masks(masks)

        img_sz = self.parent.base_size
        more_data = self.parent.get_more_data(3)

        samples = [(image, boxes, masks)] + more_data

        has_masks = any(msks is not None for _, _, msks in samples)
        if has_masks and any(msks is None for _, _, msks in samples):
            raise RuntimeError("Mosaic dataset consistency issue: some samples have masks while others do not.")
        all_masks: Optional[List[Tensor]] = [] if has_masks else None

        vectors = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz), (114, 114, 114))
        center = np.array([img_sz, img_sz])

        all_boxes = []
        for (img, bxs, msks), (vx, vy) in zip(samples, vectors):
            w, h = img.size
            paste_x, paste_y = int(center[0] + vx * w), int(center[1] + vy * h)
            mosaic_image.paste(img, (paste_x, paste_y))

            bxs = self._clone_boxes(bxs)
            if bxs.numel() > 0:
                bxs[:, [1, 3]] = (bxs[:, [1, 3]] * w + paste_x) / (2 * img_sz)
                bxs[:, [2, 4]] = (bxs[:, [2, 4]] * h + paste_y) / (2 * img_sz)
                bxs[:, 1:] = bxs[:, 1:].clamp(0.0, 1.0)

            remapped_msks = None
            if msks is not None:
                remapped_msks = []
                for msk in msks:
                    msk = msk.clone()
                    msk[1::2] = (msk[1::2] * w + paste_x) / (2 * img_sz)
                    msk[2::2] = (msk[2::2] * h + paste_y) / (2 * img_sz)
                    msk[1::2] = msk[1::2].clamp(0.0, 1.0)
                    msk[2::2] = msk[2::2].clamp(0.0, 1.0)
                    remapped_msks.append(msk)

            bxs, remapped_msks = self._filter_boxes_and_masks(bxs, remapped_msks)

            if remapped_msks is not None:
                valid_indices = [i for i, m in enumerate(remapped_msks) if self._is_valid_mask(m)]
                bxs = bxs[valid_indices]
                remapped_msks = [remapped_msks[i] for i in valid_indices]

            all_boxes.append(bxs)
            if all_masks is not None and remapped_msks is not None:
                all_masks.extend(remapped_msks)

        mosaic_image = mosaic_image.resize((img_sz, img_sz), Image.Resampling.LANCZOS)

        if sum(b.shape[0] for b in all_boxes) == 0:
            final_boxes = torch.zeros((0, 5), dtype=boxes.dtype, device=boxes.device)
        else:
            final_boxes = torch.cat(all_boxes)

        return mosaic_image, final_boxes, (all_masks if all_masks else None)


@TRANSFORMS.register_module()
class MixUp(BaseTransform):
    """Blends two images using a Beta-distributed mixing coefficient.

    Pixel values are blended as: mixed = lam * image1 + (1 - lam) * image2.

    Labels (boxes and masks) from BOTH images are kept regardless of lam,
    following the standard MixUp treatment for object detection — all objects
    in both images remain valid annotations for the blended result.

    Requires a parent dataset with:
        get_more_data(n) -> List[Tuple[PIL.Image, Tensor, Optional[List[Tensor]]]]

    Args:
        prob:  Probability of applying MixUp. Default 0.5.
        alpha: Beta distribution concentration parameter (must be > 0).
               Higher values push lam toward 0.5 (equal blend).
               Typical values: 0.2 (weak) to 1.5 (strong).
    """

    def __init__(self, prob: float = 0.5, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive for the Beta distribution, got {alpha}.")
        self.prob = prob
        self.alpha = alpha
        self.parent = None

    def set_parent(self, parent) -> "MixUp":
        self.parent = parent
        return self

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        if torch.rand(1).item() >= self.prob:
            return image, boxes, masks

        if self.parent is None:
            raise RuntimeError("Parent is not set. Call set_parent(dataset) before using MixUp.")

        img2, boxes2, masks2 = self.parent.get_more_data(1)[0]

        # Clone inputs so we never mutate the caller's tensors
        boxes = self._clone_boxes(boxes)
        masks = self._clone_masks(masks)
        boxes2 = self._clone_boxes(boxes2)
        masks2 = self._clone_masks(masks2)

        if (masks is None) != (masks2 is None):
            raise RuntimeError(
                "MixUp dataset consistency error: one sample has masks while the other does not. "
                "Either all samples must have masks or none should."
            )

        if img2.mode != image.mode:
            img2 = img2.convert(image.mode)
        if img2.size != image.size:
            img2 = img2.resize(image.size, Image.Resampling.BILINEAR)

        lam = float(np.random.beta(self.alpha, self.alpha))
        mixed_img = TF.to_pil_image(lam * TF.to_tensor(image) + (1.0 - lam) * TF.to_tensor(img2))

        merged_boxes = torch.cat([boxes, boxes2])

        if masks is None:
            merged_masks = None
        else:
            merged_masks = masks + masks2

        merged_boxes, merged_masks = self._filter_boxes_and_masks(merged_boxes, merged_masks)

        return mixed_img, merged_boxes, (merged_masks if merged_masks else None)


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Randomly crops the image to a smaller size.

    The current implementation crops the image to half of its original dimensions.
    Bounding boxes and masks are adjusted to the new crop and filtered if they
    fall outside the cropped area.

    Args:
        prob: Probability of applying the crop. Default is 0.5.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Image.Image, Tensor, Optional[List[Tensor]]]:
        if torch.rand(1).item() < self.prob:
            boxes, masks = self._clone_boxes(boxes), self._clone_masks(masks)
            orig_w, orig_h = image.size
            crop_w, crop_h = orig_w // 2, orig_h // 2
            top = torch.randint(0, orig_h - crop_h + 1, (1,)).item()
            left = torch.randint(0, orig_w - crop_w + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_h, crop_w)
            if boxes.numel() > 0:
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] * orig_w - left).clamp(0.0, crop_w) / crop_w
                boxes[:, [2, 4]] = (boxes[:, [2, 4]] * orig_h - top).clamp(0.0, crop_h) / crop_h

            if masks is not None:
                for mask in masks:
                    mask[1::2] = (mask[1::2] * orig_w - left).clamp(0.0, crop_w) / crop_w
                    mask[2::2] = (mask[2::2] * orig_h - top).clamp(0.0, crop_h) / crop_h

            boxes, masks = self._filter_boxes_and_masks(boxes, masks)

        return image, boxes, masks


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Composes several transforms together and applies final resizing/tensor conversion.

    Args:
        transforms: List of dictionaries or BaseTransform objects.
        image_size: Target (width, height) for the final PadAndResize.
    """

    def __init__(
        self,
        transforms: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (640, 640),
    ):
        self.image_size = image_size
        self.base_size = int(sum(image_size) / len(image_size))

        self.transforms = []
        for transform_cfg in transforms:
            if isinstance(transform_cfg, (DictConfig, ListConfig)):
                cfg = OmegaConf.to_container(transform_cfg, resolve=True)
            elif isinstance(transform_cfg, dict):
                cfg = transform_cfg.copy()
            else:
                raise TypeError(f"Expected dict or OmegaConf config, got {type(transform_cfg)}")

            t_type = cfg.pop("type")
            transform_cls = TRANSFORMS.get(t_type)
            if transform_cls is None:
                raise ValueError(f"Transform '{t_type}' not found in registry.")
            self.transforms.append(transform_cls(**cfg))

        self.pad_resize = PadAndResize(image_size)

        # Propagate self as parent for transforms that need it
        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def set_parent(self, parent):
        """Allow nesting if necessary, though root Compose usually acts as parent."""
        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(parent)
        return self

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[List[Tensor]], Tensor]:
        """
        Returns:
            Tuple of (processed_image_tensor, boxes, masks, transformation_info).
        """
        for transform in self.transforms:
            image, boxes, masks = transform(image, boxes, masks)

        # Final padding/resize returns (image, boxes, masks, transform_info)
        image, boxes, masks, rev_tensor = self.pad_resize(image, boxes, masks)
        image = TF.to_tensor(image)
        return image, boxes, masks, rev_tensor
