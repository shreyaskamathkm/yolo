import random
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from yolo.config.config import ModelConfig
from yolo.model.builder import YOLO
from yolo.utils.logger import logger


def get_color(idx: int):
    """Generates a consistent, bright color for a given index."""
    random.seed(idx)
    return tuple(random.randint(50, 255) for _ in range(3))


def draw_bboxes(
    img: Union[Image.Image, torch.Tensor],
    bboxes: List[List[Union[int, float]]],
    *,
    idx2label: Optional[list] = None,
) -> Image.Image:
    """Draws bounding boxes and labels onto an image with a premium look.

    Args:
        img (Union[Image.Image, torch.Tensor]): The input image.
        bboxes (List[List[Union[int, float]]]): A list of bounding boxes in
            `[class_id, x_min, y_min, x_max, y_max, (optional) confidence]` format.
        idx2label (Optional[list], optional): A list mapping class IDs to human-readable
            labels.

    Returns:
        Image.Image: The image with boxes and labels drawn.
    """

    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        img = to_pil_image(img.cpu())

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().tolist()

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    font_size = max(12, int(min(width, height) * 0.02))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for bbox in bboxes:
        if len(bbox) < 5:
            continue
        class_id, x1, y1, x2, y2, *conf = [float(val) for val in bbox]
        if class_id < 0:
            continue

        color = get_color(int(class_id))
        # Draw solid outline on the main image for sharpness
        outline_draw = ImageDraw.Draw(img)
        outline_draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)

        # Draw translucent fill on the overlay
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 60))

        class_name = idx2label[int(class_id)] if idx2label else f"ID {int(class_id)}"
        label = f"{class_name}" + (f" {conf[0]:.2f}" if conf else "")

        # Draw label box on the main image
        tw, th = outline_draw.textbbox((0, 0), label, font=font)[2:]
        outline_draw.rectangle([x1, y1 - th, x1 + tw + 4, y1], fill=(*color, 255))
        outline_draw.text((x1 + 2, y1 - th), label, fill="white", font=font)

    # Combine the main image and the translucent overlay
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def draw_masks(
    img: Union[Image.Image, torch.Tensor],
    masks: List[torch.Tensor],
    *,
    idx2label: Optional[list] = None,
    alpha: float = 0.4,
) -> Image.Image:
    """Draws segmentation masks and labels with a premium MMDetection-style look.

    Args:
        img (Union[Image.Image, torch.Tensor]): The input image.
        masks (List[torch.Tensor]): List of tensors, each [class_id, x1, y1, x2, y2, ...].
        idx2label (Optional[list]): Mapping from class ID to label name.
        alpha (float): Transparency of the mask overlay.

    Returns:
        Image.Image: The image with masks and labels drawn.
    """
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        img = to_pil_image(img.cpu())

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_draw = ImageDraw.Draw(img)

    width, height = img.size
    font_size = max(14, int(min(width, height) * 0.025))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for mask in masks:
        if mask.numel() < 3:
            continue
        mask = mask.cpu()
        class_id = int(mask[0])
        points = mask[1:].reshape(-1, 2).tolist()
        points = [(p[0] * width, p[1] * height) if max(p) <= 1.01 else (p[0], p[1]) for p in points]

        if len(points) < 3:
            continue

        color = get_color(class_id)

        # Draw semi-transparent mask
        draw.polygon(points, fill=(*color, int(255 * alpha)))

        # Draw sharp white outline for that premium look
        outline_draw.polygon(points, outline=(255, 255, 255, 200), width=2)

        # Draw label at the top-most point of the mask
        min_y = min(p[1] for p in points)
        top_point = sorted([p for p in points if p[1] == min_y], key=lambda x: x[0])[0]

        class_name = idx2label[int(class_id)] if idx2label else f"ID {int(class_id)}"
        tw, th = outline_draw.textbbox((0, 0), class_name, font=font)[2:]

        # Label background
        lx, ly = top_point[0], top_point[1] - th - 2
        outline_draw.rectangle([lx, ly, lx + tw + 6, ly + th + 2], fill=(0, 0, 0, 180))
        outline_draw.text((lx + 3, ly), class_name, fill="white", font=font)

    # Combine overlay and original image
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")



def draw_model(*, model_cfg: ModelConfig = None, model: YOLO = None, v7_base=False):
    """Generates a graphviz visualization of the model architecture.

    Args:
        model_cfg (Optional[ModelConfig]): Configuration to build a model from.
        model (Optional[YOLO]): An existing YOLO model instance.
        v7_base (bool): Whether to simplify the graph using YOLOv7-specific patterns.

    Note:
        Requires the `graphviz` library and system backend.
    """

    from graphviz import Digraph

    if model_cfg:
        from yolo.model.builder import create_model

        model = create_model(model_cfg)
    elif model is None:
        raise ValueError("Drawing Object is None")

    model_size = len(model.model) + 1
    model_mat = np.zeros((model_size, model_size), dtype=bool)

    layer_name = ["INPUT"]
    for idx, layer in enumerate(model.model, start=1):
        layer_name.append(str(type(layer)).split(".")[-1][:-2])
        if layer.tags is not None:
            layer_name[-1] = f"{layer.tags}-{layer_name[-1]}"
        if isinstance(layer.source, int):
            source = layer.source + (layer.source < 0) * idx
            model_mat[source, idx] = True
        else:
            for source in layer.source:
                source = source + (source < 0) * idx
                model_mat[source, idx] = True

    pattern_mat = []
    if v7_base:
        pattern_list = [("ELAN", 8, 3), ("ELAN", 8, 55), ("MP", 5, 11)]
        for name, size, position in pattern_list:
            pattern_mat.append(
                (name, size, model_mat[position : position + size, position + 1 : position + 1 + size].copy())
            )

    dot = Digraph(comment="Model Flow Chart")
    node_idx = 0

    for idx in range(model_size):
        for jdx in range(idx, model_size - 7):
            for name, size, pattern in pattern_mat:
                if (model_mat[idx : idx + size, jdx : jdx + size] == pattern).all():
                    layer_name[idx] = name
                    model_mat[idx : idx + size, jdx : jdx + size] = False
                    model_mat[idx, idx + size] = True
        dot.node(str(idx), f"{layer_name[idx]}")
        node_idx += 1
        for jdx in range(idx, model_size):
            if model_mat[idx, jdx]:
                dot.edge(str(idx), str(jdx))
    try:
        dot.render("Model-arch", format="png", cleanup=True)
        logger.info(":artist_palette: Drawing Model Architecture at Model-arch.png")
    except:
        logger.warning(":warning: Could not find graphviz backend, continue without drawing the model architecture")
