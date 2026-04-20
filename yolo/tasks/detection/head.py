from typing import Any, List, Tuple

import torch
from torch import nn

from yolo.model.blocks.basic import Conv
from yolo.model.blocks.implicit import Anchor2Vec, ImplicitA, ImplicitM
from yolo.utils.module_utils import round_up


class Detection(nn.Module):
    """A single YOLO Prediction Head.

    Decouples the prediction into classification and box regression branches.
    Uses Anchor2Vec to convert raw regression outputs into vector formats.
    """

    def __init__(self, in_channels: Tuple[int], num_classes: int, *, reg_max: int = 16, use_group: bool = True):
        """Initializes the Detection head.

        Args:
            in_channels (Tuple[int]): Number of input channels.
            num_classes (int): Number of target classes.
            reg_max (int, optional): Maximum regression distance. Defaults to 16.
            use_group (bool, optional): Whether to use grouped convolutions for Efficiency.
                Defaults to True.
        """
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max

        first_neck, in_channels = in_channels
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv(in_channels, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv(in_channels, class_neck, 3), Conv(class_neck, class_neck, 3), nn.Conv2d(class_neck, num_classes, 1)
        )

        self.anc2vec = Anchor2Vec(reg_max=reg_max)

        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)

    def forward(self, x) -> Tuple:
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)
        return class_x, anchor_x, vector_x


class IDetection(nn.Module):
    """A YOLOv7-style Implicit Detection head.

    Uses implicit addition and multiplication layers to refine predictions.
    """

    def __init__(self, in_channels: Tuple[int], num_classes: int, *args, anchor_num: int = 3, **kwargs):
        """Initializes the IDetection head.

        Args:
            in_channels (Tuple[int]): Number of input channels.
            num_classes (int): Number of target classes.
            anchor_num (int, optional): Number of anchors per scale. Defaults to 3.
        """
        super().__init__()

        if isinstance(in_channels, tuple):
            in_channels = in_channels[1]

        out_channel = num_classes + 5
        out_channels = out_channel * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x):
        x = self.implicit_a(x)
        x = self.head_conv(x)
        x = self.implicit_m(x)
        return x


class MultiheadDetection(nn.Module):
    """Module that manages multiple prediction heads for different scales.

    Automatically handles the instantiation of plain Detection or implicit
    IDetection heads based on the provided configuration.
    """

    def __init__(self, in_channels: List[int], num_classes: int, **head_kwargs):
        """Initializes the MultiheadDetection module.

        Args:
            in_channels (List[int]): List of input channel counts for each scale.
            num_classes (int): Number of target classes.
            **head_kwargs: Additional arguments passed to each detection head.
        """
        super().__init__()
        DetectionHead = Detection

        if head_kwargs.pop("version", None) == "v7":
            DetectionHead = IDetection

        self.heads = nn.ModuleList(
            [DetectionHead((in_channels[0], in_channel), num_classes, **head_kwargs) for in_channel in in_channels]
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]
