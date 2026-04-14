from typing import List, Tuple

import torch
from torch import Tensor, nn

from yolo.model.blocks.basic import Conv
from yolo.tasks.detection.head import MultiheadDetection


class Segmentation(nn.Module):
    def __init__(self, in_channels: Tuple[int], num_maskes: int):
        super().__init__()
        first_neck, in_channels = in_channels

        mask_neck = max(first_neck // 4, num_maskes)
        self.mask_conv = nn.Sequential(
            Conv(in_channels, mask_neck, 3), Conv(mask_neck, mask_neck, 3), nn.Conv2d(mask_neck, num_maskes, 1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.mask_conv(x)
        return x


class MultiheadSegmentation(nn.Module):
    """Multihead Segmentation module for Dual segment or Triple segment"""

    def __init__(self, in_channels: List[int], num_classes: int, num_maskes: int, **head_kwargs):
        super().__init__()
        mask_channels, proto_channels = in_channels[:-1], in_channels[-1]

        self.detect = MultiheadDetection(mask_channels, num_classes, **head_kwargs)
        self.heads = nn.ModuleList(
            [Segmentation((in_channels[0], in_channel), num_maskes) for in_channel in mask_channels]
        )
        self.heads.append(Conv(proto_channels, num_maskes, 1))

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]
