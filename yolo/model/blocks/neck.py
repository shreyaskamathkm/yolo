# TODO Phase 2: update imports — Conv, Pool from yolo.model.blocks.basic
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from yolo.model.blocks.basic import Conv, Pool
from yolo.utils.module_utils import auto_pad


class CBLinear(nn.Module):
    """Convolutional block that outputs multiple feature maps split along the channel dimension."""

    def __init__(self, in_channels: int, out_channels: List[int], kernel_size: int = 1, **kwargs):
        super(CBLinear, self).__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = list(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.conv(x)
        return x.split(self.out_channels, dim=1)


class SPPCSPConv(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels: int, out_channels: int, expand: float = 0.5, kernel_sizes: Tuple[int] = (5, 9, 13)):
        super().__init__()
        neck_channels = int(2 * out_channels * expand)
        self.pre_conv = nn.Sequential(
            Conv(in_channels, neck_channels, 1),
            Conv(neck_channels, neck_channels, 3),
            Conv(neck_channels, neck_channels, 1),
        )
        self.short_conv = Conv(in_channels, neck_channels, 1)
        self.pools = nn.ModuleList([Pool(kernel_size=kernel_size, stride=1) for kernel_size in kernel_sizes])
        self.post_conv = nn.Sequential(Conv(4 * neck_channels, neck_channels, 1), Conv(neck_channels, neck_channels, 3))
        self.merge_conv = Conv(2 * neck_channels, out_channels, 1)

    def forward(self, x):
        features = [self.pre_conv(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        features = torch.cat(features, dim=1)
        y1 = self.post_conv(features)
        y2 = self.short_conv(x)
        y = torch.cat((y1, y2), dim=1)
        return self.merge_conv(y)


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: Optional[int] = None):
        super(SPPELAN, self).__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class CBFuse(nn.Module):
    def __init__(self, index: List[int], mode: str = "nearest"):
        super().__init__()
        self.idx = index
        self.mode = mode

    def forward(self, x_list: List[torch.Tensor]) -> List[Tensor]:
        target = x_list[-1]
        target_size = target.shape[2:]

        res = [F.interpolate(x[pick_id], size=target_size, mode=self.mode) for pick_id, x in zip(self.idx, x_list)]
        out = torch.stack(res + [target]).sum(dim=0)
        return out
