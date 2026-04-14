# TODO Phase 2: update imports — Conv from yolo.model.blocks.basic,
#               RepNCSPELAN from yolo.model.blocks.backbone
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from yolo.model.blocks.backbone import RepNCSPELAN

# Phase 2: replace with: from yolo.model.blocks.basic import Conv
# Phase 2: replace with: from yolo.model.blocks.backbone import RepNCSPELAN
from yolo.model.blocks.basic import Conv


class Anchor2Vec(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, anchor_x: Tensor) -> Tensor:
        anchor_x = rearrange(anchor_x, "B (P R) h w -> B R P h w", P=4)
        vector_x = anchor_x.softmax(dim=1)
        vector_x = self.anc2vec(vector_x)[:, 0]
        return anchor_x, vector_x


class ImplicitA(nn.Module):
    """
    Implement YOLOR - implicit knowledge(Add), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit + x


class ImplicitM(nn.Module):
    """
    Implement YOLOR - implicit knowledge(multiply), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit * x


class DConv(nn.Module):
    def __init__(self, in_channels=512, alpha=0.8, atoms=512):
        super().__init__()
        self.alpha = alpha

        self.CG = Conv(in_channels, atoms, 1)
        self.GIE = Conv(atoms, atoms, 5, groups=atoms, activation=False)
        self.D = Conv(atoms, in_channels, 1, activation=False)

    def PONO(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        return x

    def forward(self, r):
        x = self.CG(r)
        x = self.GIE(x)
        x = self.PONO(x)
        x = self.D(x)
        return self.alpha * x + (1 - self.alpha) * r


class RepNCSPELAND(RepNCSPELAN):
    def __init__(self, *args, atoms: 512, rd_args={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.dconv = DConv(atoms=atoms, **rd_args)

    def forward(self, x):
        x = super().forward(x)
        return self.dconv(x)
