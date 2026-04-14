from typing import Tuple

from torch import Tensor, nn

from yolo.model.blocks.basic import Conv


class Classification(nn.Module):
    def __init__(self, in_channel: int, num_classes: int, *, neck_channels=1024, **head_args):
        super().__init__()
        self.conv = Conv(in_channel, neck_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(neck_channels, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.pool(self.conv(x))
        x = self.head(x.flatten(start_dim=1))
        return x
