from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from yolo.config.config import Config
from yolo.model.builder import create_model
from yolo.utils.logger import logger


class TorchBackend(nn.Module):
    def __init__(self, weight: str, device: str, cfg: Config):
        super().__init__()
        # Safely disable auxiliary heads if they exist in the config
        if hasattr(cfg.model.model, "auxiliary"):
            cfg.model.model.auxiliary = {}

        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=weight).to(device)
        logger.info(":rocket: Using Deploy (stripped auxiliary heads) model!")

    def forward(self, x: Tensor) -> dict:
        return self.model(x)

    def __call__(self, x: Tensor) -> dict:
        return self.forward(x)

    @staticmethod
    def export(model: nn.Module, cfg: Config, output_path: Path) -> None:
        torch.save(model.state_dict(), output_path)
        logger.info(f":inbox_tray: Torch model saved to {output_path}")
