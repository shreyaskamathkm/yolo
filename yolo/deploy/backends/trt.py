from pathlib import Path

import torch
from torch import Tensor, nn

from yolo.config.config import Config
from yolo.utils.logger import logger


class TRTBackend:
    def __init__(self, weight: str, device: str, cfg: Config):
        from torch2trt import TRTModule

        self.model = TRTModule()
        self.model.load_state_dict(torch.load(weight))
        logger.info(":rocket: Using TensorRT as MODEL framework!")

    def __call__(self, x: Tensor) -> dict:
        return self.model(x)

    @staticmethod
    def export(model: nn.Module, cfg: Config, output_path: Path) -> None:
        import torch_tensorrt

        dummy_input = torch.ones((1, 3, *cfg.image_size)).cuda()
        logger.info("♻️ Creating TensorRT model")
        # Move model to GPU and eval mode
        model = model.cuda().eval()
        # Compile with Torch-TensorRT
        model_trt = torch_tensorrt.compile(
            module=model,
            ir="dynamo",
            inputs=[dummy_input],
            enabled_precisions={torch.float32},
        )

        # Save Python runtime format (.ep)
        torch_tensorrt.save(
            model_trt,
            "model.ep",
            inputs=[dummy_input],
        )

        # Save TorchScript format (.ts)
        torch_tensorrt.save(
            model_trt,
            "model.ts",
            output_format="torchscript",
            inputs=[dummy_input],
        )

        logger.info(f"📥 TensorRT model saved to {output_path}")
