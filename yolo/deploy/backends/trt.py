from pathlib import Path

import torch
from torch import Tensor, nn

from yolo.config.config import Config
from yolo.utils.logger import logger


class DeployWrapper(nn.Module):
    """
    Ensures model outputs are a flat tuple of tensors,
    satisfying strict TorchScript and TensorRT requirements.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _flatten(self, x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, dict):
            return [item for v in x.values() for item in self._flatten(v)]
        if isinstance(x, (list, tuple)):
            return [item for v in x for item in self._flatten(v)]
        return []

    def forward(self, x: Tensor):
        output = self.model(x)
        return tuple(self._flatten(output))


class TRTBackend:
    def __init__(self, weight: str, device: str, cfg: Config):
        import torch_tensorrt

        # Load the TensorRT-compiled Executable Program (.ep)
        self.model = torch_tensorrt.load(weight).module()
        self.device = device
        logger.info(":rocket: Using TensorRT as MODEL framework!")

    def __call__(self, x: Tensor) -> dict:
        results = self.model(x)
        model_outputs, layer_output = [], []
        # Group the flat results into 3s (cls, anc, box) for each layer
        for idx, predict in enumerate(results):
            layer_output.append(predict)
            if idx % 3 == 2:
                model_outputs.append(layer_output)
                layer_output = []
        # Return in the format expected by PostProcess
        return {"Main": model_outputs}

    @staticmethod
    def export(model: nn.Module, cfg: Config, output_path: Path) -> None:
        import torch_tensorrt

        dummy_input = torch.ones((1, 3, *cfg.image_size)).cuda()
        logger.info("♻️ Creating TensorRT model")
        # Wrap model to ensure tuple output (for compiler compatibility)
        model = DeployWrapper(model)
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
            output_path.with_suffix(".ep"),
            inputs=[dummy_input],
        )

        # Save TorchScript format (.ts)
        torch_tensorrt.save(
            model_trt,
            output_path.with_suffix(".ts"),
            output_format="torchscript",
            inputs=[dummy_input],
        )

        logger.info(f"📥 TensorRT model saved to {output_path}")
