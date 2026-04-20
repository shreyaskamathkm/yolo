from pathlib import Path

import torch
from torch import Tensor, nn

from yolo.config.config import Config
from yolo.utils.logger import logger


class OnnxBackend:
    def __init__(self, weight: str, device: str, cfg: Config):
        from onnxruntime import InferenceSession

        providers = ["CUDAExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.session = InferenceSession(weight, providers=providers)
        self.device = device
        logger.info(":rocket: Using ONNX as MODEL framework!")

    def __call__(self, x: Tensor) -> dict:
        x_np = {self.session.get_inputs()[0].name: x.cpu().numpy()}
        model_outputs, layer_output = [], []
        for idx, predict in enumerate(self.session.run(None, x_np)):
            layer_output.append(torch.from_numpy(predict).to(self.device))
            if idx % 3 == 2:
                model_outputs.append(layer_output)
                layer_output = []
        if len(model_outputs) == 6:
            model_outputs = model_outputs[:3]
        return {"Main": model_outputs}

    @staticmethod
    def export(model: nn.Module, cfg: Config, output_path: Path) -> None:
        from torch.onnx import export

        dummy_input = torch.ones((1, 3, *cfg.image_size))
        export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
        logger.info(f":inbox_tray: ONNX model saved to {output_path}")
