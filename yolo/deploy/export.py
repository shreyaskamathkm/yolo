from pathlib import Path

from yolo.config.config import Config
from yolo.deploy.backends.onnx import OnnxBackend
from yolo.deploy.backends.torch import TorchBackend
from yolo.deploy.backends.trt import TRTBackend
from yolo.model.builder import create_model

_EXPORTERS = {
    "torch": TorchBackend,
    "onnx": OnnxBackend,
    "trt": TRTBackend,
}

_EXTENSIONS = {
    "torch": "pt",
    "onnx": "onnx",
    "trt": "trt",
}


class ModelExporter:
    """Manages the conversion of YOLO models to various deployment formats.

    This class handles model initialization, stripping of auxiliary heads,
    and delegation to specific backend exporters (ONNX, TensorRT, Torch).

    Attributes:
        cfg (Config): System configuration containing model and task details.
        model (YOLO): The constructed and evaluated PyTorch model.
    """

    def __init__(self, cfg: Config):
        """Initializes the ModelExporter with the provided configuration.

        Args:
            cfg (Config): Configuration object containing model version, weights path,
                and dataset information.
        """
        self.cfg = cfg
        self.cfg.model.model.auxiliary = {}
        self.model = create_model(
            cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight, weight_key="ema_shadow"
        ).eval()

    def __call__(self) -> None:
        """Executes the export process for all requested formats.

        Reads the formats specified in the configuration, determines the appropriate
        backend for each, and saves the exported model to the output directory.

        Raises:
            ValueError: If an unsupported export format is requested.
        """
        stem = Path(self.cfg.weight).stem
        output_dir = Path(self.cfg.out_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in self.cfg.task.formats:
            backend_cls = _EXPORTERS.get(fmt)
            if backend_cls is None:
                raise ValueError(f"Unknown export format: {fmt!r}. Choose from: {list(_EXPORTERS)}")
            extension = _EXTENSIONS.get(fmt, fmt)
            output_path = output_dir / f"{stem}.{extension}"
            backend_cls.export(self.model, self.cfg, output_path)
