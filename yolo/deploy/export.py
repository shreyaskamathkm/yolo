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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.model.model.auxiliary = {}
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight).eval()

    def __call__(self) -> None:
        stem = Path(self.cfg.weight).stem
        output_dir = Path(self.cfg.task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in self.cfg.task.formats:
            backend_cls = _EXPORTERS.get(fmt)
            if backend_cls is None:
                raise ValueError(f"Unknown export format: {fmt!r}. Choose from: {list(_EXPORTERS)}")
            extension = _EXTENSIONS.get(fmt, fmt)
            output_path = output_dir / f"{stem}.{extension}"
            backend_cls.export(self.model, self.cfg, output_path)
