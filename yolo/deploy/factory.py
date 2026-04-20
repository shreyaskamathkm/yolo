from yolo.config.config import Config
from yolo.deploy.backends.onnx import OnnxBackend
from yolo.deploy.backends.torch import TorchBackend
from yolo.deploy.backends.trt import TRTBackend
from yolo.deploy.protocol import InferenceBackend

_BACKENDS = {
    "onnx": OnnxBackend,
    "trt": TRTBackend,
    "torch": TorchBackend,
}


def create_inference_backend(backend: str, weight: str, device: str, cfg: Config) -> InferenceBackend:
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {list(_BACKENDS)}")
    return cls(weight, device, cfg)
