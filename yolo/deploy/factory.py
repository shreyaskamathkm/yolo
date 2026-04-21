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
    """Factory function to create an optimized inference backend.

    Args:
        backend (str): The desired backend type ('onnx', 'trt', or 'torch').
        weight (str): Path to the model weights file (.onnx, .trt, or .pt).
        device (str): Device to run inference on (e.g., 'cuda', 'cpu', 'mps').
        cfg (Config): System configuration.

    Returns:
        InferenceBackend: An initialized backend instance.

    Raises:
        ValueError: If an unsupported backend type is requested.
    """
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {list(_BACKENDS)}")
    return cls(weight, device, cfg)
