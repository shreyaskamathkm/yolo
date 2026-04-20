from typing import Protocol

from torch import Tensor


class InferenceBackend(Protocol):
    def __call__(self, x: Tensor) -> dict: ...
