from typing import Dict, Tuple, Type

from lightning import LightningModule

SOLVERS: Dict[Tuple[str, str], Type[LightningModule]] = {}

TRAINER_METHODS = {
    "train": "fit",
    "validation": "validate",
    "inference": "predict",
}


def register(task_type: str, mode: str):
    """Decorator that registers a solver class for a (task_type, mode) pair.

    Args:
        task_type (str): Task name, e.g. ``"detection"``, ``"segmentation"``.
        mode (str): Run mode — ``"train"``, ``"validation"``, or ``"inference"``.

    Example:
        ```python
        @register("detection", "train")
        class DetectionTrainModel(BaseModel): ...
        ```
    """

    def decorator(cls: Type[LightningModule]) -> Type[LightningModule]:
        SOLVERS[(task_type, mode)] = cls
        return cls

    return decorator
