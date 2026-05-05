from typing import Dict, Tuple, Type

from lightning import LightningModule

SOLVERS: Dict[Tuple[str, str], Type[LightningModule]] = {}
LOSS_FUNCTIONS: Dict[Tuple[str, str], Type] = {}

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


def register_loss(task_type: str, loss_name: str):
    """Decorator that registers a loss function class for a (task_type, loss_name) pair.

    Args:
        task_type (str): Task name, e.g. ``"detection"``, ``"segmentation"``.
        loss_name (str): Loss name, e.g. ``"dual"``, ``"single"``.
    """

    def decorator(cls: Type) -> Type:
        LOSS_FUNCTIONS[(task_type, loss_name)] = cls
        return cls

    return decorator
