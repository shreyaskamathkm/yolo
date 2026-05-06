"""Centralized registry system for the YOLO repository.

This module provides the Registry engine, global instances for components,
and decorators for task-specific registration.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Type
from yolo.data.schema import DataSplitType, TaskMode

from yolo.data.schema import DataSplitType

if TYPE_CHECKING:
    from lightning import LightningModule


class Registry:
    """A registry to map strings (or keys) to classes.

    This class provides a centralized mechanism for registering and retrieving
    classes or objects, typically used for dynamic instantiation from
    configuration files. It supports decorator-based registration and
    dictionary-like access.

    Attributes:
        name (str): The name of the registry.
    """

    def __init__(self, name: str):
        """Initializes the registry.

        Args:
            name (str): Unique name for this registry.
        """
        self._name = name
        self._module_dict: Dict[Any, Type] = {}

    @property
    def name(self) -> str:
        """str: The name of the registry."""
        return self._name

    @property
    def module_dict(self) -> Dict[Any, Type]:
        """Dict[Any, Type]: The underlying dictionary of registered modules."""
        return self._module_dict

    def register_module(self, name: Optional[str] = None):
        """Decorator for registering a module.

        Args:
            name (str, optional): The name to register the module under.
                If not provided, the class's `__name__` will be used.

        Returns:
            callable: The decorator function.

        Raises:
            KeyError: If the name is already registered in this registry.
        """

        def _register(cls: Type) -> Type:
            module_name = name if name else cls.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self.name}")
            self._module_dict[module_name] = cls
            return cls

        return _register

    def register(self, key: Any, cls: Type):
        """Explicitly registers a module with a specific key.

        Args:
            key (Any): The key to register the class under.
            cls (Type): The class or object to register.

        Raises:
            KeyError: If the key is already registered in this registry.
        """
        if key in self._module_dict:
            raise KeyError(f"{key} is already registered in {self.name}")
        self._module_dict[key] = cls

    def get(self, key: Any) -> Optional[Type]:
        """Retrieves a registered module by its key.

        Args:
            key (Any): The key to look up.

        Returns:
            Optional[Type]: The registered class, or None if not found.
        """
        return self._module_dict.get(key)

    def __getitem__(self, key: Any) -> Type:
        """Retrieves a registered module using dictionary-style access.

        Args:
            key (Any): The key to look up.

        Returns:
            Type: The registered class.

        Raises:
            KeyError: If the key is not found.
        """
        return self._module_dict[key]

    def __contains__(self, key: Any) -> bool:
        """Checks if a key is present in the registry."""
        return key in self._module_dict

    def __iter__(self):
        """Returns an iterator over the registry keys."""
        return iter(self._module_dict)

    def __len__(self):
        """Returns the number of registered items."""
        return len(self._module_dict)

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"


# --- Global Registry Instances ---

# Model architectures (e.g. YOLO)
MODELS = Registry("models")

# Neural network layers and blocks (e.g. Conv, Bottleneck, ELAN)
BLOCKS = Registry("blocks")

# Loss functions (e.g. YOLOLoss, SegmentationLoss)
LOSSES = Registry("losses")

# Data transformations and augmentations (e.g. Mosaic, MixUp)
TRANSFORMS = Registry("transforms")

# Datasets (e.g. COCO, YOLO)
DATASETS = Registry("datasets")

# Task solvers / LightningModules (e.g. DetectionTrainModel)
SOLVERS = Registry("solvers")

# Trainer method mapping
TRAINER_METHODS = {
    TaskMode.TRAIN: "fit",
    TaskMode.VAL: "validate",
    TaskMode.INFERENCE: "predict",
}
