from typing import Any, Dict, Optional, Type


class Registry:
    """A registry to map strings to classes.

    Used to dynamically create objects from strings in configuration.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

    def register_module(self, name: Optional[str] = None):
        """Registers a module.

        Args:
            name (str, optional): The module name to be registered.
                If not specified, the class name will be used.
        """

        def _register(cls: Type) -> Type:
            module_name = name if name else cls.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self.name}")
            self._module_dict[module_name] = cls
            return cls

        return _register

    def get(self, key: str) -> Optional[Type]:
        """Retrieves a module by name."""
        return self._module_dict.get(key)

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"
