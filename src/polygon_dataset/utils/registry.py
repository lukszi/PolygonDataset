# polygon_dataset/utils/registry.py
"""
Generic registry pattern implementation.

This module provides a generic registry for registering and retrieving classes
of various types, allowing for dynamic lookup and instantiation.
"""

from typing import Dict, Type, Callable, TypeVar, Generic, List

# Type variable for generic registry
T = TypeVar('T')


class Registry(Generic[T]):
    """
    Generic registry for classes of a specific type.

    This class provides methods for registering, retrieving, and listing
    classes of a specific type.
    """

    def __init__(self, type_name: str):
        """
        Initialize the registry.

        Args:
            type_name: Name of the type being registered (for error messages).
        """
        self._registry: Dict[str, Type[T]] = {}
        self._type_name = type_name

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class in the registry.

        Args:
            name: Name to register the class under.

        Returns:
            Callable: Decorator function that registers the class.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            """
            Inner decorator function that registers the class.

            Args:
                cls: Class to register.

            Returns:
                Type[T]: The registered class (unchanged).

            Raises:
                ValueError: If a class with the same name is already registered.
            """
            if name in self._registry:
                raise ValueError(f"{self._type_name} '{name}' is already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """
        Get a class by name.

        Args:
            name: Name of the class to get.

        Returns:
            Type[T]: The requested class.

        Raises:
            ValueError: If no class is registered with the given name.
        """
        if name not in self._registry:
            raise ValueError(
                f"{self._type_name} '{name}' not found in registry. "
                f"Available {self._type_name.lower()}s: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> Dict[str, Type[T]]:
        """
        Get a dictionary of all registered classes.

        Returns:
            Dict[str, Type[T]]: Dictionary mapping names to classes.
        """
        return self._registry.copy()

    def names(self) -> List[str]:
        """
        Get a list of all registered class names.

        Returns:
            List[str]: List of registered class names.
        """
        return list(self._registry.keys())