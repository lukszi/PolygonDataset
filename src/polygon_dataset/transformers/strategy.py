from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np


class TransformerStrategy(ABC):
    """
    Abstract base class for polygon transformation strategies.

    A strategy defines the core transformation algorithm without handling
    infrastructure concerns like parallelization or file I/O.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the transformation strategy.

        Args:
            config: Configuration parameters for the strategy.
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @abstractmethod
    def transform_polygon(
            self,
            polygon: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Transform a single polygon.

        Args:
            polygon: Array of polygon vertices with shape [vertices, 2].
            resolutions: List of target resolutions (for simplification algorithms).
            **kwargs: Additional parameters specific to the strategy.

        Returns:
            Transformed polygon, either as a single array or a dictionary mapping
            resolutions to arrays.
        """
        pass

    def prepare_input(self, polygons: np.ndarray) -> np.ndarray:
        """
        Validate and prepare input polygons.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].

        Returns:
            Validated and prepared polygons.

        Raises:
            ValueError: If the input shape is invalid.
        """
        # Validate input shape
        if len(polygons.shape) != 3 or polygons.shape[2] != 2:
            raise ValueError(f"Invalid polygon data shape: {polygons.shape}. Expected (N, V, 2)")

        return polygons

    def get_output_type(self) -> str:
        """
        Get the type of output produced by this strategy.

        Returns:
            str: One of "single" (single array) or "multi" (resolution dictionary).
        """
        return "single"  # Default to single output

    def filter_resolutions(self, resolutions: List[int], vertex_count: int) -> List[int]:
        """
        Filter resolutions that exceed the vertex count.

        Args:
            resolutions: List of target resolutions.
            vertex_count: Maximum vertex count.

        Returns:
            Filtered list of valid resolutions.
        """
        return [r for r in resolutions if r <= vertex_count]