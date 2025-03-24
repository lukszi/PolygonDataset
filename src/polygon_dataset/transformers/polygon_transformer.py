# polygon_dataset/transformers/polygon_transformer.py
"""
Core polygon transformation functionality.

This module provides the PolygonTransformer class, which focuses on applying
transformation strategies to polygon data without handling file I/O or parallelization.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from polygon_dataset.utils import calculate_resolution_steps
from polygon_dataset.transformers.strategy import TransformerStrategy


class PolygonTransformer:
    """
    Handles core polygon transformation functionality.

    This class focuses on applying transformation strategies to polygon data,
    without handling file I/O or parallelization concerns.
    """

    def __init__(
            self,
            strategy: TransformerStrategy,
            min_vertices: int = 10
    ) -> None:
        """
        Initialize the polygon transformer.

        Args:
            strategy: The transformation strategy to use.
            min_vertices: Minimum number of vertices (for simplification).
        """
        self.strategy = strategy
        self.min_vertices = min_vertices

    def transform(
            self,
            polygons: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Transform a batch of polygons using the strategy.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            resolutions: List of target resolutions (for simplification algorithms).
            **kwargs: Additional parameters passed to the strategy.

        Returns:
            Transformed polygons, either as a single array or a dictionary mapping
            resolutions to arrays.

        Raises:
            ValueError: If the input shape is invalid.
        """
        # Prepare the input
        polygons = self.strategy.prepare_input(polygons)

        # Calculate resolutions if not provided and needed
        if resolutions is None and self.strategy.get_output_type() == "multi":
            vertex_count = polygons.shape[1]
            resolutions = calculate_resolution_steps(vertex_count, self.min_vertices)

        # Process each polygon sequentially
        if self.strategy.get_output_type() == "single":
            return self._transform_single_output(polygons, **kwargs)
        else:
            return self._transform_multi_output(polygons, resolutions, **kwargs)

    def _transform_single_output(
            self,
            polygons: np.ndarray,
            **kwargs: Any
    ) -> np.ndarray:
        """
        Transform polygons with a strategy that produces a single output per polygon.

        Args:
            polygons: Array of polygons.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Array of transformed polygons.
        """
        result = np.zeros_like(polygons)
        for i, polygon in enumerate(polygons):
            result[i] = self.strategy.transform_polygon(polygon, **kwargs)
        return result

    def _transform_multi_output(
            self,
            polygons: np.ndarray,
            resolutions: List[int],
            **kwargs: Any
    ) -> Dict[int, np.ndarray]:
        """
        Transform polygons with a strategy that produces multiple outputs per polygon.

        Args:
            polygons: Array of polygons.
            resolutions: List of target resolutions.
            **kwargs: Additional parameters.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping resolutions to transformed polygons.

        Raises:
            ValueError: If no valid resolutions are available.
        """
        # Filter resolutions based on vertex count
        vertex_count = polygons.shape[1]
        filtered_resolutions = self.strategy.filter_resolutions(resolutions, vertex_count)

        if not filtered_resolutions:
            raise ValueError(
                f"No valid resolutions for polygons with {vertex_count} vertices"
            )

        # Initialize result dictionary
        num_polygons = len(polygons)
        result = {res: np.zeros((num_polygons, res, 2)) for res in filtered_resolutions}

        # Process each polygon
        for i, polygon in enumerate(polygons):
            transformed = self.strategy.transform_polygon(polygon, filtered_resolutions, **kwargs)
            for resolution, simplified in transformed.items():
                result[resolution][i] = simplified

        return result