from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from ..strategy import TransformerStrategy
from ..registry import register_transformer

# Configure module logger
logger = logging.getLogger(__name__)


@register_transformer("canonicalize")
class CanonicalizeStrategy(TransformerStrategy):
    """
    Strategy for canonicalizing polygons.

    Canonicalization rotates each polygon to start with its lexicographically
    smallest vertex (minimum x, then minimum y), maintaining closure.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the canonicalization strategy.

        Args:
            config: Configuration parameters.
        """
        super().__init__(config)

    def get_output_type(self) -> str:
        """
        Get the type of output produced by this strategy.

        Returns:
            str: "single" for single array output.
        """
        return "single"

    def transform_polygon(
            self,
            polygon: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> np.ndarray:
        """
        Canonicalize a polygon by rotating to start with lexicographically smallest vertex.

        Args:
            polygon: Array of polygon vertices with shape [vertices, 2].
            resolutions: Not used for canonicalization.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Canonicalized polygon.
        """
        # Check if polygon is closed
        is_closed = np.array_equal(polygon[0], polygon[-1])

        # Get open polygon (exclude closing vertex)
        open_polygon = polygon[:-1] if is_closed else polygon

        # Find lexicographically smallest vertex
        lex_min_idx = self._find_lexicographically_min_vertex(open_polygon)

        # Rotate polygon to start with the smallest vertex
        rotated = self._rotate_polygon(open_polygon, lex_min_idx)

        # Add closing vertex if original was closed
        if is_closed:
            return np.vstack([rotated, rotated[0]])
        else:
            return rotated

    def _find_lexicographically_min_vertex(self, polygon: np.ndarray) -> int:
        """
        Find the index of the lexicographically smallest vertex.

        Lexicographic order: minimum x, then minimum y if x is equal.

        Args:
            polygon: Array of polygon vertices.

        Returns:
            Index of the lexicographically smallest vertex.
        """
        # Extract x and y coordinates
        x = polygon[:, 0]
        y = polygon[:, 1]

        # Find minimum x
        min_x = np.min(x)
        min_x_indices = np.where(x == min_x)[0]

        # Among vertices with minimum x, find minimum y
        min_y_at_min_x = y[min_x_indices].min()
        lex_min_idx = np.where((x == min_x) & (y == min_y_at_min_x))[0][0]

        return lex_min_idx

    def _rotate_polygon(self, polygon: np.ndarray, start_idx: int) -> np.ndarray:
        """
        Rotate polygon to start with the vertex at start_idx.

        Args:
            polygon: Array of polygon vertices.
            start_idx: Index to rotate to the beginning.

        Returns:
            Rotated polygon.
        """
        if start_idx == 0:
            return polygon.copy()

        num_vertices = len(polygon)
        rotated = np.empty_like(polygon)

        # Apply rotation
        for i in range(num_vertices):
            new_idx = (i + start_idx) % num_vertices
            rotated[i] = polygon[new_idx]

        return rotated