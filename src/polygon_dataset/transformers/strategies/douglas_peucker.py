from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import logging

from ..strategy import TransformerStrategy
from ..registry import register_transformer

# Configure module logger
logger = logging.getLogger(__name__)


@register_transformer("douglas_peucker")
class DouglasPeuckerStrategy(TransformerStrategy):
    """
    Strategy for Douglas-Peucker polygon simplification.

    This algorithm iteratively removes vertices that contribute least to the
    polygon's shape, preserving the most important features.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Douglas-Peucker strategy.

        Args:
            config: Configuration parameters.
        """
        super().__init__(config)

    def get_output_type(self) -> str:
        """
        Get the type of output produced by this strategy.

        Returns:
            str: "multi" for multiple resolutions.
        """
        return "multi"

    def transform_polygon(
            self,
            polygon: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Dict[int, np.ndarray]:
        """
        Simplify a polygon to multiple resolutions.

        Args:
            polygon: Array of polygon vertices with shape [vertices, 2].
            resolutions: List of target vertex counts.
            **kwargs: Additional parameters.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping resolutions to simplified polygons.

        Raises:
            ValueError: If resolutions are invalid.
        """
        if not resolutions:
            raise ValueError("Resolutions must be provided for Douglas-Peucker simplification")

        # Initialize result dictionary
        result = {}

        # Process each resolution
        for resolution in resolutions:
            simplified = self._simplify_to_resolution(polygon, resolution)
            result[resolution] = simplified

        return result

    def _simplify_to_resolution(
            self,
            polygon: np.ndarray,
            target_vertices: int
    ) -> np.ndarray:
        """
        Simplify a polygon to a specific resolution.

        Args:
            polygon: Array of polygon vertices.
            target_vertices: Target number of vertices.

        Returns:
            Simplified polygon.

        Raises:
            ValueError: If target_vertices < 3.
        """
        if target_vertices < 3:
            raise ValueError("Target vertex count must be >= 3")

        # Check if polygon is closed
        is_closed = np.array_equal(polygon[0], polygon[-1])

        # Handle closure
        if is_closed:
            open_polygon = polygon[:-1]
            target_open = target_vertices - 1
        else:
            open_polygon = polygon
            target_open = target_vertices

        # Apply fixed-size Douglas-Peucker
        simplified = self._douglas_peucker_fixed_size(open_polygon, target_open)

        # Restore closure if needed
        if is_closed:
            simplified = np.vstack([simplified, simplified[0]])

        return simplified

    def _douglas_peucker_fixed_size(
            self,
            points: np.ndarray,
            target_vertices: int
    ) -> np.ndarray:
        """
        Apply Douglas-Peucker algorithm to reduce polygon to target number of vertices.

        Args:
            points: Array of points [(x1, y1), (x2, y2), ...].
            target_vertices: Desired number of vertices in simplified polygon.

        Returns:
            np.ndarray: Simplified polygon points.
        """
        # If already at or below target, return as is
        if len(points) <= target_vertices:
            return points

        # Always keep first and last points
        selected_indices = {0, len(points) - 1}

        # Add points iteratively by maximum distance
        while len(selected_indices) < target_vertices:
            # Find next point to add
            next_point = self._find_next_point(points, selected_indices)

            # Break if no more points to add
            if next_point is None:
                break

            # Add the point with maximum distance
            selected_indices.add(next_point)

        # Return simplified polygon using selected indices
        return points[sorted(selected_indices)]

    def _find_next_point(
            self,
            points: np.ndarray,
            selected_indices: Set[int]
    ) -> Optional[int]:
        """
        Find the next point to add to the simplified polygon.

        Args:
            points: Array of points.
            selected_indices: Set of already selected indices.

        Returns:
            Index of the point with maximum distance, or None if no points remain.
        """
        # Get ordered list of selected indices
        selected_list = sorted(selected_indices)

        max_distance = -1
        max_index = None

        # Check each segment between selected points
        for i in range(len(selected_list) - 1):
            start_idx = selected_list[i]
            end_idx = selected_list[i + 1]

            # Skip if adjacent
            if end_idx - start_idx <= 1:
                continue

            # Check all points between these selected points
            for j in range(start_idx + 1, end_idx):
                if j in selected_indices:
                    continue

                distance = self._perpendicular_distance(
                    points[j],
                    points[start_idx],
                    points[end_idx]
                )

                if distance > max_distance:
                    max_distance = distance
                    max_index = j

        return max_index

    def _perpendicular_distance(
            self,
            point: np.ndarray,
            line_start: np.ndarray,
            line_end: np.ndarray
    ) -> float:
        """
        Calculate the perpendicular distance from a point to a line segment.

        Args:
            point: Point coordinates (x, y).
            line_start: Start point of line segment (x, y).
            line_end: End point of line segment (x, y).

        Returns:
            float: Perpendicular distance from point to line.
        """
        # Handle degenerate case
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)

        # Calculate perpendicular distance
        numerator = abs(np.cross(line_end - line_start, line_start - point))
        denominator = np.linalg.norm(line_end - line_start)

        return numerator / denominator