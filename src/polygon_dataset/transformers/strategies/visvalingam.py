from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from ..strategy import TransformerStrategy
from ..registry import register_transformer

# Configure module logger
logger = logging.getLogger(__name__)


@register_transformer("visvalingam")
class VisvalingamStrategy(TransformerStrategy):
    """
    Strategy for Visvalingam-Whyatt polygon simplification.

    This strategy simplifies polygons by removing points based on the
    area of the triangle they form with adjacent points. The C implementation
    processes all resolutions at once for efficiency.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Visvalingam-Whyatt strategy.

        Args:
            config: Configuration parameters.

        Raises:
            ImportError: If the C implementation is not available.
        """
        super().__init__(config)

        # Check for C implementation
        try:
            from visvalingam_c import simplify_multi
            self.simplify_multi = simplify_multi
        except ImportError:
            raise ImportError(
                "C implementation of Visvalingam-Whyatt not available. "
                "Please install the visvalingam_c package."
            )

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
        Simplify a polygon to multiple resolutions at once.

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
            raise ValueError("Resolutions must be provided for Visvalingam-Whyatt simplification")

        # Filter resolutions: they must be strictly less than input vertex count
        # for the C implementation
        valid_resolutions = []
        valid_adjusted_resolutions = []

        for resolution in resolutions:
            if resolution < len(polygon):
                valid_resolutions.append(resolution)
                valid_adjusted_resolutions.append(resolution-1)

        # Create result dictionary
        result = {}

        # For any resolution that equals the full vertex count, just return the original
        for resolution in resolutions:
            if resolution >= len(polygon):
                result[resolution] = polygon.copy()

        # If no valid resolutions for C implementation, return what we have
        if not valid_resolutions:
            return result

        # Use the C implementation to simplify all valid resolutions at once
        simplified_versions = self.simplify_multi(
            polygon,
            np.array(valid_adjusted_resolutions, dtype=np.int32)
        )

        # Add simplified results to the dictionary
        for i, resolution in enumerate(valid_resolutions):
            result[resolution] = simplified_versions[i]

        return result

    def filter_resolutions(self, resolutions: List[int], vertex_count: int) -> List[int]:
        """
        Filter resolutions that exceed the vertex count.

        Overrides the base method to enforce the constraint that for Visvalingam,
        target resolutions must be strictly less than the input vertex count.

        Args:
            resolutions: List of target resolutions.
            vertex_count: Maximum vertex count.

        Returns:
            Filtered list of valid resolutions.
        """
        return [r for r in resolutions if r < vertex_count]