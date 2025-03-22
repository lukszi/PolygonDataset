# polygon_dataset/transformers/douglas_peucker.py
"""
Douglas-Peucker polygon simplification transformer.

This module provides functionality for simplifying polygons using the
Douglas-Peucker algorithm, which reduces the number of vertices while
preserving important features.
"""

import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from polygon_dataset.core import PathManager
from polygon_dataset.transformers.base import Transformer
from polygon_dataset.transformers.registry import register_transformer
from polygon_dataset.utils import calculate_resolution_steps


@register_transformer("douglas_peucker")
class DouglasPeuckerTransformer(Transformer):
    """
    Transformer for simplifying polygons using the Douglas-Peucker algorithm.

    This algorithm iteratively removes vertices that contribute least to the
    polygon's shape, preserving the most important features.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Douglas-Peucker transformer.

        Args:
            config: Configuration parameters for the transformer.
                May include:
                - 'min_vertices': Minimum vertices for lowest resolution.
                - 'batch_size': Number of polygons to process in each batch.
        """
        super().__init__(config)

    def transform(
            self,
            polygons: np.ndarray,
            **kwargs: Any
    ) -> Dict[int, np.ndarray]:
        """
        Simplify a batch of polygons to multiple resolutions.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            **kwargs: Additional parameters:
                - resolutions: List of target vertex counts (optional).
                  If not provided, calculated based on min_vertices.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping resolutions to arrays of
            simplified polygons, each with shape [num_polygons, resolution, 2].

        Raises:
            ValueError: If the input polygons have an invalid shape.
        """
        if len(polygons.shape) != 3 or polygons.shape[2] != 2:
            raise ValueError(f"Invalid polygon data shape: {polygons.shape}. Expected (N, V, 2)")

        # Extract resolution steps from kwargs or calculate them
        resolution_steps = kwargs.get("resolutions")
        if resolution_steps is None:
            vertex_count = polygons.shape[1]
            resolution_steps = calculate_resolution_steps(vertex_count, self.min_vertices)

        # Process each resolution
        result = {}
        for resolution in resolution_steps:
            print(f"Simplifying to {resolution} vertices...")
            simplified = self._simplify_batch(polygons, resolution)
            result[resolution] = simplified

        return result

    def _simplify_batch(
            self,
            polygons: np.ndarray,
            target_vertices: int
    ) -> np.ndarray:
        """
        Simplify a batch of polygons to a specific resolution.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            target_vertices: Target number of vertices for each polygon.

        Returns:
            np.ndarray: Array of simplified polygons with shape [num_polygons, target_vertices, 2].
        """
        num_polygons = len(polygons)
        simplified_polygons = np.zeros((num_polygons, target_vertices, 2))

        # Use multiprocessing for better performance
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Process in smaller chunks to avoid memory issues
        chunk_size = min(1000, num_polygons)

        for chunk_start in range(0, num_polygons, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_polygons)
            chunk_size_actual = chunk_end - chunk_start

            # Process each polygon in this chunk
            for i in range(chunk_size_actual):
                polygon_idx = chunk_start + i
                polygon = polygons[polygon_idx]
                simplified_polygons[polygon_idx] = self._simplify_polygon(polygon, target_vertices)

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{chunk_size_actual} polygons in current chunk")

            print(f"Completed chunk {chunk_start}-{chunk_end}")

        return simplified_polygons

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
        # Convert inputs to numpy arrays
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)

        # Handle degenerate case of line start == line end
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)

        # Calculate perpendicular distance
        numerator = abs(np.cross(line_end - line_start, line_start - point))
        denominator = np.linalg.norm(line_end - line_start)

        return numerator / denominator

    def _find_point_distances(
            self,
            points: np.ndarray,
            selected_indices: set
    ) -> List[Tuple[float, int]]:
        """
        Calculate distances from points to their current line segments.

        Args:
            points: Array of points [(x1, y1), (x2, y2), ...].
            selected_indices: Indices of points that will be kept.

        Returns:
            List[Tuple[float, int]]: List of (distance, index) tuples for non-selected points.
        """
        distances = []
        selected_points = sorted(selected_indices)

        # For each segment between selected points
        for i in range(len(selected_points) - 1):
            start_idx = selected_points[i]
            end_idx = selected_points[i + 1]

            # Check all points between these selected points
            for j in range(start_idx + 1, end_idx):
                if j not in selected_indices:
                    dist = self._perpendicular_distance(
                        points[j],
                        points[start_idx],
                        points[end_idx]
                    )
                    distances.append((dist, j))

        return distances

    def _douglas_peucker_fixed_size(
            self,
            points: np.ndarray,
            target_vertices: int
    ) -> np.ndarray:
        """
        Apply Douglas-Peucker algorithm to reduce polygon to target number of vertices.

        This implementation recalculates distances after each point selection.

        Args:
            points: Array of points [(x1, y1), (x2, y2), ...].
            target_vertices: Desired number of vertices in simplified polygon.

        Returns:
            np.ndarray: Simplified polygon points.
        """
        if len(points) <= target_vertices:
            return points

        # Always keep first and last points
        selected_indices = {0, len(points) - 1}

        # Iteratively add points until we reach target size
        while len(selected_indices) < target_vertices:
            # Calculate distances for current configuration
            distances = self._find_point_distances(points, selected_indices)

            if not distances:
                break

            # Add point with maximum distance
            max_dist, max_idx = max(distances)
            selected_indices.add(max_idx)

        # Return simplified polygon using selected indices
        return points[sorted(selected_indices)]

    def _simplify_polygon(
            self,
            polygon: np.ndarray,
            target_vertices: int
    ) -> np.ndarray:
        """
        Simplify a polygon to have exactly the target number of vertices.

        Args:
            polygon: Array of polygon vertices [(x1, y1), (x2, y2), ...].
            target_vertices: Desired number of vertices (must be >= 3).

        Returns:
            np.ndarray: Simplified polygon vertices with closure preserved.

        Raises:
            ValueError: If target_vertices < 3.
        """
        if target_vertices < 3:
            raise ValueError("Target vertex count must be >= 3")

        vertice_removed = False
        # Handle polygon closure (if first and last vertices are identical)
        if np.array_equal(polygon[0], polygon[-1]):
            vertice_removed = True
            polygon = polygon[:-1]
            target_vertices -= 1  # Adjust target for closure point

        # Apply fixed-size Douglas-Peucker
        simplified = self._douglas_peucker_fixed_size(polygon, target_vertices)

        # Restore closure if needed
        if vertice_removed:
            simplified = np.vstack((simplified, simplified[0]))

        return simplified

    def _process_file_by_resolution(
            self,
            path_manager: PathManager,
            file_path: Path,
            resolutions: List[int],
            split: str,
            generator: str,
            algorithm: str,
            **kwargs: Any
    ) -> None:
        """
        Process a file for each resolution using Douglas-Peucker simplification.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            resolutions: List of resolutions to process.
            split: Dataset split.
            generator: Generator name.
            algorithm: Algorithm name.
            **kwargs: Additional parameters.
        """
        # Get file shape
        with np.load(file_path) as data:
            total_polygons = len(data)
            vertex_count = data.shape[1]

        # Process each resolution
        for resolution in resolutions:
            # Skip if resolution is higher than original vertex count
            if resolution > vertex_count:
                print(f"Skipping resolution {resolution} (higher than original vertex count)")
                continue

            output_file = path_manager.get_resolution_path(
                generator=generator,
                algorithm=algorithm,
                split=split,
                resolution=resolution
            )

            # Define a function to process chunks with Douglas-Peucker algorithm
            def process_dp_chunk(chunk_data, start_idx, **chunk_kwargs):
                """Process a chunk using Douglas-Peucker algorithm"""
                simplified_polygons = np.zeros((len(chunk_data), resolution, 2))

                for i, polygon in enumerate(tqdm(chunk_data, desc=f"Resolution {resolution}")):
                    try:
                        simplified_polygons[i] = self._simplify_polygon(polygon, resolution)
                    except Exception as e:
                        print(f"Error processing polygon {start_idx + i}: {e}")
                        # Use interpolation as fallback
                        if len(polygon) >= resolution:
                            indices = np.linspace(0, len(polygon) - 1, resolution, dtype=int)
                            simplified_polygons[i] = polygon[indices]
                        else:
                            # If polygon has fewer vertices than target, pad it
                            padded = np.pad(
                                polygon,
                                ((0, resolution - len(polygon)), (0, 0)),
                                mode='edge'
                            )
                            simplified_polygons[i] = padded

                return simplified_polygons

            # Process in chunks using the base class method
            self._process_in_chunks(
                input_data=file_path,
                output_file=output_file,
                output_shape=(total_polygons, resolution, 2),
                process_chunk_func=process_dp_chunk
            )