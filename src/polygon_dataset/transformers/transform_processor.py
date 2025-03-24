# polygon_dataset/transformers/transform_processor.py
"""
Parallel and sequential processing for polygon transformations.

This module provides the TransformProcessor class for handling the parallel
and sequential processing of polygon data during transformations.
"""

import gc
import logging
import multiprocessing
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from polygon_dataset.transformers.polygon_transformer import PolygonTransformer
from polygon_dataset.utils.chunking import distribute_work

# Configure module logger
logger = logging.getLogger(__name__)


class TransformProcessor:
    """
    Handles parallel and sequential processing for polygon transformations.

    This class manages the computation aspects of transforming polygon data,
    including parallel processing and memory management.
    """

    def __init__(
            self,
            transformer: PolygonTransformer,
            chunk_size: int = 100000,
            num_processes: Optional[int] = None
    ) -> None:
        """
        Initialize the transform processor.

        Args:
            transformer: The polygon transformer to use.
            chunk_size: Size of chunks for batch processing.
            num_processes: Number of processes for parallel processing.
                If None, uses CPU count - 1.
        """
        self.transformer = transformer
        self.chunk_size = chunk_size
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)

    def process(
            self,
            polygons: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Process polygons using either parallel or sequential processing.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            resolutions: List of target resolutions (for simplification algorithms).
            **kwargs: Additional parameters.

        Returns:
            Transformed polygons, either as a single array or a dictionary mapping
            resolutions to arrays.
        """
        num_polygons = len(polygons)

        # Use parallel processing for larger batches
        if num_polygons > 1000:
            logger.info(f"Using parallel processing for {num_polygons} polygons")
            return self._process_in_parallel(polygons, resolutions, **kwargs)
        else:
            logger.info(f"Using sequential processing for {num_polygons} polygons")
            return self.transformer.transform(polygons, resolutions, **kwargs)

    def _process_in_parallel(
            self,
            polygons: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Process polygons in parallel using multiprocessing.

        Args:
            polygons: Array of polygons.
            resolutions: List of target resolutions.
            **kwargs: Additional parameters.

        Returns:
            Transformed polygons.
        """
        # Split work among processes
        num_polygons = len(polygons)
        distribution = distribute_work(num_polygons, self.num_processes)

        output_type = self.transformer.strategy.get_output_type()

        if output_type == "single":
            return self._process_single_output_parallel(polygons, distribution, **kwargs)
        else:
            return self._process_multi_output_parallel(
                polygons, distribution, resolutions, **kwargs
            )

    def _process_single_output_parallel(
            self,
            polygons: np.ndarray,
            distribution: List[Tuple[int, int]],
            **kwargs: Any
    ) -> np.ndarray:
        """
        Process single-output transformations in parallel.

        Args:
            polygons: Array of polygons.
            distribution: List of (start, end) index tuples for each worker.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Array of transformed polygons.
        """
        # Prepare batch data for workers
        batch_data = [
            (polygons[start:end], start, kwargs)
            for start, end in distribution
            if end > start
        ]

        # Process in parallel
        with Pool(processes=self.num_processes) as pool:
            batch_results = list(pool.map(self._worker_transform_single, batch_data))

        # Sort and combine results
        batch_results.sort(key=lambda x: x[0])
        result = np.concatenate([data for _, data in batch_results], axis=0)

        return result

    def _process_multi_output_parallel(
            self,
            polygons: np.ndarray,
            distribution: List[Tuple[int, int]],
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Dict[int, np.ndarray]:
        """
        Process multi-output transformations in parallel.

        Args:
            polygons: Array of polygons.
            distribution: List of (start, end) index tuples for each worker.
            resolutions: List of target resolutions.
            **kwargs: Additional parameters.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping resolutions to transformed polygons.

        Raises:
            ValueError: If no valid resolutions are available.
        """
        # Filter resolutions
        vertex_count = polygons.shape[1]
        if resolutions is None:
            resolutions = calculate_resolution_steps(vertex_count, self.transformer.min_vertices)

        filtered_resolutions = self.transformer.strategy.filter_resolutions(
            resolutions, vertex_count
        )

        if not filtered_resolutions:
            raise ValueError(
                f"No valid resolutions for polygons with {vertex_count} vertices"
            )

        # Prepare batch data for workers
        batch_data = [
            (polygons[start:end], start, filtered_resolutions, kwargs)
            for start, end in distribution
            if end > start
        ]

        # Process in parallel
        with Pool(processes=self.num_processes) as pool:
            batch_results = list(pool.map(self._worker_transform_multi, batch_data))

        # Initialize result dictionary
        num_polygons = len(polygons)
        result = {res: np.zeros((num_polygons, res, 2)) for res in filtered_resolutions}

        # Combine results by copying each batch to the appropriate location
        for start_idx, res_dict in batch_results:
            batch_size = next(iter(res_dict.values())).shape[0]
            end_idx = start_idx + batch_size

            for resolution, data in res_dict.items():
                result[resolution][start_idx:end_idx] = data

        return result

    def _worker_transform_single(
            self,
            batch_data: Tuple[np.ndarray, int, Dict]
    ) -> Tuple[int, np.ndarray]:
        """
        Worker function for parallel processing of single-output strategies.

        Args:
            batch_data: Tuple containing:
                - Array of polygons to process
                - Starting index within the full batch
                - Additional keyword arguments

        Returns:
            Tuple containing:
                - Starting index
                - Array of transformed polygons
        """
        polygons, start_idx, kwargs = batch_data
        result = self.transformer.transform(polygons, **kwargs)
        return start_idx, result

    def _worker_transform_multi(
            self,
            batch_data: Tuple[np.ndarray, int, List[int], Dict]
    ) -> Tuple[int, Dict[int, np.ndarray]]:
        """
        Worker function for parallel processing of multi-output strategies.

        Args:
            batch_data: Tuple containing:
                - Array of polygons to process
                - Starting index within the full batch
                - List of target resolutions
                - Additional keyword arguments

        Returns:
            Tuple containing:
                - Starting index
                - Dictionary mapping resolutions to transformed polygons
        """
        polygons, start_idx, resolutions, kwargs = batch_data
        result = self.transformer.transform(polygons, resolutions, **kwargs)
        return start_idx, result