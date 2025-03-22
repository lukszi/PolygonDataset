# polygon_dataset/transformers/visvalingam.py
"""
Visvalingam-Whyatt polygon simplification transformer.

This module provides functionality for simplifying polygons using the
Visvalingam-Whyatt algorithm, which eliminates points based on the
area of the triangle they form with adjacent points.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import multiprocessing
from multiprocessing import Pool
import gc
from tqdm import tqdm

from polygon_dataset.core import PathManager
from polygon_dataset.transformers.base import Transformer
from polygon_dataset.transformers.registry import register_transformer
from polygon_dataset.utils import calculate_resolution_steps

# Configure module logger
logger = logging.getLogger(__name__)

def _process_polygon_batch(batch_data: Tuple[np.ndarray, List[int], List[int], int]) -> Tuple[
    int, List[Tuple[int, np.ndarray]]]:
    """
    Process a batch of polygons using the Visvalingam-Whyatt algorithm.

    Args:
        batch_data: Tuple containing:
            - Array of polygons to process
            - List of target resolutions
            - List of adjusted resolutions
            - Starting index within the chunk

    Returns:
        Tuple containing:
            - Starting index for this batch
            - List of (resolution, processed_data) tuples
    """
    from visvalingam_c import simplify_multi

    polygons, resolutions, adjusted_resolutions, start_idx = batch_data

    # Process each polygon in the batch
    simplified_polygons = []
    for polygon in polygons:
        # Get simplified versions
        results = simplify_multi(polygon.astype(np.float64),
                                 np.array(adjusted_resolutions, dtype=np.int32))
        simplified_polygons.append(results)

    # Reorganize results by resolution
    result_data = []
    for i, resolution in enumerate(resolutions):
        resolution_polygons = np.array([poly[i] for poly in simplified_polygons])
        result_data.append((resolution, resolution_polygons))

    return start_idx, result_data


@register_transformer("visvalingam")
class VisvalingamTransformer(Transformer):
    """
    Transformer for simplifying polygons using the Visvalingam-Whyatt algorithm.

    This algorithm iteratively removes points based on the area of the triangle
    they form with adjacent points, removing points that contribute least
    to the overall shape.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Visvalingam-Whyatt transformer.

        Args:
            config: Configuration parameters for the transformer.
                May include:
                - 'min_vertices': Minimum vertices for lowest resolution.
                - 'batch_size': Number of polygons to process in each batch.
        """
        super().__init__(config)

        # Check if the C implementation is available
        try:
            from visvalingam_c import simplify_multi
            self.simplify_multi = simplify_multi
        except ImportError:
            raise ImportError(
                "C implementation of Visvalingam-Whyatt not available. "
                "Please install the visvalingam_c package."
            )

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
        vertex_count = polygons.shape[1]
        if len(polygons.shape) != 3 or polygons.shape[2] != 2:
            raise ValueError(f"Invalid polygon data shape: {polygons.shape}. Expected (N, V, 2)")

        # Extract resolution steps from kwargs or calculate them
        resolution_steps = kwargs.get("resolutions")
        if resolution_steps is None:
            resolution_steps = calculate_resolution_steps(vertex_count, self.min_vertices)

        # remove resolutions that are larger than the number of vertices
        resolution_steps = [res for res in resolution_steps if res < vertex_count]

        # Process the polygons
        num_polygons = len(polygons)
        result = {}

        # Initialize result dictionaries for each resolution
        for resolution in resolution_steps:
            result[resolution] = np.zeros((num_polygons, resolution, 2))

        # Determine the number of processes to use (leave some CPU cores free)
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Account for closing vertex in C implementation which doesn't count the closing vertex
        adjusted_resolutions = [resolution - 1 for resolution in resolution_steps]

        # Calculate batch size for each worker
        batch_size = num_polygons // num_processes
        remainder = num_polygons % num_processes

        # Create tasks for worker processes
        batch_data = []
        for i in range(num_processes):
            batch_start = (i * batch_size) + min(i, remainder)
            batch_end = min((i + 1) * batch_size + min(i + 1, remainder), num_polygons)

            if batch_start < batch_end:
                batch_data.append((
                    polygons[batch_start:batch_end],
                    resolution_steps,
                    adjusted_resolutions,
                    batch_start
                ))

        # Process batches in parallel
        with Pool(processes=num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(_process_polygon_batch, batch_data),
                total=len(batch_data),
                desc="Simplifying polygons"
            ))

        # Combine results from all batches
        for batch_start, resolution_data in batch_results:
            for resolution, data in resolution_data:
                result[resolution][batch_start:batch_start + len(data)] = data

        return result

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
        Process a file for each resolution using Visvalingam-Whyatt simplification.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            resolutions: List of resolutions to process.
            split: Dataset split.
            generator: Generator name.
            algorithm: Algorithm name.
            **kwargs: Additional parameters.
        """
        # Get file shape and total count without loading all data
        data = np.load(file_path, mmap_mode='r')
        total_polygons = len(data)
        poly_shape = data.shape
        del data
        if len(poly_shape) != 3 or poly_shape[2] != 2:
            raise ValueError(f"Invalid polygon data shape: {poly_shape}. Expected (N, V, 2)")

        # Remove resolutions that are larger than the number of vertices
        resolutions = [res for res in resolutions if res < poly_shape[1]]

        # Adjusted resolutions for C implementation (accounting for closing vertex)
        adjusted_resolutions = [resolution - 1 for resolution in resolutions]

        # Create output files
        output_paths = {}
        for resolution in resolutions:
            output_file = path_manager.get_resolution_path(
                generator=generator,
                algorithm=algorithm,
                split=split,
                resolution=resolution
            )

            # Create the output file with proper shape
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Create memory-mapped file
            memmap = np.lib.format.open_memmap(
                output_file,
                dtype='float64',
                mode='w+',
                shape=(total_polygons, resolution, 2)
            )
            memmap.flush()
            del memmap  # Close to free memory

            output_paths[resolution] = output_file

        # Determine number of processes to use
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Process data in chunks to manage memory
        for chunk_start in range(0, total_polygons, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_polygons)
            chunk_size_actual = chunk_end - chunk_start

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

            # Load chunk data
            data = np.load(file_path, mmap_mode='r')
            chunk_data = data[chunk_start:chunk_end].copy()
            del data  # Close the file to free memory

            # Calculate batch size for each worker
            batch_size = chunk_size_actual // num_processes
            remainder = chunk_size_actual % num_processes

            # Create tasks for worker processes
            batch_data = []
            for i in range(num_processes):
                batch_start = (i * batch_size) + min(i, remainder)
                batch_end = min((i + 1) * batch_size + min(i + 1, remainder), chunk_size_actual)

                if batch_start < batch_end:
                    batch_data.append((
                        chunk_data[batch_start:batch_end],
                        resolutions,
                        adjusted_resolutions,
                        batch_start  # Index within the chunk, not global index
                    ))

            # Process batches in parallel
            with Pool(processes=num_processes) as pool:
                batch_results = list(tqdm(
                    pool.imap(_process_polygon_batch, batch_data),
                    total=len(batch_data),
                    desc=f"Processing chunk {chunk_start}-{chunk_end}"
                ))

            # Organize results by resolution
            chunk_results = {res: [] for res in resolutions}
            for batch_start, resolution_data in batch_results:
                for resolution, data in resolution_data:
                    chunk_results[resolution].append((batch_start, data))

            # Write results to output files
            for resolution in resolutions:
                # Sort by batch start to ensure correct order
                sorted_results = sorted(chunk_results[resolution], key=lambda x: x[0])

                # Concatenate data
                resolution_data = np.concatenate([data for _, data in sorted_results], axis=0)

                # Write to output file
                memmap = np.lib.format.open_memmap(
                    output_paths[resolution],
                    dtype='float64',
                    mode='r+',
                    shape=(total_polygons, resolution, 2)
                )
                memmap[chunk_start:chunk_end] = resolution_data
                memmap.flush()
                del memmap  # Close to free memory

            # Clean up to free memory
            del chunk_data, batch_results, chunk_results
            gc.collect()