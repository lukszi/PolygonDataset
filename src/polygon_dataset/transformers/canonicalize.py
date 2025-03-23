# polygon_dataset/transformers/canonicalize.py
"""
Polygon canonicalization transformer with multiprocessing support.

This module provides functionality for canonicalizing polygons by rotating
them to start with their lexicographically smallest vertex, using multiprocessing
for improved performance on large datasets.
"""

import gc
import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)

from polygon_dataset.core import PathManager
from polygon_dataset.transformers.base import Transformer
from polygon_dataset.transformers.registry import register_transformer


def _process_polygon_batch(
        batch_data: Tuple[np.ndarray, int]
) -> Tuple[int, np.ndarray]:
    """
    Process a batch of polygons by canonicalizing them.

    Args:
        batch_data: Tuple containing:
            - Array of polygons to process
            - Starting index within the current chunk

    Returns:
        Tuple containing:
            - Starting index for this batch
            - Array of canonicalized polygons
    """
    polygons, start_idx = batch_data

    # Call the canonicalization function
    batch_size, num_vertices, dim = polygons.shape

    # Remove closing vertices for processing
    open_polygons = polygons[:, :-1, :]
    open_vertices = num_vertices - 1

    # Extract x and y coordinates from open polygons
    x = open_polygons[..., 0]
    y = open_polygons[..., 1]

    # Find minimum x coordinates for each polygon
    min_x = np.min(x, axis=1, keepdims=True)
    mask_min_x = (x == min_x)

    # Find minimum y among vertices with minimum x
    y_masked = np.where(mask_min_x, y, np.finfo(np.float32).max)
    min_y = np.min(y_masked, axis=1, keepdims=True)
    mask_min = np.logical_and(mask_min_x, (y == min_y))

    # Find the index of the lexicographically smallest vertex for each polygon
    indices = np.argmax(mask_min.astype(np.float32), axis=1)

    # Create a matrix of indices for each vertex in each polygon after rotation
    vertex_indices = np.arange(open_vertices)
    # Broadcasting to create a matrix of shape (batch_size, open_vertices)
    offset_matrix = (vertex_indices + indices[:, np.newaxis]) % open_vertices

    # Create batch indices
    batch_indices = np.arange(batch_size)[:, np.newaxis]
    batch_indices = np.tile(batch_indices, (1, open_vertices))

    # Gather vertices according to rotation offsets
    rotated_polygons = open_polygons[batch_indices, offset_matrix]

    # Create the result array including the closing vertex
    result = np.zeros_like(polygons)
    result[:, :-1, :] = rotated_polygons
    result[:, -1, :] = rotated_polygons[:, 0, :]  # Add closing vertex

    return start_idx, result


@register_transformer("canonicalize")
class CanonicalizeTransformer(Transformer):
    """
    Transformer for canonicalizing polygons with multiprocessing support.

    Canonicalization rotates each polygon to start with its lexicographically
    smallest vertex (minimum x, then minimum y), maintaining closure.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the canonicalization transformer.

        Args:
            config: Configuration parameters for the transformer.
                May include 'chunk_size' for batch processing.
        """
        super().__init__(config)

    def transform(self, polygons: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Canonicalize a batch of polygons.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            **kwargs: Additional transformation parameters (unused).

        Returns:
            np.ndarray: Canonicalized polygons.

        Raises:
            ValueError: If the input polygons have an invalid shape.
        """
        if len(polygons.shape) != 3 or polygons.shape[2] != 2:
            raise ValueError(f"Invalid polygon data shape: {polygons.shape}. Expected (N, V, 2)")

        # For small batches, process directly without multiprocessing
        if len(polygons) <= 1000:
            return self._canonicalize_polygons(polygons)

        # For larger batches, use multiprocessing
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        num_polygons = len(polygons)

        # Calculate batch size for each worker
        batch_size = num_polygons // num_processes
        remainder = num_polygons % num_processes

        # Create batch data for parallel processing
        batch_data = []
        for i in range(num_processes):
            batch_start = i * batch_size + min(i, remainder)
            batch_end = min((i + 1) * batch_size + min(i + 1, remainder), num_polygons)

            if batch_end > batch_start:  # Only create batches with data
                batch_data.append((
                    polygons[batch_start:batch_end],
                    batch_start
                ))

        # Process batches in parallel
        with Pool(processes=num_processes) as pool:
            batch_results = list(pool.map(_process_polygon_batch, batch_data))

        # Sort results by batch start index
        batch_results.sort(key=lambda x: x[0])

        # Concatenate results
        result = np.concatenate([data for _, data in batch_results], axis=0)

        return result

    def _canonicalize_polygons(self, polygons: np.ndarray) -> np.ndarray:
        """
        Canonicalize a batch of closed polygons.

        Rotates each polygon to start with its lexicographically smallest vertex
        (minimum x, then minimum y). Assumes all input polygons are closed
        (first and last vertices are identical).

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].

        Returns:
            np.ndarray: Canonicalized polygons with the same shape as input.
        """
        canonicalized_batch = _process_polygon_batch((polygons, 1))
        _, result = canonicalized_batch
        return result

    def transform_dataset(
            self,
            path_manager: PathManager,
            **kwargs: Any
    ) -> None:
        """
        Canonicalize an entire dataset of polygons.

        Args:
            path_manager: Path manager for the dataset.
            **kwargs: Additional parameters:
                - resolutions: List of resolutions to process (optional).
                - generator: Generator name filter (optional).
                - algorithm: Algorithm name filter (optional).
                - split: Split name filter (optional).

        Raises:
            ValueError: If canonicalization fails.
        """
        # Extract optional parameters
        resolutions = kwargs.get("resolutions", [None])  # Default to processing original resolution
        generator_filter = kwargs.get("generator")
        algorithm_filter = kwargs.get("algorithm")
        split_filter = kwargs.get("split")

        logger.info("Starting dataset canonicalization...")

        # Process extracted dir (original resolution files)
        extracted_dir = path_manager.get_extracted_dir()
        if extracted_dir.exists() and None in resolutions:
            self._process_directory(
                path_manager,
                extracted_dir,
                None,  # No resolution for extracted files
                generator_filter,
                algorithm_filter,
                split_filter
            )

        # Process transformed dir (resolution-specific files)
        transformed_dir = path_manager.get_transformed_dir()
        if transformed_dir.exists():
            # Filter resolutions that are not None (i.e., specific resolution values)
            filtered_resolutions = [res for res in resolutions if res is not None]
            if filtered_resolutions:
                for resolution in filtered_resolutions:
                    self._process_directory(
                        path_manager,
                        transformed_dir,
                        resolution,
                        generator_filter,
                        algorithm_filter,
                        split_filter
                    )

        logger.info("Dataset canonicalization complete!")

    def _process_directory(
            self,
            path_manager: PathManager,
            directory: Path,
            resolution: Optional[int],
            generator_filter: Optional[str],
            algorithm_filter: Optional[str],
            split_filter: Optional[str]
    ) -> None:
        """
        Process all files in a directory for canonicalization.

        Args:
            path_manager: Path manager for the dataset.
            directory: Directory containing files to process.
            resolution: Resolution value (None for original resolution).
            generator_filter: Generator name filter (optional).
            algorithm_filter: Algorithm name filter (optional).
            split_filter: Split name filter (optional).
        """
        # Determine pattern based on resolution
        if resolution is not None:
            pattern = f"*_res{resolution}.npy"
        else:
            pattern = "*.npy"

        # Use the enhanced PathManager to find and filter files
        npy_files = path_manager.find_npy_files(
            directory=directory,
            pattern=pattern,
            generator_filter=generator_filter,
            algorithm_filter=algorithm_filter,
            split_filter=split_filter,
            resolution_filter=resolution
        )

        if not npy_files:
            logger.info(f"No matching files found in {directory} for resolution {resolution}")
            return

        logger.info(f"Found {len(npy_files)} files to process in {directory} for resolution {resolution}")

        # Process each file
        for file_path in npy_files:
            self._process_canonicalize_file(path_manager, file_path, resolution)

    def _process_canonicalize_file(
            self,
            path_manager: PathManager,
            file_path: Path,
            resolution: Optional[int]
    ) -> None:
        """
        Canonicalize a single file.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the file to canonicalize.
            resolution: Resolution value (None for original resolution).
        """
        # Parse file name to extract components
        from polygon_dataset.utils.filename_parser import parse_polygon_filename

        try:
            components = parse_polygon_filename(file_path.name)
            split = components['split']
            generator = components['generator']
            algorithm = components['algorithm']
        except ValueError:
            raise ValueError(f"Invalid file name format: {file_path.name}")

        # Get output path
        output_path = path_manager.get_canonical_path(
            generator, algorithm, split, resolution
        )

        # Process the canonicalization
        self._canonicalize_file(path_manager, file_path, output_path)

    def _canonicalize_file(
            self,
            path_manager: PathManager,
            input_file: Path,
            output_file: Path
    ) -> None:
        """
        Canonicalize a single .npy file containing polygons using multiprocessing.

        Args:
            path_manager: Path manager for the dataset.
            input_file: Path to the input .npy file.
            output_file: Path to the output .npy file.

        Raises:
            ValueError: If canonicalization fails.
        """
        logger.info(f"Canonicalizing {input_file.name}...")

        # Get input file shape without loading the entire data
        data = np.load(input_file, mmap_mode='r')
        poly_shape = data.shape
        total_polygons = len(data)
        del data  # Close the file immediately to free memory

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create initial empty file with correct shape
        memmap = open_memmap(
            output_file,
            dtype='float64',
            mode='w+',
            shape=poly_shape
        )
        memmap.flush()
        del memmap  # Close the file immediately to free memory

        # Determine number of processes to use (leave one core free)
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Process data in chunks to manage memory
        for chunk_start in range(0, total_polygons, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_polygons)
            chunk_size_actual = chunk_end - chunk_start

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

            # Load only the current chunk from input file
            data = np.load(input_file, mmap_mode='r')
            chunk_data = data[chunk_start:chunk_end].copy()  # Copy to separate from mmap
            del data  # Close the file immediately to free memory

            # Force garbage collection after loading chunk
            gc.collect()

            # Calculate number of batches for this chunk
            batch_size = max(1, chunk_size_actual // num_processes)
            remainder = chunk_size_actual % num_processes

            # Prepare batch data for this chunk
            batch_data = []
            for i in range(num_processes):
                batch_start = i * batch_size + min(i, remainder)
                batch_end = min((i + 1) * batch_size + min(i + 1, remainder), chunk_size_actual)
                if batch_end > batch_start:  # Only create batches with data
                    batch_data.append((
                        chunk_data[batch_start:batch_end],
                        batch_start  # Index within the chunk, not global index
                    ))

            # Process batches in parallel
            with Pool(processes=min(num_processes, len(batch_data))) as pool:
                batch_results = list(tqdm(
                    pool.imap(_process_polygon_batch, batch_data),
                    total=len(batch_data),
                    desc=f"Processing chunk {chunk_start}-{chunk_end}"
                ))

            # Sort results by batch start index
            batch_results.sort(key=lambda x: x[0])

            # Concatenate results
            processed_data = np.concatenate([data for _, data in batch_results], axis=0)

            # Write to output file
            memmap = open_memmap(
                output_file,
                dtype='float64',
                mode='r+',
                shape=poly_shape
            )
            memmap[chunk_start:chunk_end] = processed_data
            memmap.flush()
            del memmap  # Close immediately to free memory

            # Clean up chunk data and results
            del batch_results
            del processed_data
            del chunk_data
            gc.collect()

            logger.info(f"Finished processing and writing chunk {chunk_start}-{chunk_end}")

    def _get_output_dir(self, path_manager: PathManager) -> Path:
        """
        Get the output directory for canonicalization.

        Args:
            path_manager: Path manager for the dataset.

        Returns:
            Path: Directory for canonicalized files.
        """
        return path_manager.get_canonical_dir()