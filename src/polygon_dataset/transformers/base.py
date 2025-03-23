# polygon_dataset/transformers/base.py
"""
Base class for polygon transformers.

This module provides the abstract base class for polygon transformers
that process and modify polygons in various ways.
"""

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from polygon_dataset.core import PathManager
from polygon_dataset.utils import calculate_resolution_steps


class Transformer(ABC):
    """
    Abstract base class for polygon transformers.

    Transformers process polygon data to produce modified versions,
    such as simplified, canonicalized, or otherwise transformed polygons.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the transformer with configuration parameters.

        Args:
            config: Configuration parameters for the transformer.
        """
        self.config: Dict[str, Any] = config
        self.name: str = config.get("name", "unknown")
        self.chunk_size: int = config.get("batch_size", config.get("chunk_size", 100000))
        self.min_vertices: int = config.get("min_vertices", 10)

    @abstractmethod
    def transform(
            self,
            polygons: np.ndarray,
            **kwargs: Any
    ) -> Any:
        """
        Transform a batch of polygons.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            **kwargs: Additional transformation parameters.

        Returns:
            Any: Transformed polygons, format depends on the specific transformer.

        Raises:
            ValueError: If transformation fails or if the configuration is invalid.
        """
        pass

    def transform_dataset(
            self,
            path_manager: PathManager,
            **kwargs: Any
    ) -> None:
        """
        Transform an entire dataset of polygons.

        This default implementation handles the common logic for finding files,
        filtering based on criteria, and managing the transformation process.

        Args:
            path_manager: Path manager for the dataset.
            **kwargs: Additional parameters:
                - resolutions: List of target vertex counts (optional).
                - generator: Generator name filter (optional).
                - algorithm: Algorithm name filter (optional).
                - split: Split name filter (optional).

        Raises:
            ValueError: If transformation fails or if the configuration is invalid.
        """
        # Extract parameters
        generator_filter = kwargs.get("generator")
        algorithm_filter = kwargs.get("algorithm")
        split_filter = kwargs.get("split")
        resolutions = kwargs.get("resolutions")
        source_dir = kwargs.get("source_dir", path_manager.get_extracted_dir())

        # Find input files to process using the enhanced PathManager
        try:
            npy_files = path_manager.find_npy_files(
                directory=source_dir,
                pattern="*.npy",
                generator_filter=generator_filter,
                algorithm_filter=algorithm_filter,
                split_filter=split_filter
            )
        except ValueError as e:
            raise ValueError(f"Error finding input files: {e}")

        if not npy_files:
            raise ValueError(f"No matching files found in {source_dir}")

        print(f"Found {len(npy_files)} files to process in {source_dir}")

        # Process each file
        for file_path in npy_files:
            self._transform_file(path_manager, file_path, resolutions)

    def _transform_file(
            self,
            path_manager: PathManager,
            file_path: Path,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> None:
        """
        Transform a single .npy file containing polygons.

        This default implementation handles the common operations for
        transforming a file, including determining resolutions, creating
        output directories, and managing memory.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the .npy file containing polygons.
            resolutions: List of target vertex counts (optional).
            **kwargs: Additional parameters.

        Raises:
            ValueError: If transformation fails.
        """
        print(f"Transforming {file_path.name}...")

        # Parse file name to extract components using the utility
        try:
            from polygon_dataset.utils.filename_parser import parse_polygon_filename
            components = parse_polygon_filename(file_path.name)
            split = components['split']
            generator = components['generator']
            algorithm = components['algorithm']

            # If the file already has a resolution, adjust behavior
            if components['resolution'] is not None:
                file_resolution = int(components['resolution'])
                print(f"Processing file with existing resolution: {file_resolution}")
        except ValueError:
            raise ValueError(f"Invalid file name format: {file_path.name}")

        # Determine resolution steps
        if resolutions is None:
            # Load one polygon to determine shape
            data = np.load(file_path, mmap_mode="r")
            vertex_count = data.shape[1]
            if len(data) == 0:
                raise ValueError(f"Empty file: {file_path}")
            del data

            resolutions = calculate_resolution_steps(vertex_count, self.min_vertices)

        print(f"Resolution steps: {resolutions}")

        # Create output directory
        output_dir = self._get_output_dir(path_manager)
        path_manager._ensure_dir(output_dir)

        # Process the file according to transformer-specific logic
        self._process_file_by_resolution(
            path_manager=path_manager,
            file_path=file_path,
            resolutions=resolutions,
            split=split,
            generator=generator,
            algorithm=algorithm,
            **kwargs
        )

    def _get_output_dir(self, path_manager: PathManager) -> Path:
        """
        Get the output directory for this transformer.

        By default, transforms go to the transformed directory.
        Subclasses can override this to use different directories.

        Args:
            path_manager: Path manager for the dataset.

        Returns:
            Path: Directory for output files.
        """
        return path_manager.get_transformed_dir()

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
        Process a file for each resolution.

        This method should be implemented by subclasses to handle
        the specific transformation logic.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            resolutions: List of resolutions to process.
            split: Dataset split.
            generator: Generator name.
            algorithm: Algorithm name.
            **kwargs: Additional parameters.

        Raises:
            NotImplementedError: By default, if a subclass doesn't implement it.
        """
        raise NotImplementedError(
            "Subclasses must implement _process_file_by_resolution or override _transform_file"
        )

    def _create_output_memmap(
            self,
            output_file: Path,
            shape: Tuple,
            dtype: str = 'float64'
    ) -> None:
        """
        Create a memory-mapped output file with given shape.

        Args:
            output_file: Path to output file.
            shape: Shape of the output array.
            dtype: Data type for the array.
        """
        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create memory-mapped file
        memmap = open_memmap(
            output_file,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        memmap.flush()
        del memmap  # Close immediately to free memory

    def _process_in_chunks(
            self,
            input_data: Union[np.ndarray, Path],
            output_file: Path,
            output_shape: Tuple,
            process_chunk_func: callable,
            chunk_size: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """
        Process a large dataset in chunks to manage memory.

        Args:
            input_data: Input data array or file path.
            output_file: Path to the output file.
            output_shape: Shape of the output array.
            process_chunk_func: Function to process each chunk.
                Should accept (chunk_data, start_idx, **kwargs) and
                return the processed chunk.
            chunk_size: Size of each chunk. Defaults to self.chunk_size.
            **kwargs: Additional parameters to pass to process_chunk_func.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Load data appropriately
        if isinstance(input_data, Path):
            data = np.load(input_data, mmap_mode='r')
        else:
            data = input_data

        total_items = len(data)

        # Create output file
        self._create_output_memmap(output_file, output_shape)

        # Process in chunks
        for chunk_start in range(0, total_items, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_items)
            chunk_size_actual = chunk_end - chunk_start

            print(f"  Processing chunk {chunk_start}-{chunk_end} "
                  f"({chunk_size_actual} items)")

            # Load chunk
            chunk_data = data[chunk_start:chunk_end].copy()

            # Process chunk
            processed_chunk = process_chunk_func(
                chunk_data=chunk_data,
                start_idx=chunk_start,
                **kwargs
            )

            # Write to output file
            memmap = open_memmap(
                output_file,
                dtype='float64',
                mode='r+',
                shape=output_shape
            )
            memmap[chunk_start:chunk_end] = processed_chunk
            memmap.flush()
            del memmap  # Close immediately to free memory

            # Clean up
            del chunk_data
            del processed_chunk
            gc.collect()

        # Close input file if it was loaded from disk
        if isinstance(input_data, Path):
            del data
            gc.collect()

        print(f"  Saved {total_items} items to {output_file}")