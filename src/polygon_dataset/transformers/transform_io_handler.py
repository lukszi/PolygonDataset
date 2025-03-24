# polygon_dataset/transformers/transform_io_handler.py
"""
File I/O handling for polygon transformations.

This module provides the TransformIOHandler class for handling file operations
during polygon transformations, including parsing filenames, loading and saving data.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from polygon_dataset.utils.filename_parser import parse_polygon_filename
from polygon_dataset.utils.chunking import load_chunk, create_empty_memory_mapped_file, write_chunk_to_file

if TYPE_CHECKING:
    from polygon_dataset.core import PathManager

# Configure module logger
logger = logging.getLogger(__name__)


class TransformIOHandler:
    """
    Handles file I/O operations for polygon transformations.

    This class manages loading, parsing, and saving of polygon data during
    transformations, including chunked processing of large files.
    """

    def __init__(self, output_dir_getter: callable = None) -> None:
        """
        Initialize the transform I/O handler.

        Args:
            output_dir_getter: Function that returns the output directory for a given
                PathManager. If None, uses a default implementation.
        """
        self.output_dir_getter = output_dir_getter or self._default_output_dir_getter

    def find_input_files(
            self,
            path_manager: "PathManager",
            source_dir: Path,
            generator_filter: Optional[str] = None,
            algorithm_filter: Optional[str] = None,
            split_filter: Optional[str] = None
    ) -> List[Path]:
        """
        Find input files matching specified criteria.

        Args:
            path_manager: Path manager for the dataset.
            source_dir: Source directory to search.
            generator_filter: Filter by generator name.
            algorithm_filter: Filter by algorithm name.
            split_filter: Filter by split name.

        Returns:
            List of matching file paths.

        Raises:
            ValueError: If no matching files are found.
        """
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

        logger.info(f"Found {len(npy_files)} files to process")
        return npy_files

    def process_file(
            self,
            path_manager: "PathManager",
            file_path: Path,
            transformer: callable,
            chunk_size: int,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> None:
        """
        Process a single file, transforming its contents in chunks.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            transformer: Function that transforms polygon data.
            chunk_size: Size of chunks for processing.
            resolutions: List of target resolutions.
            **kwargs: Additional parameters for the transformer.

        Raises:
            ValueError: If file parsing or transformation fails.
        """
        logger.info(f"Processing file: {file_path.name}")

        # Parse file name and extract metadata
        components = self._parse_file_name(file_path)
        split, generator, algorithm, file_resolution = components

        # Load file metadata
        data_shape, total_polygons, vertex_count = self._load_file_metadata(file_path)

        # Process file based on output type
        output_type = kwargs.get("output_type", "single")

        if output_type == "multi" and resolutions is not None:
            self._process_multi_resolution_file(
                path_manager, file_path, transformer, chunk_size,
                split, generator, algorithm,
                resolutions, vertex_count, total_polygons,
                **kwargs
            )
        else:
            self._process_single_resolution_file(
                path_manager, file_path, transformer, chunk_size,
                split, generator, algorithm, file_resolution,
                data_shape, total_polygons,
                **kwargs
            )

    def _parse_file_name(self, file_path: Path) -> Tuple[str, str, str, Optional[int]]:
        """
        Parse components from a file name.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple containing split, generator, algorithm, and resolution (if any).

        Raises:
            ValueError: If file name format is invalid.
        """
        try:
            components = parse_polygon_filename(file_path.name)
            split = components['split']
            generator = components['generator']
            algorithm = components['algorithm']
            file_resolution = None

            if components['resolution'] is not None:
                file_resolution = int(components['resolution'])
                logger.info(f"Processing file with existing resolution: {file_resolution}")

            return split, generator, algorithm, file_resolution

        except ValueError:
            raise ValueError(f"Invalid file name format: {file_path.name}")

    def _load_file_metadata(self, file_path: Path) -> Tuple[Tuple, int, int]:
        """
        Load metadata about the file without loading all data.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple containing data shape, total polygons, and vertex count.

        Raises:
            ValueError: If file is empty.
        """
        data = np.load(file_path, mmap_mode="r")

        if len(data) == 0:
            raise ValueError(f"Empty file: {file_path}")

        shape = data.shape
        total_polygons = len(data)
        vertex_count = data.shape[1]

        del data  # Release memory
        return shape, total_polygons, vertex_count

    def _process_multi_resolution_file(
            self,
            path_manager: "PathManager",
            file_path: Path,
            transformer: callable,
            chunk_size: int,
            split: str,
            generator: str,
            algorithm: str,
            resolutions: List[int],
            vertex_count: int,
            total_polygons: int,
            **kwargs: Any
    ) -> None:
        """
        Process a file for multiple resolutions.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            transformer: Function that transforms polygon data.
            chunk_size: Size of chunks for processing.
            split: Dataset split.
            generator: Generator name.
            algorithm: Algorithm name.
            resolutions: List of target resolutions.
            vertex_count: Number of vertices per polygon.
            total_polygons: Total number of polygons.
            **kwargs: Additional parameters.
        """
        # Determine output directory
        output_dir = self.output_dir_getter(path_manager)

        # Filter resolutions to those valid for this vertex count
        strategy = kwargs.get("strategy")
        if strategy:
            filtered_resolutions = strategy.filter_resolutions(resolutions, vertex_count)
        else:
            # Default filtering if no strategy is provided
            filtered_resolutions = [r for r in resolutions if r <= vertex_count]

        if not filtered_resolutions:
            logger.warning(f"No valid resolutions for vertex count {vertex_count}")
            return

        logger.info(f"Processing resolutions: {filtered_resolutions}")

        # Create output files
        output_paths = {}
        for resolution in filtered_resolutions:
            output_file = path_manager.get_resolution_path(
                generator=generator,
                algorithm=algorithm,
                split=split,
                resolution=resolution
            )

            # Create memory-mapped file
            create_empty_memory_mapped_file(
                output_file=output_file,
                shape=(total_polygons, resolution, 2)
            )

            output_paths[resolution] = output_file

        # Process the file in chunks
        self._process_file_in_chunks(
            file_path, output_paths, transformer, chunk_size,
            filtered_resolutions, total_polygons, **kwargs
        )

    def _process_single_resolution_file(
            self,
            path_manager: "PathManager",
            file_path: Path,
            transformer: callable,
            chunk_size: int,
            split: str,
            generator: str,
            algorithm: str,
            file_resolution: Optional[int],
            data_shape: Tuple,
            total_polygons: int,
            **kwargs: Any
    ) -> None:
        """
        Process a file for a single resolution.

        Args:
            path_manager: Path manager for the dataset.
            file_path: Path to the input file.
            transformer: Function that transforms polygon data.
            chunk_size: Size of chunks for processing.
            split: Dataset split.
            generator: Generator name.
            algorithm: Algorithm name.
            file_resolution: Resolution in the file name (if any).
            data_shape: Shape of the input data.
            total_polygons: Total number of polygons.
            **kwargs: Additional parameters.
        """
        # Determine output directory
        output_dir = self.output_dir_getter(path_manager)

        # Determine output file path
        if file_resolution is None:
            output_file = path_manager.get_canonical_path(
                generator=generator,
                algorithm=algorithm,
                split=split
            )
        else:
            output_file = path_manager.get_canonical_path(
                generator=generator,
                algorithm=algorithm,
                split=split,
                resolution=file_resolution
            )

        # Create memory-mapped output file
        create_empty_memory_mapped_file(
            output_file=output_file,
            shape=data_shape
        )

        # Process the file in chunks
        self._process_file_in_chunks_single(
            file_path, output_file, transformer, chunk_size,
            data_shape, total_polygons, **kwargs
        )

    def _process_file_in_chunks(
            self,
            file_path: Path,
            output_paths: Dict[int, Path],
            transformer: callable,
            chunk_size: int,
            resolutions: List[int],
            total_polygons: int,
            **kwargs: Any
    ) -> None:
        """
        Process a file in chunks for multiple resolutions.

        Args:
            file_path: Path to the input file.
            output_paths: Dictionary mapping resolutions to output file paths.
            transformer: Function that transforms polygon data.
            chunk_size: Size of chunks for processing.
            resolutions: List of target resolutions.
            total_polygons: Total number of polygons.
            **kwargs: Additional parameters.
        """
        for chunk_start in range(0, total_polygons, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_polygons)
            chunk_size_actual = chunk_end - chunk_start

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

            # Load and process chunk
            chunk_data = load_chunk(file_path, chunk_start, chunk_size_actual)

            try:
                result = transformer(chunk_data, resolutions=resolutions, **kwargs)

                # Write results to output files
                for resolution, data in result.items():
                    if resolution in output_paths:
                        write_chunk_to_file(
                            output_file=output_paths[resolution],
                            data=data,
                            chunk_start=chunk_start,
                            shape=(total_polygons, resolution, 2)
                        )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                raise

            # Clean up
            del chunk_data
            gc.collect()

    def _process_file_in_chunks_single(
            self,
            file_path: Path,
            output_file: Path,
            transformer: callable,
            chunk_size: int,
            data_shape: Tuple,
            total_polygons: int,
            **kwargs: Any
    ) -> None:
        """
        Process a file in chunks for a single output.

        Args:
            file_path: Path to the input file.
            output_file: Path to the output file.
            transformer: Function that transforms polygon data.
            chunk_size: Size of chunks for processing.
            data_shape: Shape of the input/output data.
            total_polygons: Total number of polygons.
            **kwargs: Additional parameters.
        """
        for chunk_start in range(0, total_polygons, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_polygons)
            chunk_size_actual = chunk_end - chunk_start

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

            # Load and process chunk
            chunk_data = load_chunk(file_path, chunk_start, chunk_size_actual)

            try:
                result = transformer(chunk_data, **kwargs)

                # Write result to output file
                write_chunk_to_file(
                    output_file=output_file,
                    data=result,
                    chunk_start=chunk_start,
                    shape=data_shape
                )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                raise

            # Clean up
            del chunk_data
            gc.collect()

    @staticmethod
    def _default_output_dir_getter(path_manager: "PathManager") -> Path:
        """
        Default implementation for getting the output directory.

        Args:
            path_manager: Path manager for the dataset.

        Returns:
            Path: Directory for output files.
        """
        # Default is transformed directory
        return path_manager.get_transformed_dir()