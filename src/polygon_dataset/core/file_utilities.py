# polygon_dataset/core/file_utilities.py
"""
File operation utilities for polygon datasets.

This module provides utility classes for file operations within the polygon
datasets package, including directory creation, file search, and memory-mapped file
operations.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np
from numpy.lib.format import open_memmap

from polygon_dataset.utils.filename_parser import parse_polygon_filename

# Configure module logger
logger = logging.getLogger(__name__)


class DirectoryManager:
    """
    Handles directory creation and validation for dataset operations.

    This class ensures that necessary directories exist before operations
    that require them.
    """

    def __init__(self, create_dirs: bool = False) -> None:
        """
        Initialize the directory manager.

        Args:
            create_dirs: Whether to create directories if they don't exist.
        """
        self.create_dirs: bool = create_dirs

    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure directory exists if create_dirs is True.

        Args:
            path: Directory path to check/create.

        Returns:
            Path: The input path, potentially newly created.
        """
        if self.create_dirs and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path


class FileLocator:
    """
    Provides file search and filtering capabilities for datasets.

    This class helps locate NPY files that match specific criteria such as
    generator, algorithm, or split.
    """

    @staticmethod
    def find_npy_files(
            directory: Path,
            pattern: str = "*.npy",
            generator_filter: Optional[str] = None,
            algorithm_filter: Optional[str] = None,
            split_filter: Optional[str] = None,
            resolution_filter: Optional[int] = None
    ) -> List[Path]:
        """
        Find and filter NPY files in a directory based on criteria.

        Args:
            directory: Directory to search in.
            pattern: Glob pattern for files.
            generator_filter: Filter by generator name.
            algorithm_filter: Filter by algorithm name.
            split_filter: Filter by split name.
            resolution_filter: Filter by resolution.

        Returns:
            List[Path]: List of matching file paths.

        Raises:
            ValueError: If the directory doesn't exist.
        """
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Find files matching the pattern
        npy_files = list(directory.glob(pattern))

        # Apply filters if any are provided
        if not (generator_filter or algorithm_filter or split_filter or resolution_filter):
            return npy_files

        filtered_files = []
        for file_path in npy_files:
            # Try to parse filename
            try:
                components = parse_polygon_filename(file_path.name)

                # Apply filters
                if (split_filter and components['split'] != split_filter) or \
                        (generator_filter and components['generator'] != generator_filter) or \
                        (algorithm_filter and components['algorithm'] != algorithm_filter):
                    continue

                # Check resolution if applicable
                if resolution_filter is not None:
                    if components['resolution'] is None or int(components['resolution']) != resolution_filter:
                        continue

                filtered_files.append(file_path)

            except (ValueError, KeyError):
                # Skip files that don't match the expected pattern
                continue

        return filtered_files

    @staticmethod
    def get_available_algorithms(directory: Path, generator: str) -> set:
        """
        Get all available algorithms for a given generator in a directory.

        Args:
            directory: Directory to search in (typically the extracted directory).
            generator: Generator name to filter by.

        Returns:
            set: Set of available algorithm names.
        """
        algorithms = set()

        if not directory.exists():
            return algorithms

        # Search for matching files in the directory
        pattern = f"*_{generator}_*.npy"
        for file_path in directory.glob(pattern):
            try:
                # Parse file name to extract algorithm
                components = parse_polygon_filename(file_path.name)
                if components['generator'] == generator:
                    algorithms.add(components['algorithm'])
            except ValueError:
                # Skip files that don't match the expected pattern
                continue

        return algorithms


class MemoryMappedFileManager:
    """
    Manages creation and access of memory-mapped files for large dataset operations.

    This class provides utilities for creating and writing to memory-mapped files,
    which are useful for processing large datasets that don't fit in memory.
    """

    @staticmethod
    def create_memory_mapped_file(
            output_file: Path,
            shape: Tuple,
            dtype: str = 'float64'
    ) -> None:
        """
        Create a memory-mapped output file with the given shape and dtype.

        This method creates a memory-mapped array file that can be accessed
        without loading the entire array into memory.

        Args:
            output_file: Path to the output file.
            shape: Shape of the output array.
            dtype: Data type for the output array.

        Note:
            This method creates the file and immediately closes it to free memory.
        """
        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create the memmap file
        memmap = open_memmap(
            output_file,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        memmap.flush()
        del memmap  # Close immediately to free memory

    @staticmethod
    def write_chunk_to_memmap(
            output_file: Path,
            data: np.ndarray,
            start_index: int,
            shape: Tuple,
            dtype: str = 'float64'
    ) -> None:
        """
        Write a chunk of data to a memory-mapped file.

        Args:
            output_file: Path to the memory-mapped file.
            data: Data to write to the file.
            start_index: Starting index for writing the data.
            shape: Total shape of the memory-mapped file.
            dtype: Data type of the memory-mapped file.
        """
        # Open the existing memory-mapped file in read-write mode
        memmap = open_memmap(output_file, dtype=dtype, mode='r+', shape=shape)

        # Write the data to the specified location
        end_index = start_index + len(data)
        memmap[start_index:end_index] = data

        # Flush to ensure data is written to disk
        memmap.flush()
        del memmap  # Close immediately to free memory