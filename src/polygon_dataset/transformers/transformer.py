# polygon_dataset/transformers/transformer.py
"""
Main transformer interface for polygon transformations.

This module provides the Transformer class, which integrates the specialized components
for transforming polygon data and serves as the main entry point for transformations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

from polygon_dataset.utils import calculate_resolution_steps
from polygon_dataset.transformers.polygon_transformer import PolygonTransformer
from polygon_dataset.transformers.transform_processor import TransformProcessor
from polygon_dataset.transformers.transform_io_handler import TransformIOHandler
from polygon_dataset.transformers.strategy import TransformerStrategy

if TYPE_CHECKING:
    from polygon_dataset.core import PathManager

# Configure module logger
logger = logging.getLogger(__name__)


class Transformer:
    """
    Main interface for transforming polygon data.

    This class integrates the specialized components for transforming polygon data,
    including the core transformation logic, parallel processing, and file I/O.
    """

    def __init__(
            self,
            strategy: TransformerStrategy,
            chunk_size: int = 100000,
            min_vertices: int = 10,
            num_processes: Optional[int] = None
    ) -> None:
        """
        Initialize the transformer.

        Args:
            strategy: The transformation strategy to use.
            chunk_size: Size of chunks for batch processing.
            min_vertices: Minimum number of vertices (for simplification).
            num_processes: Number of processes for parallel processing.
                If None, uses CPU count - 1.
        """
        # Initialize components
        self.polygon_transformer = PolygonTransformer(strategy, min_vertices)
        self.processor = TransformProcessor(
            self.polygon_transformer, chunk_size, num_processes
        )
        self.io_handler = TransformIOHandler(self._get_output_dir)

        # Store strategy for direct access
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.min_vertices = min_vertices

        # Setup output_type based on strategy
        self.output_type = strategy.get_output_type()

    def transform(
            self,
            polygons: np.ndarray,
            resolutions: Optional[List[int]] = None,
            **kwargs: Any
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Transform a batch of polygons.

        Args:
            polygons: Array of polygons with shape [num_polygons, vertices, 2].
            resolutions: List of target resolutions (for simplification algorithms).
            **kwargs: Additional parameters passed to the strategy.

        Returns:
            Transformed polygons, either as a single array or a dictionary mapping
            resolutions to arrays.
        """
        return self.processor.process(polygons, resolutions, **kwargs)

    def transform_dataset(
            self,
            path_manager: "PathManager",
            resolutions: Optional[List[int]] = None,
            generator_filter: Optional[str] = None,
            algorithm_filter: Optional[str] = None,
            split_filter: Optional[str] = None,
            source_dir: Optional[Path] = None,
            **kwargs: Any
    ) -> None:
        """
        Transform an entire dataset of polygons.

        Args:
            path_manager: Path manager for the dataset.
            resolutions: List of target resolutions (for simplification).
            generator_filter: Filter by generator name.
            algorithm_filter: Filter by algorithm name.
            split_filter: Filter by split name.
            source_dir: Source directory for input files.
            **kwargs: Additional parameters.

        Raises:
            ValueError: If no matching files are found or transformation fails.
        """
        # Determine source directory
        if source_dir is None:
            source_dir = path_manager.get_extracted_dir()

        logger.info(f"Transforming dataset using {self.strategy.name}")

        # Find input files
        npy_files = self.io_handler.find_input_files(
            path_manager, source_dir,
            generator_filter, algorithm_filter, split_filter
        )

        # Calculate resolutions if not provided and needed
        if resolutions is None and self.output_type == "multi":
            # We need to get vertex count from the first file
            try:
                _, _, vertex_count = self.io_handler._load_file_metadata(npy_files[0])
                resolutions = calculate_resolution_steps(vertex_count, self.min_vertices)
                logger.info(f"Calculated resolution steps: {resolutions}")
            except Exception as e:
                logger.error(f"Error calculating resolutions: {e}")
                raise

        # Process each file
        for file_path in npy_files:
            self.io_handler.process_file(
                path_manager=path_manager,
                file_path=file_path,
                transformer=self.transform,
                chunk_size=self.chunk_size,
                resolutions=resolutions,
                output_type=self.output_type,
                strategy=self.strategy,
                **kwargs
            )

    def _get_output_dir(self, path_manager: "PathManager") -> Path:
        """
        Get the output directory for this transformer.

        Args:
            path_manager: Path manager for the dataset.

        Returns:
            Path: Directory for output files.
        """
        if self.strategy.name == "canonicalize":
            return path_manager.get_canonical_dir()
        else:
            return path_manager.get_transformed_dir()