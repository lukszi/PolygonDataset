# polygon_dataset/core/dataset.py
"""
Polygon dataset access API.

This module provides functionality for accessing and manipulating polygon datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

import numpy as np

from polygon_dataset.core.path_manager import PathManager


class PolygonDataset:
    """
    Interface for accessing polygon datasets.

    This class provides methods for loading and accessing polygon data from
    a dataset directory, with support for different splits, generators, algorithms,
    resolutions, and canonicalization options.
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """
        Initialize the dataset accessor.

        Args:
            dataset_path: Path to the dataset directory.

        Raises:
            FileNotFoundError: If the dataset directory doesn't exist.
        """
        self.path: Path = Path(dataset_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.path}")

        self.path_manager = PathManager(self.path.parent, self.path.name)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        Load dataset metadata from configuration file.

        This method loads and parses the dataset configuration file to extract
        metadata such as dataset name, state, vertex count, and available
        generators.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the configuration file is invalid.
        """
        config_path = self.path_manager.get_config_path()

        if not config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found at {config_path}")

        with open(config_path, 'r') as f:
            try:
                self.config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in configuration file {config_path}")

        # Extract key information
        dataset_info = self.config.get("dataset_info", {})
        self.name: str = dataset_info.get("name", self.path.name)
        self.state: str = dataset_info.get("dataset_state", "unknown")
        self.vertex_count: int = dataset_info.get("vertex_count", 0)

        # Get available generators
        self.generators: List[str] = list(self.config.get("generator_configs", {}).keys())

        # Load transformation information
        transform_config = self.config.get("transformation_config", {})
        self.resolution_steps: List[int] = transform_config.get("resolution_steps", [])

    def get_generators(self) -> List[str]:
        """
        Get all available generators in this dataset.

        Returns:
            List[str]: List of available generator names.
        """
        return self.generators

    def get_algorithms(self, generator: str) -> Set[str]:
        """
        Get all available algorithms for a given generator in this dataset.

        Args:
            generator: Generator name.

        Returns:
            Set[str]: Set of available algorithm names.
        """
        return self.path_manager.get_available_algorithms(generator)

    def get_resolutions(self) -> List[int]:
        """
        Get available resolution steps for this dataset.

        Returns:
            List[int]: List of available resolution steps.
        """
        return self.resolution_steps

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.

        Returns:
            Dict[str, Any]: Dictionary of dataset metadata.
        """
        return {
            "name": self.name,
            "state": self.state,
            "vertex_count": self.vertex_count,
            "generators": self.generators,
            "resolution_steps": self.resolution_steps
        }

    def get_polygons(
            self,
            split: str = "train",
            generator: Optional[str] = None,
            algorithm: Optional[str] = None,
            resolution: Optional[int] = None,
            canonicalized: bool = False
    ) -> np.ndarray:
        """
        Get polygons with specified properties.

        Args:
            split: Dataset split ('train', 'val', or 'test').
            generator: Generator name (if None, uses first available).
            algorithm: Algorithm name (if None, uses first available for the generator).
            resolution: Vertex count (if None, uses original resolution).
            canonicalized: Whether to get canonicalized polygons.

        Returns:
            np.ndarray: Array of polygons with shape [num_polygons, vertices, 2].

        Raises:
            ValueError: If the requested generator, algorithm, or resolution is not available.
            FileNotFoundError: If the requested polygon data file doesn't exist.
        """
        # Validate and set default generator if needed
        if not generator:
            if not self.generators:
                raise ValueError("No generators available in this dataset")
            generator = self.generators[0]
        elif generator not in self.generators:
            raise ValueError(f"Generator '{generator}' not available in this dataset")

        # Validate and set default algorithm if needed
        available_algorithms = self.get_algorithms(generator)
        if not algorithm:
            if not available_algorithms:
                raise ValueError(f"No algorithms available for generator '{generator}'")
            algorithm = next(iter(available_algorithms))
        elif algorithm not in available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not available for generator '{generator}'")

        # Validate resolution if specified
        if resolution and self.resolution_steps and resolution not in self.resolution_steps:
            raise ValueError(
                f"Resolution {resolution} not available. Available resolutions: {self.resolution_steps}"
            )

        # Determine the appropriate file path
        if canonicalized:
            path = self.path_manager.get_canonical_path(generator, algorithm, split, resolution)
        elif resolution:
            path = self.path_manager.get_resolution_path(generator, algorithm, split, resolution)
        else:
            path = self.path_manager.get_processed_path(generator, algorithm, split)

        if not path.exists():
            raise FileNotFoundError(
                f"Polygon data not found at {path}. Make sure the dataset contains "
                f"the requested {split} split for generator '{generator}', algorithm '{algorithm}', "
                f"{'with' if canonicalized else 'without'} canonicalization, "
                f"and resolution {resolution if resolution else 'original'}."
            )

        # Load and return the data
        return np.load(path)

    def get_polygon_count(
            self,
            split: str = "train",
            generator: Optional[str] = None,
            algorithm: Optional[str] = None
    ) -> int:
        """
        Get the number of polygons in a specific split, generator, and algorithm.

        Args:
            split: Dataset split ('train', 'val', or 'test').
            generator: Generator name (if None, uses first available).
            algorithm: Algorithm name (if None, uses first available for the generator).

        Returns:
            int: Number of polygons.

        Raises:
            Same exceptions as get_polygons().
        """
        # Load just the shape information without loading all the data
        polygons = self.get_polygons(split, generator, algorithm)
        return len(polygons)