# polygon_dataset/core/path_manager.py
"""
Path management utilities for polygon datasets.

This module provides a PathManager class for handling file paths in a consistent
manner across the package.
"""

import json
from pathlib import Path
from typing import List, Optional, Union, Set, Dict, Any, Tuple

from numpy.lib.format import open_memmap

from polygon_dataset.utils.filename_parser import parse_polygon_filename


class PathManager:
    """
    Manages paths for dataset storage and access.

    This class provides a centralized way to generate paths to dataset files and
    directories, maintaining a consistent structure and optionally creating
    directories as needed.
    """

    def __init__(
            self,
            base_path: Union[str, Path],
            dataset_name: str,
            create_dirs: bool = False
    ) -> None:
        """
        Initialize the path manager.

        Args:
            base_path: Root directory for all datasets.
            dataset_name: Name of the dataset.
            create_dirs: Whether to create directories if they don't exist.
        """
        self.base_path: Path = Path(base_path)
        self.dataset_name: str = dataset_name
        self.dataset_path: Path = self.base_path / self.dataset_name
        self.create_dirs: bool = create_dirs

        if create_dirs:
            self.dataset_path.mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self, path: Path) -> Path:
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

    def get_dataset_root(self) -> Path:
        """
        Get the root directory for this dataset.

        Returns:
            Path: The dataset root directory.
        """
        return self._ensure_dir(self.dataset_path)

    def get_raw_dir(self) -> Path:
        """
        Get directory for raw polygon data.

        Returns:
            Path: Directory for raw polygon data.
        """
        return self._ensure_dir(self.dataset_path / "raw")

    def get_raw_split_dir(self, split: str, generator: str) -> Path:
        """
        Get directory for raw files of a specific split and generator.

        Args:
            split: Dataset split (train/val/test).
            generator: Generator name.

        Returns:
            Path: Directory for raw files of the specified split and generator.
        """
        return self._ensure_dir(self.get_raw_dir() / split / generator)

    def get_raw_paths(self, generator: str, split: str) -> List[Path]:
        """
        Get paths to all raw .line files for given generator and split.

        Args:
            generator: Generator name.
            split: Dataset split (train/val/test).

        Returns:
            List[Path]: List of paths to raw .line files.
        """
        split_dir = self.get_raw_split_dir(split, generator)
        if not split_dir.exists():
            return []
        return sorted(split_dir.glob("*.line"))

    def get_extracted_dir(self) -> Path:
        """
        Get directory for extracted dataset files.

        Returns:
            Path: Directory for extracted dataset files.
        """
        return self._ensure_dir(self.dataset_path / "extracted")

    def get_transformed_dir(self) -> Path:
        """
        Get directory for transformed dataset files.

        Returns:
            Path: Directory for transformed dataset files.
        """
        return self._ensure_dir(self.dataset_path / "transformed")

    def get_processed_path(self, generator: str, algorithm: str, split: str) -> Path:
        """
        Get path to the processed NPY file for given generator, algorithm, and split.

        Args:
            generator: Generator name.
            algorithm: Algorithm name.
            split: Dataset split (train/val/test).

        Returns:
            Path: Path to the processed NPY file.
        """
        filename = f"{split}_{generator}_{algorithm}.npy"
        return self.get_extracted_dir() / filename

    def get_resolution_path(
            self,
            generator: str,
            algorithm: str,
            split: str,
            resolution: int
    ) -> Path:
        """
        Get path to NPY file for given generator, algorithm, split, and resolution.

        Args:
            generator: Generator name.
            algorithm: Algorithm name.
            split: Dataset split (train/val/test).
            resolution: Number of vertices in the simplified polygon.

        Returns:
            Path: Path to the resolution-specific NPY file.
        """
        filename = f"{split}_{generator}_{algorithm}_res{resolution}.npy"
        return self.get_transformed_dir() / filename

    def get_canonical_dir(self) -> Path:
        """
        Get directory for canonicalized polygon data.

        Returns:
            Path: Directory for canonicalized polygon data.
        """
        return self._ensure_dir(self.dataset_path / "canonicalized")

    def get_canonical_path(
            self,
            generator: str,
            algorithm: str,
            split: str,
            resolution: Optional[int] = None
    ) -> Path:
        """
        Get path to the canonicalized NPY file.

        Args:
            generator: Generator name.
            algorithm: Algorithm name.
            split: Dataset split (train/val/test).
            resolution: Number of vertices in the simplified polygon (optional).

        Returns:
            Path: Path to the canonicalized NPY file.
        """
        if resolution:
            filename = f"{split}_{generator}_{algorithm}_res{resolution}.npy"
        else:
            filename = f"{split}_{generator}_{algorithm}.npy"
        return self.get_canonical_dir() / filename

    def get_config_path(self) -> Path:
        """
        Get path to the dataset configuration file.

        Returns:
            Path: Path to the dataset configuration file.
        """
        return self.dataset_path / "config.json"

    def get_available_algorithms(self, generator: str) -> Set[str]:
        """
        Get all available algorithms for a given generator in this dataset.

        Args:
            generator: Generator name.

        Returns:
            Set[str]: Set of available algorithm names.
        """
        algorithms = set()
        extraction_dir = self.get_extracted_dir()

        if not extraction_dir.exists():
            return algorithms

        # Search for matching files in the extraction directory
        pattern = f"*_{generator}_*.npy"
        for file_path in extraction_dir.glob(pattern):
            try:
                # Parse file name to extract algorithm
                components = parse_polygon_filename(file_path.name)
                if components['generator'] == generator:
                    algorithms.add(components['algorithm'])
            except ValueError:
                # Skip files that don't match the expected pattern
                continue

        return algorithms

    def create_output_memmap(
            self,
            output_file: Path,
            shape: Tuple,
            dtype: str = 'float64'
    ) -> None:
        """
        Create a memory-mapped output file with the given shape and dtype.

        This is a utility method for creating memory-mapped arrays in a consistent way.

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

    def find_npy_files(
            self,
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
                from polygon_dataset.utils.filename_parser import parse_polygon_filename

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

    def load_config(self) -> Dict[str, Any]:
        """
        Load the dataset configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the configuration file is invalid.
        """
        config_path = self.get_config_path()

        if not config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found at {config_path}")

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {config_path}")

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save the dataset configuration to a file.

        Args:
            config: Configuration dictionary to save.

        Raises:
            IOError: If writing the configuration file fails.
        """
        config_path = self.get_config_path()

        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except (IOError, OSError) as e:
            raise IOError(f"Error writing configuration to {config_path}: {e}")

    def update_dataset_state(self, config: Any, state: str) -> None:
        """
        Update the dataset state in the configuration file.

        Args:
            config: Configuration dictionary or Hydra configuration object.
            state: New state to set.

        Raises:
            IOError: If writing the configuration file fails.
        """
        # Load existing config if it exists
        if self.get_config_path().exists():
            dataset_config = self.load_config()
        else:
            # Create a new config if it doesn't exist
            dataset_config = {
                "dataset_info": {
                    "name": config.dataset.name,
                    "size": config.dataset.size,
                    "vertex_count": config.dataset.vertex_count,
                    "dimensionality": config.dataset.dimensionality,
                    "dataset_state": state,
                    "split_ratios": {
                        "train": config.dataset.split.train_ratio,
                        "val": config.dataset.split.val_ratio,
                        "test": config.dataset.split.test_ratio
                    }
                },
                "generator_configs": {},
                "transformation_config": {}
            }

        # Update the state
        dataset_config["dataset_info"]["dataset_state"] = state

        # Add generator configurations if not present
        for gen_config in config.generators:
            gen_name = f"{gen_config.name}_{gen_config.implementation}"
            if gen_name not in dataset_config["generator_configs"]:
                dataset_config["generator_configs"][gen_name] = {
                    "name": gen_config.name,
                    "implementation": gen_config.implementation,
                    "params": dict(gen_config.params) if hasattr(gen_config, "params") else {}
                }

        # Add transformation configuration if not present
        if hasattr(config, "transform") and config.transform:
            dataset_config["transformation_config"] = {
                "algorithm": config.transform.algorithm,
                "min_vertices": config.transform.min_vertices,
                "batch_size": config.transform.batch_size,
                "resolution_steps": list(config.transform.resolution_steps) if config.transform.resolution_steps else []
            }

        # Save the updated config
        self.save_config(dataset_config)