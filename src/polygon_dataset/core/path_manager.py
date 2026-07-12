# polygon_dataset/core/path_manager.py
"""
Path management for polygon datasets.

This module provides a unified interface for managing dataset paths, file operations,
and configuration, integrating the specialized components.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from polygon_dataset.core.dataset_paths import DatasetPaths
from polygon_dataset.core.file_utilities import DirectoryManager, FileLocator, MemoryMappedFileManager
from polygon_dataset.core.config_manager import ConfigManager


class PathManager:
    """
    Manages paths and file operations for polygon datasets.

    This class provides a unified interface for path management, file operations,
    and configuration, delegating to specialized components for each responsibility.
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
        self.dataset_paths = DatasetPaths(base_path, dataset_name)
        self.directory_manager = DirectoryManager(create_dirs)
        self.config_manager = ConfigManager(self.dataset_paths.get_config_path())
        self.file_locator = FileLocator()
        self.memmap_manager = MemoryMappedFileManager()

        # Store base dataset properties for easy access
        self.base_path: Path = Path(base_path)
        self.dataset_name: str = dataset_name
        self.dataset_path: Path = self.base_path / self.dataset_name
        self.create_dirs: bool = create_dirs

        # Auto-detect old-format ("legacy") datasets from the on-disk config so
        # path construction matches their algorithm-token-free filenames. New or
        # not-yet-written datasets default to the current naming scheme.
        self._legacy: bool = self._detect_legacy_mode()
        self.dataset_paths.legacy = self._legacy

        # Create the dataset directory if requested
        if create_dirs:
            self.directory_manager.ensure_dir(self.dataset_path)

    @property
    def legacy(self) -> bool:
        """
        Whether this dataset uses the old-format (pre-Hydra) naming scheme.

        When ``True``, path construction omits the algorithm token (e.g.
        ``train_spg_res11.npy`` rather than ``train_spg_2opt_res11.npy``) and
        the ``algorithm`` argument of the path methods is accepted but ignored.
        Detected automatically from ``config.json`` at construction time.
        """
        return self._legacy

    def _detect_legacy_mode(self) -> bool:
        """
        Detect whether the dataset on disk uses the old-format naming scheme.

        Returns:
            bool: ``True`` for an old-format dataset, ``False`` otherwise
            (including when no config is present or it cannot be parsed).
        """
        try:
            config = self.config_manager.load_config()
        except (FileNotFoundError, ValueError):
            return False
        return self._is_legacy_config(config)

    @staticmethod
    def _is_legacy_config(config: Dict[str, Any]) -> bool:
        """
        Classify a loaded dataset config as old-format ("legacy") or new-format.

        New-format configs (written by the Hydra pipeline) key
        ``generator_configs`` by full generator name and store per-generator
        details under ``implementation`` / ``params``. Old-format configs (e.g.
        the v7 dataset) key it by bare generator name (``spg``/``fpg``/``rpg``)
        and carry a top-level ``generator_type`` field instead.

        Args:
            config: The loaded configuration dictionary.

        Returns:
            bool: ``True`` when the config matches the old format.
        """
        gen_cfgs = config.get("generator_configs")
        if not isinstance(gen_cfgs, dict) or not gen_cfgs:
            return False

        entries = [e for e in gen_cfgs.values() if isinstance(e, dict)]
        # Any new-format marker is decisive: treat as current naming.
        if any("implementation" in e or "params" in e for e in entries):
            return False
        # Otherwise require a positive old-format marker to switch modes.
        return any("generator_type" in e for e in entries)

    # Directory access methods with directory creation support
    def get_dataset_root(self) -> Path:
        """Get the dataset root directory, creating it if needed."""
        return self.directory_manager.ensure_dir(self.dataset_paths.get_dataset_root())

    def get_raw_dir(self) -> Path:
        """Get the raw data directory, creating it if needed."""
        return self.directory_manager.ensure_dir(self.dataset_paths.get_raw_dir())

    def get_raw_split_dir(self, split: str, generator: str) -> Path:
        """Get the raw split directory, creating it if needed."""
        return self.directory_manager.ensure_dir(
            self.dataset_paths.get_raw_split_dir(split, generator)
        )

    def get_extracted_dir(self) -> Path:
        """Get the extracted data directory, creating it if needed."""
        return self.directory_manager.ensure_dir(self.dataset_paths.get_extracted_dir())

    def get_transformed_dir(self) -> Path:
        """Get the transformed data directory, creating it if needed."""
        return self.directory_manager.ensure_dir(self.dataset_paths.get_transformed_dir())

    def get_canonical_dir(self) -> Path:
        """Get the canonicalized data directory, creating it if needed."""
        return self.directory_manager.ensure_dir(self.dataset_paths.get_canonical_dir())

    # File path methods (delegated to dataset_paths)
    def get_config_path(self) -> Path:
        """Get the configuration file path."""
        return self.dataset_paths.get_config_path()

    def get_processed_path(self, generator: str, algorithm: str, split: str) -> Path:
        """Get the processed data file path."""
        return self.dataset_paths.get_processed_path(generator, algorithm, split)

    def get_resolution_path(
            self,
            generator: str,
            algorithm: str,
            split: str,
            resolution: int
    ) -> Path:
        """Get the resolution-specific data file path."""
        return self.dataset_paths.get_resolution_path(generator, algorithm, split, resolution)

    def get_canonical_path(
            self,
            generator: str,
            algorithm: str,
            split: str,
            resolution: Optional[int] = None
    ) -> Path:
        """Get the canonicalized data file path."""
        return self.dataset_paths.get_canonical_path(generator, algorithm, split, resolution)

    # Raw file paths access
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

    # Generator and algorithm discovery
    def get_available_algorithms(self, generator: str) -> Set[str]:
        """
        Get all available algorithms for a given generator in this dataset.

        Args:
            generator: Generator name.

        Returns:
            Set[str]: Set of available algorithm names.
        """
        return self.file_locator.get_available_algorithms(
            self.get_extracted_dir(), generator
        )

    # File search and filtering
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
        """
        return self.file_locator.find_npy_files(
            directory, pattern, generator_filter, algorithm_filter,
            split_filter, resolution_filter
        )

    # Memory-mapped file operations
    def create_output_memmap(
            self,
            output_file: Path,
            shape: Tuple,
            dtype: str = 'float64'
    ) -> None:
        """
        Create a memory-mapped output file with the given shape and dtype.

        Args:
            output_file: Path to the output file.
            shape: Shape of the output array.
            dtype: Data type for the output array.
        """
        self.memmap_manager.create_memory_mapped_file(output_file, shape, dtype)

    # Configuration operations
    def load_config(self) -> Dict[str, Any]:
        """
        Load the dataset configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        return self.config_manager.load_config()

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save the dataset configuration to a file.

        Args:
            config: Configuration dictionary to save.
        """
        self.config_manager.save_config(config)

    def update_dataset_state(self, config: Any, state: str) -> None:
        """
        Update the dataset state in the configuration file.

        Args:
            config: Configuration object with necessary attributes.
            state: New state to set.
        """
        self.config_manager.update_dataset_state(config, state)

    # Helper methods
    @staticmethod
    def get_full_generator_name(generator_name: str, implementation: str) -> str:
        """
        Get the full generator name including implementation.

        Args:
            generator_name: Base generator name (e.g., 'rpg').
            implementation: Implementation type (e.g., 'binary', 'native').

        Returns:
            str: Full generator name (e.g., 'rpg_binary').
        """
        return DatasetPaths.get_full_generator_name(generator_name, implementation)
