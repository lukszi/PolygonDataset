# polygon_dataset/core/dataset_paths.py
"""
Dataset path management for polygon datasets.

This module provides the DatasetPaths class for constructing consistent
paths for dataset resources, maintaining a standard directory structure.
"""

from pathlib import Path
from typing import Optional, Union


class DatasetPaths:
    """
    Constructs consistent paths for dataset resources.

    This class provides methods to generate standardized paths for various
    components of a polygon dataset, ensuring a consistent directory structure.
    """

    def __init__(
            self,
            base_path: Union[str, Path],
            dataset_name: str,
            legacy: bool = False
    ) -> None:
        """
        Initialize the dataset paths.

        Args:
            base_path: Root directory for all datasets.
            dataset_name: Name of the dataset.
            legacy: When ``True``, construct old-format (pre-Hydra) filenames
                that omit the algorithm token, e.g. ``train_spg_res11.npy``
                instead of ``train_spg_2opt_res11.npy``. Defaults to ``False``
                (current/new-format behaviour).
        """
        self.base_path: Path = Path(base_path)
        self.dataset_name: str = dataset_name
        self.dataset_path: Path = self.base_path / self.dataset_name
        self.legacy: bool = legacy

    def get_dataset_root(self) -> Path:
        """
        Get the root directory for this dataset.

        Returns:
            Path: The dataset root directory.
        """
        return self.dataset_path

    def get_raw_dir(self) -> Path:
        """
        Get directory for raw polygon data.

        Returns:
            Path: Directory for raw polygon data.
        """
        return self.dataset_path / "raw"

    def get_raw_split_dir(self, split: str, generator: str) -> Path:
        """
        Get directory for raw files of a specific split and generator.

        Args:
            split: Dataset split (train/val/test).
            generator: Generator name.

        Returns:
            Path: Directory for raw files of the specified split and generator.
        """
        return self.get_raw_dir() / split / generator

    def get_extracted_dir(self) -> Path:
        """
        Get directory for extracted dataset files.

        Returns:
            Path: Directory for extracted dataset files.
        """
        return self.dataset_path / "extracted"

    def get_transformed_dir(self) -> Path:
        """
        Get directory for transformed dataset files.

        Returns:
            Path: Directory for transformed dataset files.
        """
        return self.dataset_path / "transformed"

    def get_canonical_dir(self) -> Path:
        """
        Get directory for canonicalized polygon data.

        Returns:
            Path: Directory for canonicalized polygon data.
        """
        return self.dataset_path / "canonicalized"

    def get_config_path(self) -> Path:
        """
        Get path to the dataset configuration file.

        Returns:
            Path: Path to the dataset configuration file.
        """
        return self.dataset_path / "config.json"

    def get_processed_path(self, generator: str, algorithm: str, split: str) -> Path:
        """
        Get path to the processed NPY file for given generator, algorithm, and split.

        Args:
            generator: Generator name.
            algorithm: Algorithm name.
            split: Dataset split (train/val/test).

        Returns:
            Path: Path to the processed NPY file.

        Note:
            In legacy mode the ``algorithm`` argument is accepted for signature
            compatibility but ignored, producing ``{split}_{generator}.npy``.
        """
        if self.legacy:
            filename = f"{split}_{generator}.npy"
        else:
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

        Note:
            In legacy mode the ``algorithm`` argument is accepted for signature
            compatibility but ignored, producing ``{split}_{generator}_res{N}.npy``.
        """
        if self.legacy:
            filename = f"{split}_{generator}_res{resolution}.npy"
        else:
            filename = f"{split}_{generator}_{algorithm}_res{resolution}.npy"
        return self.get_transformed_dir() / filename

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

        Note:
            In legacy mode the ``algorithm`` argument is accepted for signature
            compatibility but ignored.
        """
        if self.legacy:
            if resolution:
                filename = f"{split}_{generator}_res{resolution}.npy"
            else:
                filename = f"{split}_{generator}.npy"
        elif resolution:
            filename = f"{split}_{generator}_{algorithm}_res{resolution}.npy"
        else:
            filename = f"{split}_{generator}_{algorithm}.npy"
        return self.get_canonical_dir() / filename

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
        return f"{generator_name}_{implementation}"