# polygon_dataset/generators/base.py
"""
Base class for polygon generators.

This module provides the abstract base class for all polygon generators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from polygon_dataset.core import PathManager


class Generator(ABC):
    """
    Abstract base class for polygon generators.

    This class defines the interface that all polygon generators must implement,
    providing a consistent API for generating polygons regardless of the underlying
    implementation (binary or native).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the generator with configuration parameters.

        Args:
            config: Configuration parameters for the generator.
        """
        self.config: Dict[str, Any] = config
        self.name: str = config.get("name", "unknown")
        self.implementation: str = config.get("implementation", "unknown")
        self.params: Dict[str, Any] = config.get("params", {})

    @abstractmethod
    def generate(
            self,
            path_manager: PathManager,
            vertex_count: int,
            num_samples: int,
            split_ratios: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Generate polygon samples according to the configuration.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            num_samples: Total number of samples to generate.
            split_ratios: Dictionary mapping split names to their ratios.
                Default is {"train": 0.8, "val": 0.1, "test": 0.1}.

        Raises:
            ValueError: If generation fails or if the configuration is invalid.
        """
        pass

    def get_full_name(self) -> str:
        """
        Get the full name of the generator.

        Returns:
            str: Full name of the generator.
        """
        return PathManager.get_full_generator_name(self.name, self.implementation)

    def _get_split_sizes(
            self,
            num_samples: int,
            split_ratios: Optional[Dict[str, float]] = None
    ) -> Dict[str, int]:
        """
        Calculate the number of samples for each split based on the split ratios.

        Args:
            num_samples: Total number of samples to generate.
            split_ratios: Dictionary mapping split names to their ratios.
                Default is {"train": 0.8, "val": 0.1, "test": 0.1}.

        Returns:
            Dict[str, int]: Dictionary mapping split names to their sample counts.

        Raises:
            ValueError: If the split ratios don't sum to 1.0 or if any ratio is negative.
        """
        if split_ratios is None:
            split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

        # Validate split ratios
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        if any(ratio < 0 for ratio in split_ratios.values()):
            raise ValueError("Split ratios must be non-negative")

        # Calculate split sizes
        split_sizes = {}
        remaining = num_samples

        # Allocate integral number of samples to each split
        for split, ratio in split_ratios.items():
            if split == list(split_ratios.keys())[-1]:
                # Allocate all remaining samples to the last split
                split_sizes[split] = remaining
            else:
                # Allocate proportional number of samples to this split
                split_size = int(num_samples * ratio)
                split_sizes[split] = split_size
                remaining -= split_size

        return split_sizes