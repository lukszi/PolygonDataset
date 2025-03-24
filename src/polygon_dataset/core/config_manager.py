# polygon_dataset/core/config_manager.py
"""
Configuration management for polygon datasets.

This module provides the ConfigManager class for loading, saving, and updating
dataset configuration files.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """
    Manages dataset configuration files.

    This class handles loading, saving, and updating JSON configuration files
    for polygon datasets.
    """

    def __init__(self, config_path: Path) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path: Path = config_path

    def load_config(self) -> Dict[str, Any]:
        """
        Load the dataset configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the configuration file is invalid JSON.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found at {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save the dataset configuration to a file.

        Args:
            config: Configuration dictionary to save.

        Raises:
            IOError: If writing the configuration file fails.
        """
        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except (IOError, OSError) as e:
            raise IOError(f"Error writing configuration to {self.config_path}: {e}")

    def update_dataset_state(self, config: Any, state: str) -> None:
        """
        Update the dataset state in the configuration file.

        Args:
            config: Configuration dictionary or object with necessary attributes.
            state: New state to set.

        Raises:
            IOError: If updating the configuration file fails.
        """
        # Load existing config if it exists
        if self.config_path.exists():
            dataset_config = self.load_config()
        else:
            # Create a new config if it doesn't exist
            dataset_config = self._create_new_config(config, state)

        # Update the state
        dataset_config["dataset_info"]["dataset_state"] = state

        # Add generator configurations if not present
        self._update_generator_configs(dataset_config, config)

        # Add transformation configuration if not present
        self._update_transform_config(dataset_config, config)

        # Save the updated config
        self.save_config(dataset_config)

    @staticmethod
    def _create_new_config(config: Any, state: str) -> Dict[str, Any]:
        """
        Create a new configuration dictionary.

        Args:
            config: Configuration object with necessary attributes.
            state: Dataset state to set.

        Returns:
            Dict[str, Any]: New configuration dictionary.
        """
        return {
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

    @staticmethod
    def _update_generator_configs(dataset_config: Dict[str, Any], config: Any) -> None:
        """
        Update generator configurations in the dataset config.

        Args:
            dataset_config: Dataset configuration dictionary.
            config: Configuration object with generator information.
        """
        if not hasattr(config, "generators"):
            return

        for gen_config in config.generators:
            gen_name = f"{gen_config.name}_{gen_config.implementation}"
            if gen_name not in dataset_config["generator_configs"]:
                dataset_config["generator_configs"][gen_name] = {
                    "name": gen_config.name,
                    "implementation": gen_config.implementation,
                    "params": dict(gen_config.params) if hasattr(gen_config, "params") else {}
                }

    @staticmethod
    def _update_transform_config(dataset_config: Dict[str, Any], config: Any) -> None:
        """
        Update transformation configuration in the dataset config.

        Args:
            dataset_config: Dataset configuration dictionary.
            config: Configuration object with transform information.
        """
        if hasattr(config, "transform") and config.transform:
            dataset_config["transformation_config"] = {
                "algorithm": config.transform.algorithm,
                "min_vertices": config.transform.min_vertices,
                "batch_size": config.transform.batch_size,
                "resolution_steps": list(config.transform.resolution_steps)
                if config.transform.resolution_steps else []
            }