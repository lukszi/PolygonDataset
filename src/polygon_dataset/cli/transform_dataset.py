#!/usr/bin/env python3
# polygon_dataset/cli/transform_dataset.py
"""
Script for transforming polygon datasets.

This script applies polygon transformations such as simplification
to create multi-resolution versions of the polygon dataset.

The script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig

from ..core.path_manager import PathManager
from .cli_utils import run_cli_command
from ..core.pipeline import transform_dataset


def validate_transform_config(cfg: DictConfig) -> None:
    """
    Validate configuration specific to transformation.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid for transformation.
    """
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("Dataset name is required in configuration")

    if not cfg.get("transform", {}).get("algorithm"):
        raise ValueError("Transformation algorithm is required in configuration")


def run_transformation(config: any, path_manager: PathManager) -> None:
    """
    Run the transformation process.

    Args:
        config: Configuration object.
        path_manager: Path manager for the dataset.
    """
    transform_dataset(path_manager, config)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Transform polygon dataset to multiple resolutions.

    Args:
        cfg: Hydra configuration object.
    """
    run_cli_command(
        cfg=cfg,
        command_func=run_transformation,
        config_validator=validate_transform_config,
        required_sections=["dataset", "transform"],
        update_state="transformed"
    )


if __name__ == "__main__":
    main()