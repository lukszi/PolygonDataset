#!/usr/bin/env python3
# polygon_dataset/cli/generate_dataset.py
"""
Main script for generating polygon datasets.

This script implements the complete dataset generation pipeline, including:
- Generating raw polygon data using specified generators
- Extracting polygons into numpy arrays
- Transforming polygons to multiple resolutions
- Canonicalizing polygons

The script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig

from ..core import PathManager, run_generation_pipeline
from .cli_utils import run_cli_command


def validate_generate_config(cfg: DictConfig) -> None:
    """
    Validate configuration specific to dataset generation.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid for generation.
    """
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("Dataset name is required in configuration")

    if not cfg.get("generators"):
        raise ValueError("At least one generator must be specified in configuration")


def run_generation(config: any, path_manager: PathManager) -> None:
    """
    Run the complete generation pipeline.

    Args:
        config: Configuration object.
        path_manager: Path manager for the dataset.
    """
    run_generation_pipeline(config, path_manager)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run the complete polygon dataset generation pipeline.

    Args:
        cfg: Hydra configuration object.
    """
    run_cli_command(
        cfg=cfg,
        command_func=run_generation,
        config_validator=validate_generate_config,
        required_sections=["dataset", "generators"]
    )


if __name__ == "__main__":
    main()