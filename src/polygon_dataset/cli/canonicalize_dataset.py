#!/usr/bin/env python3
# polygon_dataset/cli/canonicalize_dataset.py
"""
Script for canonicalizing polygon datasets.

This script canonicalizes polygon datasets by rotating each polygon to
start with its lexicographically smallest vertex, ensuring a standardized
representation.

The script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig

from ..core.path_manager import PathManager
from .cli_utils import run_cli_command
from ..core.pipeline import canonicalize_dataset


def validate_canonicalize_config(cfg: DictConfig) -> None:
    """
    Validate configuration specific to canonicalization.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid for canonicalization.
    """
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("Dataset name is required in configuration")


def run_canonicalization(config: any, path_manager: PathManager) -> None:
    """
    Run the canonicalization process.

    Args:
        config: Configuration object.
        path_manager: Path manager for the dataset.
    """
    canonicalize_dataset(path_manager, config)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Canonicalize polygon dataset.

    Args:
        cfg: Hydra configuration object.
    """
    run_cli_command(
        cfg=cfg,
        command_func=run_canonicalization,
        config_validator=validate_canonicalize_config,
        required_sections=["dataset"],
        update_state="canonicalized"
    )


if __name__ == "__main__":
    main()