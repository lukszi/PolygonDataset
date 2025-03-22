#!/usr/bin/env python3
# polygon_dataset/cli/extract_dataset.py
"""
Script for extracting polygon data from raw files.

This script extracts polygon data from raw .line files into numpy arrays,
which are easier to work with for subsequent processing steps.

The script uses Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig

from ..core.path_manager import PathManager
from .cli_utils import run_cli_command
from ..utils.extract_utils import extract_dataset


def validate_extract_config(cfg: DictConfig) -> None:
    """
    Validate configuration specific to extraction.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid for extraction.
    """
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("Dataset name is required in configuration")


def run_extraction(config: any, path_manager: PathManager) -> None:
    """
    Run the extraction process.

    Args:
        config: Configuration object.
        path_manager: Path manager for the dataset.
    """
    extract_dataset(path_manager, config)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Extract polygon data from raw files.

    Args:
        cfg: Hydra configuration object.
    """
    run_cli_command(
        cfg=cfg,
        command_func=run_extraction,
        config_validator=validate_extract_config,
        update_state="extracted"
    )


if __name__ == "__main__":
    main()