# scripts/cli_utils.py
"""
CLI utilities for polygon datasets.

This module provides common utilities for command-line scripts
to reduce code duplication and ensure consistent behavior.
"""

import logging
import sys
import traceback
from typing import Any, List, Optional, Callable, TypeVar

from omegaconf import DictConfig, OmegaConf

from polygon_dataset.config import Config
from polygon_dataset.core import PathManager

# Type variable for generic config types
T = TypeVar('T')

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_logging() -> logging.Logger:
    """
    Set up and return a configured logger.

    Returns:
        logging.Logger: Configured logger.
    """
    return logger


def validate_config(cfg: DictConfig, required_sections: List[str] = None) -> None:
    """
    Validate configuration for required sections and parameters.

    Args:
        cfg: Hydra configuration object.
        required_sections: List of required configuration sections.
            Defaults to ["dataset"].

    Raises:
        ValueError: If required sections or parameters are missing.
    """
    if required_sections is None:
        required_sections = ["dataset"]

    # Check for required sections
    for section in required_sections:
        if not cfg.get(section):
            raise ValueError(f"'{section}' section is required in configuration")

    # Validate dataset section if required
    if "dataset" in required_sections:
        if not cfg.get("dataset", {}).get("name"):
            raise ValueError("Dataset name is required in configuration")


def convert_config(cfg: DictConfig) -> Config:
    """
    Convert Hydra DictConfig to structured Config object.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Config: Structured configuration object.
    """
    return OmegaConf.to_object(cfg)


def init_path_manager(cfg: DictConfig) -> PathManager:
    """
    Initialize PathManager from configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        PathManager: Initialized path manager.

    Raises:
        ValueError: If required configuration parameters are missing.
    """
    if not cfg.get("dataset", {}).get("name"):
        raise ValueError("Dataset name is required for path manager initialization")

    output_dir = cfg.get("output_dir", "./datasets")
    dataset_name = cfg.dataset.name

    return PathManager(
        base_path=output_dir,
        dataset_name=dataset_name,
        create_dirs=True
    )


def update_dataset_state(
        path_manager: PathManager,
        config: Any,
        state: str
) -> None:
    """
    Update the dataset state in the configuration file.

    Args:
        path_manager: Path manager for the dataset.
        config: Configuration object.
        state: New state to set.

    Raises:
        IOError: If writing the configuration file fails.
    """
    path_manager.update_dataset_state(config, state)


def handle_errors(e: Exception, cfg: DictConfig, exit_code: int = 1) -> None:
    """
    Handle exceptions in a consistent way across CLI scripts.

    Args:
        e: The exception to handle.
        cfg: Hydra configuration object (for debug mode checking).
        exit_code: Exit code to use when terminating. Defaults to 1.

    Note:
        This function will terminate the program with sys.exit().
    """
    logger.error(f"Error: {e}")

    # In debug mode, print full traceback
    if cfg.get("debug", False):
        logger.error(traceback.format_exc())

    sys.exit(exit_code)


def run_cli_command(
        cfg: DictConfig,
        command_func: Callable[[Any, PathManager], None],
        config_validator: Callable[[DictConfig], None] = None,
        required_sections: List[str] = None,
        update_state: Optional[str] = None
) -> None:
    """
    Run a CLI command with standard error handling and configuration management.

    This function encapsulates the common pattern used across CLI scripts:
    1. Log the configuration
    2. Validate the configuration
    3. Convert the configuration
    4. Initialize the path manager
    5. Run the command function
    6. Update the dataset state (optional)

    Args:
        cfg: Hydra configuration object.
        command_func: Function that implements the command.
            Should accept (config, path_manager) as arguments.
        config_validator: Optional custom configuration validator.
        required_sections: List of required configuration sections.
        update_state: If provided, update the dataset state to this value
            after successful command execution.

    Raises:
        SystemExit: If an error occurs during command execution.
    """
    # Log the configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    try:
        # Validate configuration
        if config_validator:
            config_validator(cfg)
        else:
            validate_config(cfg, required_sections)

        # Convert DictConfig to structured Config
        config = convert_config(cfg)

        # Initialize path manager
        path_manager = init_path_manager(cfg)

        # Run the command
        command_func(cfg, path_manager)

        # Update dataset state if requested
        if update_state:
            update_dataset_state(path_manager, config, update_state)

        logger.info(f"Command completed successfully for dataset: {cfg.dataset.name}")

    except Exception as e:
        handle_errors(e, cfg)