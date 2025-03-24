"""
Pipeline orchestration for polygon dataset generation and transformation.

This module provides functions for running the complete dataset generation pipeline
or individual steps, based on the configuration.
"""

from typing import Any, Dict, Optional, List
import logging

from polygon_dataset.core.path_manager import PathManager
from polygon_dataset.transformers import get_transformer
from polygon_dataset.generators import get_generator
from polygon_dataset.utils import calculate_resolution_steps
from polygon_dataset.utils.extract_utils import extract_dataset as extract_dataset_impl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_generation_pipeline(config: Any, path_manager: Optional[PathManager] = None) -> None:
    """
    Run the complete polygon dataset generation pipeline.

    This function orchestrates the entire pipeline for generating, extracting,
    transforming, and canonicalizing polygon datasets based on the provided
    configuration.

    Args:
        config: Hydra configuration object.
        path_manager: Optional path manager. If None, one will be created.

    Raises:
        ValueError: If the configuration is invalid or if a requested step fails.
    """
    # Initialize path manager if not provided
    if not path_manager:
        path_manager = PathManager(
            base_path=config.output_dir,
            dataset_name=config.dataset.name,
            create_dirs=True
        )

    # Validate configuration
    if not config.generators:
        raise ValueError("No generators specified in configuration")

    logger.info(f"Starting dataset generation for '{config.dataset.name}'")

    # 1. Generate polygons with each generator
    generate_polygons(config, path_manager)
    path_manager.update_dataset_state(config, "generated")

    # 2. Extract polygons (if using binary generators)
    if any(g.implementation == "binary" for g in config.generators):
        logger.info("Extracting polygons from raw format...")
        extract_dataset(config, path_manager)
    path_manager.update_dataset_state(config, "extracted")

    # 3. Transform polygons to multiple resolutions
    if hasattr(config, "transform") and config.transform:
        logger.info("Transforming polygons to multiple resolutions...")
        transform_dataset(config, path_manager)
    path_manager.update_dataset_state(config, "transformed")

    # 4. Canonicalize polygons
    logger.info("Canonicalizing polygons...")
    canonicalize_dataset(config, path_manager)
    path_manager.update_dataset_state(config, "canonicalized")

    logger.info(f"Dataset generation complete: {config.dataset.name}")


def generate_polygons(config: Any, path_manager: PathManager) -> None:
    """
    Generate polygons using each configured generator.

    Args:
        config: Configuration object with dataset and generator settings.
        path_manager: Path manager for the dataset.

    Raises:
        ValueError: If generation fails or if the configuration is invalid.
    """
    for gen_config in config.generators:
        full_generator_name = path_manager.get_full_generator_name(gen_config.name, gen_config.implementation)
        logger.info(f"Generating polygons with {gen_config.name} ({gen_config.implementation})...")

        # Get and initialize the generator
        generator_cls = get_generator(full_generator_name)
        generator = generator_cls(gen_config)

        # Generate polygons
        generator.generate(
            path_manager=path_manager,
            vertex_count=config.dataset.vertex_count - 1,  # -1 to account for closing vertex
            num_samples=config.dataset.size,
            split_ratios={
                "train": config.dataset.split.train_ratio,
                "val": config.dataset.split.val_ratio,
                "test": config.dataset.split.test_ratio
            }
        )


def extract_dataset(config: Any, path_manager: PathManager) -> None:
    """
    Extract polygon data from raw .line files into numpy arrays.

    Args:
        config: Configuration object with dataset and generator settings.
        path_manager: Path manager for the dataset.

    Raises:
        ValueError: If extraction fails or if a required file is missing.
    """
    extract_dataset_impl(path_manager, config)


def transform_dataset(config: Any, path_manager: PathManager) -> None:
    """
    Transform polygon dataset to multiple resolutions.

    Args:
        config: Configuration object with dataset and transformation settings.
        path_manager: Path manager for the dataset.

    Raises:
        ValueError: If transformation fails or required files are missing.
    """
    # Calculate resolution steps if not provided
    if not config.transform.resolution_steps:
        vertex_count = config.dataset.vertex_count
        min_vertices = config.transform.min_vertices
        config.transform.resolution_steps = calculate_resolution_steps(vertex_count, min_vertices)
        logger.info(f"Calculated resolution steps: {config.transform.resolution_steps}")

    # Convert config to dictionary for transformer
    transform_config = _config_to_dict(config.transform)

    # Get the transformer with strategy
    transformer = get_transformer(config.transform.algorithm, transform_config)

    # Transform the dataset
    transformer.transform_dataset(
        path_manager=path_manager,
        resolutions=config.transform.resolution_steps
    )


def canonicalize_dataset(config: Any, path_manager: PathManager) -> None:
    """
    Canonicalize polygon dataset.

    Args:
        config: Configuration object with dataset settings.
        path_manager: Path manager for the dataset.

    Raises:
        ValueError: If canonicalization fails or required files are missing.
    """
    # Prepare configuration for canonicalization
    canon_config = {
        "name": "canonicalize",
        "chunk_size": getattr(config.transform, "batch_size", 100000)
    }

    # Get the transformer with canonicalization strategy
    transformer = get_transformer("canonicalize", canon_config)

    # Determine resolutions to process
    resolutions = [None]  # Start with original resolution
    if hasattr(config, "transform") and config.transform.resolution_steps:
        resolutions.extend(config.transform.resolution_steps)

    # Canonicalize the dataset
    transformer.transform_dataset(
        path_manager=path_manager,
        resolutions=resolutions
    )


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert OmegaConf configuration object to dictionary.

    Args:
        config: Configuration object.

    Returns:
        Dictionary representation of the configuration.
    """
    if hasattr(config, "items"):
        # If config has an 'items' method (like a DictConfig), convert it directly
        return {k: _config_to_dict(v) if hasattr(v, "items") else v for k, v in config.items()}
    else:
        # Otherwise, try to convert to dictionary if it has attributes
        try:
            return {k: _config_to_dict(v) if hasattr(v, "items") else v
                    for k, v in config.__dict__.items()
                    if not k.startswith("_")}
        except AttributeError:
            # If it's a simple value, return it directly
            return config