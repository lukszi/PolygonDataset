# polygon_dataset/utils/extract_utils.py
"""
Utilities for extracting polygon data from raw files.

This module provides functionality for extracting polygon vertex data from
raw .line files into numpy arrays.
"""

import logging
from pathlib import Path
from typing import List, Any, Union, Callable

import numpy as np

from polygon_dataset.core import PathManager
from polygon_dataset.utils.read_line_file import read_line_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_dataset(path_manager: PathManager, config: Any) -> None:
    """
    Extract polygon data from raw files into numpy arrays.

    This function orchestrates the extraction process, delegating to specialized
    functions for the actual extraction work.

    Args:
        path_manager: Path manager for the dataset.
        config: Configuration object.

    Raises:
        ValueError: If extraction fails.
    """
    logger.info("Starting polygon extraction")

    for generator_config in config.generators:
        generator_name = generator_config.name

        # Skip native generators as they don't need extraction
        if generator_config.implementation != "binary":
            logger.info(f"Skipping extraction for native generator '{generator_name}'")
            continue

        logger.info(f"Processing generator '{generator_name}'")

        # Process each split
        split_ratios = config.dataset.split
        splits = ["train", "val", "test"]

        for split in splits:
            extract_split(
                path_manager=path_manager,
                generator_name=generator_name,
                algorithm=generator_config.params.get("algorithm", "default"),
                split=split,
                expected_vertices=config.dataset.vertex_count
            )

    logger.info("Polygon extraction completed")


def extract_split(
        path_manager: PathManager,
        generator_name: str,
        algorithm: str,
        split: str,
        expected_vertices: int
) -> None:
    """
    Extract polygon data for a specific split and generator.

    This function handles the extraction process for a specific split and generator,
    reading raw .line files and converting them to numpy arrays.

    Args:
        path_manager: Path manager for the dataset.
        generator_name: Name of the generator.
        algorithm: Algorithm used for generation.
        split: Dataset split (train/val/test).
        expected_vertices: Expected number of vertices in each polygon.

    Raises:
        ValueError: If extraction fails.
    """
    # Get raw files for this generator and split
    raw_dir = path_manager.get_raw_split_dir(split, generator_name)
    if not raw_dir.exists():
        logger.warning(f"Raw directory not found: {raw_dir}")
        return

    raw_files = list(raw_dir.glob("*.line"))
    if not raw_files:
        logger.warning(f"No .line files found in {raw_dir}")
        return

    logger.info(f"Processing {len(raw_files)} files for '{split}' split")

    # Get output directory (will be created if necessary)
    output_file = path_manager.get_processed_path(
        generator_name, algorithm, split
    )

    # Process raw files into polygons array
    polygons = process_raw_files(
        raw_files=raw_files,
        expected_vertices=expected_vertices,
        read_file_func=read_line_file  # Dependency injection for testability
    )

    if not polygons:
        logger.warning(f"No valid polygons found for '{split}' split")
        return

    # Save processed polygons
    save_polygons(polygons, output_file)
    logger.info(f"Saved {len(polygons)} polygons to {output_file}")


def process_raw_files(
        raw_files: List[Path],
        expected_vertices: int,
        read_file_func: Callable[[Union[str, Path]], np.ndarray] = read_line_file
) -> List[np.ndarray]:
    """
    Process raw .line files into polygon arrays.

    This function reads and validates multiple .line files, converting them
    to numpy arrays for further processing.

    Args:
        raw_files: List of paths to raw .line files.
        expected_vertices: Expected number of vertices in each polygon.
        read_file_func: Function to read a .line file (injectable for testing).

    Returns:
        List[np.ndarray]: List of polygon arrays.
    """
    polygons: List[np.ndarray] = []

    for file_path in raw_files:
        try:
            # Read polygon vertices
            vertices = read_file_func(file_path)

            # Validate vertex count
            if len(vertices) != expected_vertices:
                logger.warning(
                    f"Skipping {file_path.name}: expected {expected_vertices} vertices, "
                    f"got {len(vertices)}"
                )
                continue

            polygons.append(vertices)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return polygons


def save_polygons(polygons: List[np.ndarray], output_file: Path) -> None:
    """
    Save processed polygons to a numpy file.

    This function converts a list of polygon arrays to a single numpy array
    and saves it to disk.

    Args:
        polygons: List of polygon arrays.
        output_file: Path to the output file.

    Raises:
        IOError: If saving fails.
    """
    try:
        # Convert to numpy array and save
        logger.info(f"Converting {len(polygons)} polygons to numpy array")
        polygons_array = np.array(polygons)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_file, polygons_array)
    except Exception as e:
        raise IOError(f"Error saving polygons to {output_file}: {e}")