# polygon_dataset/utils/extract_utils.py
"""
Utilities for extracting polygon data from raw files.

This module provides functionality for extracting polygon vertex data from
raw .line files into numpy arrays.
"""

import logging
import numpy as np
from typing import Any

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
            # Get raw files for this generator and split
            raw_dir = path_manager.get_raw_split_dir(split, generator_name)
            if not raw_dir.exists():
                logger.warning(f"Raw directory not found: {raw_dir}")
                continue

            raw_files = list(raw_dir.glob("*.line"))
            if not raw_files:
                logger.warning(f"No .line files found in {raw_dir}")
                continue

            logger.info(f"Processing {len(raw_files)} files for '{split}' split")

            # Extract algorithm from the first file
            # Algorithms are specified as parameters in the generator config
            algorithm = generator_config.params.get("algorithm", "default")

            # Get output directory (will be created if necessary)
            output_dir = path_manager.get_extracted_dir()

            # Create output file
            output_file = path_manager.get_processed_path(
                generator_name, algorithm, split
            )

            # Read and process all polygon files
            polygons = []
            expected_vertices = config.dataset.vertex_count

            for file_path in raw_files:
                try:
                    # Read polygon vertices
                    vertices = read_line_file(file_path)

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

            if not polygons:
                logger.warning(f"No valid polygons found for '{split}' split")
                continue

            # Convert to numpy array and save
            logger.info(f"Converting {len(polygons)} polygons to numpy array")
            polygons_array = np.array(polygons)
            np.save(output_file, polygons_array)
            logger.info(f"Saved {len(polygons)} polygons to {output_file}")

    logger.info("Polygon extraction completed")