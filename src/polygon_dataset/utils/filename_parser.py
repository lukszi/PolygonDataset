"""
Filename parsing utilities for polygon datasets.

This module provides functions for parsing and handling the standard filename
patterns used throughout the polygon datasets package.
"""

from typing import Dict, Optional, Tuple


def parse_polygon_filename(filename: str) -> Dict[str, str]:
    """
    Parse a polygon dataset filename into its components.

    Standard filename formats:
    - Without resolution: {split}_{generator}_{algorithm}.npy
    - With resolution: {split}_{generator}_{algorithm}_res{resolution}.npy

    Args:
        filename: The filename to parse (without directory path).

    Returns:
        Dict[str, str]: Dictionary containing the parsed components:
            - 'split': Dataset split (train/val/test)
            - 'generator': Generator name
            - 'algorithm': Algorithm name
            - 'resolution': Resolution value (if present, as string) or None

    Raises:
        ValueError: If the filename doesn't follow the expected pattern.
    """
    # Remove file extension if present
    if filename.endswith('.npy'):
        filename = filename[:-4]  # Remove '.npy'

    # Split the filename into parts
    parts = filename.split('_')

    # Filter out empty parts
    parts = [part for part in parts if part]

    # Basic validation
    if len(parts) < 3:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            "Expected format: {split}_{generator}_{algorithm}[_res{resolution}].npy"
        )

    # Extract components
    result = {
        'split': parts[0],
        'generator': parts[1],
    }

    # Check if resolution is present
    res_index = None
    for i, part in enumerate(parts):
        if part.startswith('res'):
            res_index = i
            break

    if res_index is not None:
        # Format with resolution
        if res_index < 3:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                "Missing algorithm component before resolution."
            )

        # Check that res is followed by a number
        if len(parts[res_index]) <= 3 or not parts[res_index][3:].isdigit():
            raise ValueError(
                f"Invalid filename format: {filename}. "
                "Resolution must be followed by a number (e.g., res44)."
            )

        result['algorithm'] = '_'.join(parts[2:res_index])
        result['resolution'] = parts[res_index][3:]  # Remove 'res' prefix
    else:
        # Format without resolution
        # Check that the algorithm part is not empty
        if not parts[2:]:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                "Algorithm component cannot be empty."
            )

        result['algorithm'] = '_'.join(parts[2:])
        result['resolution'] = None

    return result


def build_polygon_filename(
        split: str,
        generator: str,
        algorithm: str,
        resolution: Optional[int] = None,
) -> str:
    """
    Build a polygon dataset filename from components.

    Args:
        split: Dataset split (train/val/test)
        generator: Generator name
        algorithm: Algorithm name
        resolution: Resolution value (optional)

    Returns:
        str: The constructed filename with .npy extension
    """
    if resolution is not None:
        return f"{split}_{generator}_{algorithm}_res{resolution}.npy"
    else:
        return f"{split}_{generator}_{algorithm}.npy"


def parse_filename_components(
        filename: str
) -> Tuple[str, str, str, Optional[int]]:
    """
    Parse a polygon dataset filename into its components.

    This is a convenience function that returns the components as a tuple
    instead of a dictionary.

    Args:
        filename: The filename to parse (without directory path).

    Returns:
        Tuple[str, str, str, Optional[int]]: Tuple containing:
            - split: Dataset split (train/val/test)
            - generator: Generator name
            - algorithm: Algorithm name
            - resolution: Resolution value (if present, as int, otherwise None)

    Raises:
        ValueError: If the filename doesn't follow the expected pattern.
    """
    result = parse_polygon_filename(filename)

    resolution = None
    if result['resolution'] is not None:
        resolution = int(result['resolution'])

    return (
        result['split'],
        result['generator'],
        result['algorithm'],
        resolution
    )