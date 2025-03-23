"""
Filename parsing utilities for polygon datasets.

This module provides functions for parsing and handling the standard filename
patterns used throughout the polygon datasets package.
"""

from typing import Dict, List, Optional, Tuple


def _extract_resolution(parts: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Extract resolution from filename parts if present.

    Args:
        parts: List of filename parts split by underscore.

    Returns:
        Tuple[List[str], Optional[str]]: Tuple containing:
            - List of remaining parts with the resolution part removed
            - Resolution value as string, or None if not present

    Raises:
        ValueError: If resolution part is invalid (not followed by a number).
    """
    # Search for a part starting with 'res'
    res_index = None
    for i, part in enumerate(parts):
        if part.startswith('res'):
            res_index = i
            break

    if res_index is None:
        return parts, None

    # Ensure res is followed by a number
    if len(parts[res_index]) <= 3 or not parts[res_index][3:].isdigit():
        raise ValueError(
            f"Invalid resolution format: {parts[res_index]}. "
            "Resolution must be followed by a number (e.g., res44)."
        )

    # Extract resolution value
    resolution = parts[res_index][3:]  # Remove 'res' prefix

    # Remove resolution part from the list
    remaining_parts = parts[:res_index] + parts[res_index+1:]

    return remaining_parts, resolution


def _validate_parts(parts: List[str], filename: str) -> None:
    """
    Validate the number of parts in a filename.

    Args:
        parts: List of filename parts split by underscore.
        filename: Original filename for error messages.

    Raises:
        ValueError: If there are not enough parts in the filename.
    """
    if len(parts) < 4:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            "Expected format: {split}_{generator_name}_{generator_implementation}_{algorithm}[_res{resolution}].npy"
        )


def parse_polygon_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse a polygon dataset filename into its components.

    Standard filename formats:
    - Without resolution: {split}_{generator_name}_{generator_implementation}_{algorithm}.npy
    - With resolution: {split}_{generator_name}_{generator_implementation}_{algorithm}_res{resolution}.npy

    Args:
        filename: The filename to parse (without directory path).

    Returns:
        Dict[str, Optional[str]]: Dictionary containing the parsed components:
            - 'split': Dataset split (train/val/test)
            - 'generator': Generator name including implementation (e.g., 'rpg_binary')
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

    # Extract resolution if present
    parts, resolution = _extract_resolution(parts)

    # Validate the remaining parts
    _validate_parts(parts, filename)

    # Extract split
    split = parts[0]

    # Extract generator (combine generator_name and generator_implementation)
    generator = f"{parts[1]}_{parts[2]}"

    # Extract algorithm (everything after the generator)
    algorithm = '_'.join(parts[3:])

    return {
        'split': split,
        'generator': generator,
        'algorithm': algorithm,
        'resolution': resolution
    }


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
        generator: Generator name with implementation (e.g., 'rpg_binary')
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
            - generator: Generator name with implementation (e.g., 'rpg_binary')
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