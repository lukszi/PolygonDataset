# polygon_dataset/utils/read_line_file.py
"""
Utility for reading polygon data from .line files.

This module provides functionality for reading polygon vertex data from
.line files, which are a common format for storing polygon geometries.
"""

from pathlib import Path
from typing import List, Union

import numpy as np


def read_line_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Read a .line file and return its vertices as a numpy array.

    .line files are a simple text format where the first line typically contains
    the number of vertices, and each subsequent line contains the x,y coordinates
    of a vertex.

    Args:
        file_path: Path to the .line file.

    Returns:
        np.ndarray: Array of shape (num_vertices, 2) containing x,y coordinates.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.

    Example:
        >>> vertices = read_line_file("polygon_000001.line")
        >>> print(vertices.shape)
        (100, 2)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Line file not found: {file_path}")

    with open(file_path, 'r') as f:
        # Skip the first line (number of vertices might be inaccurate)
        next(f)
        lines = f.readlines()

    # Parse each line as a pair of float coordinates
    vertices = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Split by whitespace and convert to floats
            coords = list(map(float, line.split()))

            # Ensure we have exactly 2 coordinates
            if len(coords) != 2:
                raise ValueError(f"Invalid line format in {file_path}: {line}")

            vertices.append(coords)
        except ValueError as e:
            raise ValueError(f"Error parsing coordinates in {file_path}: {e}")

    if not vertices:
        raise ValueError(f"No valid vertices found in {file_path}")

    return np.array(vertices)


def read_multiple_line_files(
        directory: Union[str, Path],
        max_files: int = 50
) -> List[np.ndarray]:
    """
    Read multiple .line files from a directory.

    Args:
        directory: Directory containing .line files.
        max_files: Maximum number of files to read (default: 50).

    Returns:
        List[np.ndarray]: List of arrays, each containing the vertices of a polygon.

    Raises:
        FileNotFoundError: If the directory doesn't exist.

    Example:
        >>> polygons = read_multiple_line_files("raw/train/rpg", max_files=10)
        >>> print(len(polygons))
        10
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    polylines: List[np.ndarray] = []

    # Get all .line files in the directory
    line_files = list(directory.glob('*.line'))

    # Sort files to ensure consistent order
    line_files.sort()

    # Read files up to max_files limit
    for i, file_path in enumerate(line_files):
        if i >= max_files:
            print(f"Reached maximum file limit ({max_files}). Read {i} polylines.")
            break

        try:
            polylines.append(read_line_file(file_path))
        except (ValueError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")

    if not polylines:
        print(f"No valid polylines found in {directory}")
    else:
        print(f"Successfully read {len(polylines)} polylines from {directory}")

    return polylines