# polygon_dataset/utils/calculate_resolution_steps.py
"""
Utility for calculating resolution steps for polygon simplification.

This module provides functionality for calculating intermediate resolution
steps when simplifying polygons from their original vertex count down to a
minimum number of vertices.
"""

from typing import List


def calculate_resolution_steps(vertex_count: int, min_vertices: int = 10) -> List[int]:
    """
    Calculate intermediate resolution steps from original vertex count down to minimum.

    This function generates a sequence of vertex counts that gradually decrease from
    the original count to the minimum, attempting to find steps that divide evenly
    by factors of 2, 3, or 4.

    Args:
        vertex_count: Original number of vertices in the polygon.
        min_vertices: Minimum number of vertices to maintain in the simplification.

    Returns:
        List[int]: List of vertex counts for each resolution level, in ascending order
                  (from minimum to original).

    Examples:
        >>> calculate_resolution_steps(100, 10)
        [10, 25, 50, 100]

        >>> calculate_resolution_steps(88, 10)
        [11, 22, 44, 88]
    """
    # Start with the original count and work downward
    resolutions = [vertex_count]
    current_count = vertex_count

    while current_count > min_vertices:
        # Try to divide by common factors (2, 3, 4)
        for factor in [2, 3, 4]:
            if current_count % factor == 0 and current_count // factor >= min_vertices:
                current_count = current_count // factor
                resolutions.append(current_count)
                break
        else:
            # If no factor works (i.e., we didn't break out of the loop),
            # we can't divide evenly further. Stop here.
            break

    # Reverse the list to get ascending order (min_vertices to vertex_count)
    return list(reversed(resolutions))