"""
Transformation strategies implementation.

This subpackage contains implementations of transformation strategies for polygon datasets,
including simplification and canonicalization algorithms.
"""

from .visvalingam import VisvalingamStrategy
from .douglas_peucker import DouglasPeuckerStrategy
from .canonicalize import CanonicalizeStrategy

__all__ = [
    'VisvalingamStrategy',
    'DouglasPeuckerStrategy',
    'CanonicalizeStrategy',
]