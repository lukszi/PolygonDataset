# polygon_dataset/generators/__init__.py
"""
Polygon generator implementations.

This sub-package contains implementations of various polygon generators,
including both binary (executable-based) and native (Python binding-based)
implementations.
"""

from polygon_dataset.generators.base import Generator
from polygon_dataset.generators.registry import register_generator, get_generator, list_generators

# Import all available generators to register them
from polygon_dataset.generators.binary import (
    RPGBinaryGenerator,
    FPGBinaryGenerator,
    SPGBinaryGenerator,
    SRPGBinaryGenerator,
)
from polygon_dataset.generators.native import RPGNativeGenerator

__all__ = [
    "Generator",
    "register_generator",
    "get_generator",
    "list_generators",
    "RPGBinaryGenerator",
    "FPGBinaryGenerator",
    "SPGBinaryGenerator",
    "SRPGBinaryGenerator",
    "RPGNativeGenerator",
]
