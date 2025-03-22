# polygon_dataset/generators/binary/__init__.py
"""
Binary-based polygon generators.

This subpackage provides implementations of polygon generators that use external
executables to generate polygons.
"""

from polygon_dataset.generators.binary.base import (
    BinaryGenerator,
    BinaryGeneratorTask,
    RPGTask,
    FPGTask,
    SPGTask,
    SRPGTask
)

from polygon_dataset.generators.binary.rpg import RPGBinaryGenerator
from polygon_dataset.generators.binary.fpg import FPGBinaryGenerator
from polygon_dataset.generators.binary.spg import SPGBinaryGenerator
from polygon_dataset.generators.binary.srpg import SRPGBinaryGenerator

__all__ = [
    "BinaryGenerator",
    "BinaryGeneratorTask",
    "RPGTask",
    "FPGTask",
    "SPGTask",
    "SRPGTask",
    "RPGBinaryGenerator",
    "FPGBinaryGenerator",
    "SPGBinaryGenerator",
    "SRPGBinaryGenerator",
]