# polygon_dataset/generators/binary.py
"""
Binary-based polygon generators.

This module serves as a facade to the binary subpackage, re-exporting
the binary generator classes for backward compatibility.
"""

from polygon_dataset.generators.binary import (
    BinaryGenerator,
    BinaryGeneratorTask,
    RPGTask,
    FPGTask,
    SPGTask,
    SRPGTask,
    RPGBinaryGenerator,
    FPGBinaryGenerator,
    SPGBinaryGenerator,
    SRPGBinaryGenerator
)

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