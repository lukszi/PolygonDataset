# polygon_dataset/__init__.py
"""
Polygon Datasets Package.

This package provides a comprehensive framework for generating, transforming,
and working with polygon datasets, with support for various generators,
algorithms, and transformations.

Main components:
- generators: Implementations of polygon generators
- transformers: Polygon transformation algorithms
- core: Core functionality for dataset management
- utils: Utility functions for file handling and calculations
- config: Configuration schemas for Hydra integration
"""

from polygon_dataset.core import PolygonDataset, PathManager
from polygon_dataset.config import Config, DatasetConfig, GeneratorConfig, TransformConfig

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "PolygonDataset",
    "PathManager",
    "Config",
    "DatasetConfig",
    "GeneratorConfig",
    "TransformConfig",
]