# polygon_dataset/core/__init__.py
"""
Core functionality for the polygon datasets package.

This sub-package contains the core components for managing polygon datasets,
including path management, dataset access, and pipeline orchestration.
"""

from polygon_dataset.core.path_manager import PathManager
from polygon_dataset.core.dataset import PolygonDataset
from polygon_dataset.core.pipeline import (
    run_generation_pipeline,
    generate_polygons,
    extract_dataset,
    transform_dataset,
    canonicalize_dataset,
)

__all__ = [
    "PathManager",
    "PolygonDataset",
    "run_generation_pipeline",
    "generate_polygons",
    "extract_dataset",
    "transform_dataset",
    "canonicalize_dataset",
]