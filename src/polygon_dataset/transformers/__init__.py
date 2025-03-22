# polygon_dataset/transformers/__init__.py
"""
Polygon transformation implementations.

This sub-package contains implementations of various polygon transformers,
which process and modify polygons in different ways, such as simplification
and canonicalization.
"""

from polygon_dataset.transformers.base import Transformer
from polygon_dataset.transformers.registry import register_transformer, get_transformer, list_transformers
from polygon_dataset.transformers.canonicalize import CanonicalizeTransformer
from polygon_dataset.transformers.douglas_peucker import DouglasPeuckerTransformer
from polygon_dataset.transformers.visvalingam import VisvalingamTransformer

__all__ = [
    "Transformer",
    "register_transformer",
    "get_transformer",
    "list_transformers",
    "CanonicalizeTransformer",
    "DouglasPeuckerTransformer",
    "VisvalingamTransformer",
]