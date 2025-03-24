"""
Polygon transformation implementations.

This package provides functionality for transforming polygons, such as
simplification and canonicalization, using a strategy pattern.
"""

# Import the base classes
from .strategy import TransformerStrategy
from .transformer import Transformer

# Import the registry functions
from .registry import register_transformer, get_transformer, list_transformers

# Import all strategies to register them
from .strategies import VisvalingamStrategy, DouglasPeuckerStrategy, CanonicalizeStrategy

__all__ = [
    "Transformer",
    "TransformerStrategy",
    "register_transformer",
    "get_transformer",
    "list_transformers",
    "VisvalingamStrategy",
    "DouglasPeuckerStrategy",
    "CanonicalizeStrategy",
]