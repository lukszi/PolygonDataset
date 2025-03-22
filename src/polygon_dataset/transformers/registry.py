# polygon_dataset/transformers/registry.py
"""
Registry for polygon transformers.

This module provides a registry for polygon transformers, allowing for dynamic
lookup and instantiation of transformer classes by name.
"""

from polygon_dataset.transformers.base import Transformer
from polygon_dataset.utils import Registry

# Create a registry for transformer classes
transformer_registry = Registry[Transformer]("Transformer")

# Alias functions for backward compatibility
register_transformer = transformer_registry.register
get_transformer = transformer_registry.get
list_transformers = transformer_registry.list