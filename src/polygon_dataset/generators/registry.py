# polygon_dataset/generators/registry.py
"""
Registry for polygon generators.

This module provides a registry for polygon generators, allowing for dynamic
lookup and instantiation of generator classes by name.
"""

from polygon_dataset.generators.base import Generator
from polygon_dataset.utils import Registry

# Create a registry for generator classes
generator_registry = Registry[Generator]("Generator")

# Alias functions for backward compatibility
register_generator = generator_registry.register
get_generator = generator_registry.get
list_generators = generator_registry.list