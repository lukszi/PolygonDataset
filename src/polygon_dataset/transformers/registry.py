from typing import Dict, Type, Any

from polygon_dataset.utils.registry import Registry
from .strategy import TransformerStrategy
from .transformer import Transformer

# Create a registry for transformer strategies
strategy_registry = Registry[TransformerStrategy]("TransformerStrategy")

# Alias functions for consistency with current API
register_transformer = strategy_registry.register


def get_transformer(name: str, config: Dict[str, Any]) -> Transformer:
    """
    Get a transformer by name.

    Args:
        name: Name of the transformer strategy.
        config: Configuration parameters.

    Returns:
        Transformer: Configured transformer with the requested strategy.

    Raises:
        ValueError: If strategy is not found.
    """
    # Get the strategy class
    strategy_cls = strategy_registry.get(name)

    # Create and configure the strategy
    strategy = strategy_cls(config)

    # Extract common transformer parameters
    chunk_size = config.get("batch_size", config.get("chunk_size", 100000))
    min_vertices = config.get("min_vertices", 10)

    # Create and return the transformer
    return Transformer(strategy, chunk_size, min_vertices)


def list_transformers() -> Dict[str, Type[TransformerStrategy]]:
    """
    Get a dictionary of all registered transformer strategies.

    Returns:
        Dict[str, Type[TransformerStrategy]]: Dictionary mapping names to classes.
    """
    return strategy_registry.list()