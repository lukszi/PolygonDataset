# polygon_dataset/config/config_schema.py
"""
Configuration schemas for polygon datasets.

This module defines the structured configuration schemas used with Hydra
for validating and managing dataset generation, transformation, and other
operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DatasetSplit(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class DatasetState(Enum):
    """States of dataset processing."""

    CONFIGURED = "configured"
    GENERATED = "generated"
    EXTRACTED = "extracted"
    TRANSFORMED = "transformed"
    CANONICALIZED = "canonicalized"


@dataclass
class SplitConfig:
    """
    Configuration for dataset splits.

    This defines the ratios of data allocated to training, validation, and testing.
    """

    train_ratio: float = 0.8
    """Ratio of data for the training set."""

    val_ratio: float = 0.1
    """Ratio of data for the validation set."""

    test_ratio: float = 0.1
    """Ratio of data for the test set."""

    def __post_init__(self) -> None:
        """
        Validate that split ratios sum to 1.0.

        Raises:
            ValueError: If split ratios don't sum to 1.0 or any ratio is negative.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio

        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        if self.train_ratio < 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")


@dataclass
class DatasetConfig:
    """
    Configuration for a polygon dataset.

    This defines the basic properties of the dataset, such as name, size,
    and vertex count.
    """

    name: str = MISSING
    """Name of the dataset."""

    size: int = MISSING
    """Total number of polygons in the dataset."""

    vertex_count: int = MISSING
    """Number of vertices in each polygon (including closing vertex)."""

    dimensionality: int = 2
    """Dimensionality of the vertices (typically 2 for x,y coordinates)."""

    split: SplitConfig = field(default_factory=SplitConfig)
    """Configuration for dataset splits."""

    state: DatasetState = DatasetState.CONFIGURED
    """Current state of the dataset."""


@dataclass
class GeneratorParams:
    """
    Base class for generator-specific parameters.

    This is an abstract base class for all generator parameter classes.
    """
    pass


@dataclass
class RPGParams(GeneratorParams):
    """
    Parameters for the Random Polygon Generator (RPG).

    These parameters control the behavior of the RPG generator.
    """

    algorithm: str = "2opt"
    """Algorithm to use for polygon generation (2opt, 2opt_ii, growth, space)."""

    holes: int = 0
    """Number of holes to generate in each polygon (0 for simple polygons)."""

    smooth: int = 0
    """Level of smoothing to apply (0 for no smoothing)."""

    cluster: bool = True
    """Whether to use clustering mode (true) or random mode (false)."""


@dataclass
class FPGParams(GeneratorParams):
    """
    Parameters for the Fast Polygon Generator (FPG).

    These parameters control the behavior of the FPG generator.
    """

    initial_vertices: int = 20
    """Number of initial vertices to start with."""

    holes: int = 0
    """Number of holes to generate in each polygon (0 for simple polygons)."""

    kinetic: bool = False
    """Whether to use kinetic mode for generation."""


@dataclass
class SPGParams(GeneratorParams):
    """
    Parameters for the Simple Polygon Generator (SPG).

    These parameters control the behavior of the SPG generator.
    """

    algorithm: str = "2opt"
    """Algorithm to use for polygon generation."""


@dataclass
class SRPGParams(GeneratorParams):
    """
    Parameters for the Super Random Polygon Generator (SRPG).

    These parameters control the behavior of the SRPG generator.
    """

    grid_x: int = 15
    """Number of grid cells in the x direction."""

    grid_y: int = 15
    """Number of grid cells in the y direction."""

    percent: float = 0.1
    """Percentage of grid cells to include in the polygon."""

    smooth: int = 3
    """Level of smoothing to apply."""

    holes: bool = False
    """Whether to generate holes in the polygons."""


@dataclass
class GeneratorConfig:
    """
    Configuration for a polygon generator.

    This defines the generator type, implementation, and parameters.
    """

    name: str = MISSING
    """Name of the generator (rpg, fpg, spg, srpg)."""

    implementation: str = "binary"
    """Implementation to use (binary or native)."""

    bin_dir: Optional[str] = None
    """Directory containing the generator binaries (required for binary implementation)."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Parameters specific to the generator, such as algorithm, holes, etc."""


@dataclass
class TransformConfig:
    """
    Configuration for polygon transformation.

    This defines the parameters for simplifying and transforming polygons.
    """

    algorithm: str = "visvalingam"
    """Transformation algorithm to use (visvalingam or douglas_peucker)."""

    min_vertices: int = 10
    """Minimum number of vertices for the lowest resolution."""

    batch_size: int = 400000
    """Number of polygons to process in each batch."""

    resolution_steps: Optional[List[int]] = None
    """Specific resolution steps to generate (calculated automatically if None)."""


@dataclass
class Config:
    """
    Root configuration for the polygon datasets package.

    This is the top-level configuration that includes all other config components.
    """

    dataset: DatasetConfig = MISSING
    """Configuration for the dataset."""

    generators: List[GeneratorConfig] = field(default_factory=list)
    """List of generator configurations."""

    transform: TransformConfig = field(default_factory=TransformConfig)
    """Configuration for polygon transformation."""

    output_dir: str = "./datasets"
    """Directory where datasets will be stored."""


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)

# Register generator parameter types
cs.store(group="generator", name="rpg_params", node=RPGParams)
cs.store(group="generator", name="fpg_params", node=FPGParams)
cs.store(group="generator", name="spg_params", node=SPGParams)
cs.store(group="generator", name="srpg_params", node=SRPGParams)