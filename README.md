# Polygon Datasets

A comprehensive framework for generating, transforming, and managing polygon datasets for computational geometry and machine learning applications.

## Overview

Polygon Datasets provides tools to create, process, and analyze polygon data with a focus on consistent data management, flexible configuration, and efficient processing of large datasets. The framework implements multiple polygon generation algorithms and transformation techniques, allowing researchers and developers to create standardized polygon datasets for various applications.

## Features

- **Multiple Polygon Generation Methods**:
  - Random Polygon Generator (RPG) with various algorithms (2opt, 2opt_ii, growth, space)
  - Fast Polygon Generator (FPG)
  - Simple Polygon Generator (SPG)
  - Super Random Polygon Generator (SRPG)

- **Polygon Transformation Techniques**:
  - Polygon simplification using Douglas-Peucker algorithm
  - Polygon simplification using Visvalingam-Whyatt algorithm
  - Polygon canonicalization (standardizing vertex ordering)

- **Consistent Dataset Management**:
  - Organized directory structure for raw, extracted, transformed, and canonicalized data
  - Dataset splitting (train/validation/test)
  - Consistent file naming conventions

- **Visualization Tools**:
  - Interactive polygon viewer for comparing different simplification levels
  - Support for generating visualizations of polygon datasets

- **Flexible Configuration**:
  - Hydra-based configuration system
  - Support for multiple resolutions (vertex counts)
  - Customizable generation parameters

## Installation

### Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

### Installing the Package

```bash
# Clone the repository
git clone https://github.com/lukszi/polygon-dataset.git
cd polygon-dataset

# Install in development mode
pip install -e .

# Or install with native dependencies
pip install -e ".[native]"
```

## Usage

### Generating a Dataset

```bash
# Generate a small dataset using default settings
generate-dataset dataset=small_dataset generators=rpg_binary

# Generate a larger dataset with specific generator and transform settings
generate-dataset dataset=large_dataset generators=rpg_native transform=visvalingam
```

### Dataset Extraction

If using binary generators, extract the raw polygon data:

```bash
extract-dataset dataset=polygon_dataset_v1
```

### Transforming a Dataset

Apply polygon simplification to create multi-resolution versions:

```bash
transform-dataset dataset=polygon_dataset_v1 transform=douglas_peucker
```

### Canonicalizing a Dataset

Standardize polygon representation by rotating to start with lexicographically smallest vertex:

```bash
canonicalize-dataset dataset=polygon_dataset_v1
```

### Visualizing Polygons

Use the interactive viewer to explore the dataset:

```bash
visualize-dataset dataset_name=polygon_dataset_v1 generator=rpg algorithm=2opt
```

## Architecture

The project follows a modular architecture organized into several packages:

- **core**: Central components for dataset management and pipeline orchestration
- **generators**: Implementations of polygon generation algorithms
- **transformers**: Polygon transformation algorithms
- **utils**: Utility functions for file handling and calculations
- **config**: Configuration schemas for Hydra integration
- **cli**: Command-line tools for interacting with the framework

## Configuration System

The framework uses Hydra for configuration management, allowing for flexible parameterization of all aspects of dataset generation and processing.

### Example Configuration

```yaml
# Dataset configuration
dataset:
  name: polygon_dataset_v1
  size: 10000
  vertex_count: 88
  dimensionality: 2
  split:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1

# Generator configuration
generators:
  - name: rpg
    implementation: native
    params:
      algorithm: 2opt
      holes: 0
      smooth: 0
      cluster: true

# Transformation configuration
transform:
  algorithm: visvalingam
  min_vertices: 10
  batch_size: 400000
  resolution_steps: [11, 22, 44, 88]
```

## Dataset Structure

The framework organizes datasets in a consistent directory structure:

```
dataset_name/
├── config.json              # Dataset configuration
├── raw/                     # Raw polygon files (.line format)
│   ├── train/
│   │   ├── generator_name/  # Raw files by generator
│   ├── val/
│   └── test/
├── extracted/               # Extracted numpy arrays
│   ├── train_*.npy
│   ├── val_*.npy
│   └── test_*.npy
├── transformed/             # Simplified polygons at different resolutions
│   ├── train_*_res*.npy
│   ├── val_*_res*.npy
│   └── test_*_res*.npy
└── canonicalized/           # Canonicalized versions
    ├── train_*.npy
    ├── train_*_res*.npy
    └── ...
```

## Using the Dataset API

```python
from polygon_dataset import PolygonDataset

# Load a dataset
dataset = PolygonDataset("/path/to/dataset")

# Get metadata
metadata = dataset.get_metadata()
print(f"Dataset: {metadata['name']}")
print(f"Available generators: {metadata['generators']}")
print(f"Resolution steps: {metadata['resolution_steps']}")

# Load polygons for training
train_polygons = dataset.get_polygons(
  split="train",
  generator="rpg",
  algorithm="2opt",
  resolution=22,  # Optional: specific resolution
  canonicalized=True  # Optional: get canonicalized version
)

print(f"Loaded {len(train_polygons)} polygons with shape {train_polygons.shape}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.