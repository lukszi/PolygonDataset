#!/usr/bin/env python3
# polygon_dataset/cli/visualize_dataset.py
"""
Script for visualizing polygon datasets.

This script provides functionality for visualizing polygons from a dataset,
with support for viewing at different resolutions and comparing multiple
polygon versions.

The script uses Hydra for configuration management.
"""

import logging
import random
import sys
import tkinter as tk
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from omegaconf import DictConfig

from ..core.dataset import PolygonDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PolygonViewer:
    """
    Interactive viewer for polygon datasets.

    This class provides a GUI for viewing polygons from a dataset,
    with controls for navigating between different polygons and
    comparing multiple resolutions.
    """

    def __init__(
            self,
            root: tk.Tk,
            polygons: List[np.ndarray],
            titles: List[str]
    ) -> None:
        """
        Initialize the polygon viewer.

        Args:
            root: Tkinter root window.
            polygons: List of polygon arrays to display.
            titles: Titles for each polygon array.
        """
        self.root = root
        self.polygons = polygons
        self.titles = titles
        self.current_index = 0
        self.num_polygons = len(polygons[0])

        # Create figure and canvas
        self.figure = plt.figure(figsize=(12, 8), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create navigation controls
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(fill=tk.X, padx=10, pady=5)

        # Previous button
        self.prev_button = tk.Button(
            self.nav_frame,
            text="Previous",
            command=self.previous_polygon
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        # Label showing current polygon
        self.nav_label = tk.Label(
            self.nav_frame,
            text=f"Polygon 1 of {self.num_polygons}"
        )
        self.nav_label.pack(side=tk.LEFT, padx=10)

        # Next button
        self.next_button = tk.Button(
            self.nav_frame,
            text="Next",
            command=self.next_polygon
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Random button
        self.random_button = tk.Button(
            self.nav_frame,
            text="Random",
            command=self.random_polygon
        )
        self.random_button.pack(side=tk.LEFT, padx=20)

        # Keyboard shortcuts
        root.bind('<Left>', lambda e: self.previous_polygon())
        root.bind('<Right>', lambda e: self.next_polygon())
        root.bind('r', lambda e: self.random_polygon())

        # Display initial polygon
        self.display_current_polygon()

    def display_current_polygon(self) -> None:
        """
        Display the current polygon.

        This method updates the display to show the current polygon at all resolutions.
        """
        # Clear the figure
        self.figure.clear()

        # Calculate layout dimensions
        num_plots = len(self.polygons)
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols

        # Plot each polygon version
        for i, (polygon, title) in enumerate(zip(self.polygons, self.titles)):
            # Create subplot
            ax = self.figure.add_subplot(rows, cols, i + 1)

            # Get current polygon
            poly = polygon[self.current_index]

            # Extract x and y coordinates
            x, y = poly[:, 0], poly[:, 1]

            # Plot the polygon
            ax.plot(x, y, 'b-', linewidth=2)
            ax.plot(x[0], y[0], 'ro', markersize=6)  # Highlight first vertex

            # Configure the plot
            ax.set_title(f"{title} ({len(poly)} vertices)")
            ax.set_aspect('equal')
            ax.grid(True)

        # Update the navigation label
        self.nav_label.config(
            text=f"Polygon {self.current_index + 1} of {self.num_polygons}"
        )

        # Refresh the canvas
        self.figure.tight_layout()
        self.canvas.draw()

    def next_polygon(self) -> None:
        """
        Move to the next polygon.
        """
        if self.current_index < self.num_polygons - 1:
            self.current_index += 1
            self.display_current_polygon()

    def previous_polygon(self) -> None:
        """
        Move to the previous polygon.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_polygon()

    def random_polygon(self) -> None:
        """
        Jump to a random polygon.
        """
        self.current_index = random.randint(0, self.num_polygons - 1)
        self.display_current_polygon()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Visualize polygons from a dataset.

    Args:
        cfg: Hydra configuration object.
    """
    try:
        # Get visualization parameters
        dataset_name = cfg.get("dataset_name")
        dataset_dir = cfg.get("dataset_dir", cfg.get("output_dir", "./datasets"))
        generator = cfg.get("generator")
        algorithm = cfg.get("algorithm")
        split = cfg.get("split", "train")
        resolutions = cfg.get("resolutions", [])
        canonicalized = cfg.get("canonicalized", False)

        # Validate parameters
        if not dataset_name:
            logger.error("dataset_name is required")
            sys.exit(1)

        # Initialize dataset
        dataset_path = Path(dataset_dir) / dataset_name
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)

        try:
            dataset = PolygonDataset(dataset_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            sys.exit(1)

        # If generator is not specified, use the first available one
        available_generators = dataset.get_generators()
        if not generator and available_generators:
            generator = available_generators[0]
            logger.info(f"Using generator: {generator}")

        # If algorithm is not specified, use the first available one
        available_algorithms = dataset.get_algorithms(generator)
        if not algorithm and available_algorithms:
            algorithm = next(iter(available_algorithms))
            logger.info(f"Using algorithm: {algorithm}")

        # If resolutions are not specified, use all available resolutions plus original
        available_resolutions = dataset.get_resolutions()
        if not resolutions and available_resolutions:
            resolutions = available_resolutions
            # Add None to include original resolution
            resolutions = [None] + resolutions
            logger.info(f"Using resolutions: {resolutions}")

        # Load polygons at all specified resolutions
        polygons = []
        titles = []

        for resolution in resolutions:
            try:
                polys = dataset.get_polygons(
                    split=split,
                    generator=generator,
                    algorithm=algorithm,
                    resolution=resolution,
                    canonicalized=canonicalized
                )

                polygons.append(polys)

                if resolution is None:
                    titles.append("Original")
                else:
                    titles.append(f"Resolution {resolution}")

            except Exception as e:
                logger.warning(f"Could not load resolution {resolution}: {e}")

        if not polygons:
            logger.error("No polygon data could be loaded")
            sys.exit(1)

        # Create and run the viewer
        root = tk.Tk()
        root.title(f"Polygon Viewer - {dataset_name}")
        root.geometry("1024x768")

        viewer = PolygonViewer(root, polygons, titles)

        logger.info("Starting visualization. Use left/right arrow keys to navigate, 'r' for random")
        root.mainloop()

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        # In debug mode, print full traceback
        if cfg.get("debug", False):
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()