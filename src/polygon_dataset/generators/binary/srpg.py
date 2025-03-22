# polygon_dataset/generators/binary/srpg.py
"""
Super Random Polygon Generator implementation.

This module provides an implementation of the Super Random Polygon Generator (SRPG)
that uses an external executable to generate polygons.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from polygon_dataset.core import PathManager
from polygon_dataset.generators.binary.base import BinaryGenerator, SRPGTask
from polygon_dataset.generators.registry import register_generator


@register_generator("srpg_binary")
class SRPGBinaryGenerator(BinaryGenerator):
    """
    Super Random Polygon Generator using external binary executables.

    This generator uses the 'srpg' executable to generate random polygons
    with grid-based parameters.
    """

    def _get_binary_name(self) -> str:
        """
        Get the name of the binary executable.

        Returns:
            str: Name of the binary executable.
        """
        return "srpg"

    def _execute_task(self, task: SRPGTask) -> Optional[Path]:
        """
        Execute a single SRPG generation task.

        Args:
            task: SRPG task to execute.

        Returns:
            Optional[Path]: Path to the generated file if successful, None otherwise.
        """
        # Build command
        cmd = [
            str(self.bin_dir / "srpg"),
            "--Nx", str(task.grid_x),
            "--Ny", str(task.grid_y),
            "--percent", str(task.percent),
            "--output", str(task.output_file),
            "--perturb",
            "--smooth", str(task.smooth),
            "--seed", str(random.randint(1, 10000000))
        ]

        # Add holes flag if requested
        if task.holes:
            cmd.append("--holes")

        # Run command
        if self._run_subprocess(cmd):
            return task.output_file
        else:
            print(f"Error generating SRPG sample {task.file_idx}")
            return None

    def _create_tasks(
            self,
            path_manager: PathManager,
            vertex_count: int,
            split_sizes: Dict[str, int]
    ) -> List[SRPGTask]:
        """
        Create tasks for parallel SRPG generation.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            split_sizes: Dictionary mapping split names to their sizes.

        Returns:
            List[SRPGTask]: List of tasks for worker processes.
        """
        grid_x = self.params.get("grid_x", 15)
        grid_y = self.params.get("grid_y", 15)
        percent = self.params.get("percent", 0.1)
        smooth = self.params.get("smooth", 3)
        holes = self.params.get("holes", False)

        tasks = []

        # Create tasks for each split
        for split, count in split_sizes.items():
            # Ensure output directory exists
            split_dir = path_manager.get_raw_split_dir(split, self.name)
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create tasks
            for i in range(count):
                # Define output file path
                output_file = split_dir / f"polygon_{i:06d}.line"

                task = SRPGTask(
                    output_dir=split_dir,
                    file_idx=i,
                    output_file=output_file,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    percent=percent,
                    smooth=smooth,
                    holes=holes
                )

                tasks.append(task)

        return tasks