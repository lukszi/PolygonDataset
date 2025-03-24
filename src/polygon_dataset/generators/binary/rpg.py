# polygon_dataset/generators/binary/rpg.py
"""
Random Polygon Generator implementation.

This module provides an implementation of the Random Polygon Generator (RPG)
that uses an external executable to generate polygons.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from polygon_dataset.core import PathManager
from polygon_dataset.generators.binary.base import BinaryGenerator, RPGTask
from polygon_dataset.generators.registry import register_generator


@register_generator("rpg_binary")
class RPGBinaryGenerator(BinaryGenerator):
    """
    Random Polygon Generator using external binary executables.

    This generator uses the 'rpg' executable to generate random polygons
    with various algorithms (2opt, growth, space, etc.).
    """

    def _get_binary_name(self) -> str:
        """
        Get the name of the binary executable.

        Returns:
            str: Name of the binary executable.
        """
        return "rpg"

    def _execute_task(self, task: RPGTask) -> Optional[Path]:
        """
        Execute a single RPG generation task.

        Args:
            task: RPG task to execute.

        Returns:
            Optional[Path]: Path to the generated file if successful, None otherwise.
        """
        # Determine clustering mode
        cluster_flag = "--cluster" if task.cluster else "--random"

        # Build command
        cmd = [
            str(self.bin_dir / self._get_binary_name()),
            cluster_flag,
            str(task.vertices),
            "--algo", task.algorithm,
            "--format", "line",
            "--output", str(task.output_file),
            "--seed", str(random.randint(1, 100000000))
        ]

        # Add optional parameters if specified
        if task.holes > 0:
            cmd.extend(["--holes", str(task.holes)])

        if task.smooth > 0:
            cmd.extend(["--smooth", str(task.smooth)])

        # Run command
        if self._run_subprocess(cmd):
            # RPG adds .line extension
            return Path(f"{task.output_file}.line")
        else:
            print(f"Error generating RPG sample {task.file_idx}")
            return None

    def _create_tasks(
            self,
            path_manager: PathManager,
            vertex_count: int,
            split_sizes: Dict[str, int]
    ) -> List[RPGTask]:
        """
        Create tasks for parallel RPG generation.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            split_sizes: Dictionary mapping split names to their sizes.

        Returns:
            List[RPGTask]: List of tasks for worker processes.
        """
        algorithm = self.params.get("algorithm", "2opt")
        cluster = self.params.get("cluster", True)
        holes = self.params.get("holes", 0)
        smooth = self.params.get("smooth", 0)

        tasks = []

        # Create tasks for each split
        for split, count in split_sizes.items():
            # Ensure output directory exists
            split_dir = path_manager.get_raw_split_dir(split, self.get_full_name())
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create tasks
            for i in range(count):
                # Define output file path (without extension, as RPG adds .line)
                output_file = split_dir / f"polygon_{i:06d}"

                task = RPGTask(
                    output_dir=split_dir,
                    file_idx=i,
                    output_file=output_file,
                    vertices=vertex_count,
                    algorithm=algorithm,
                    cluster=cluster,
                    holes=holes,
                    smooth=smooth
                )

                tasks.append(task)

        return tasks