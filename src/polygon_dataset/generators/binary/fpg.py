# polygon_dataset/generators/binary/fpg.py
"""
Fast Polygon Generator implementation.

This module provides an implementation of the Fast Polygon Generator (FPG)
that uses an external executable to generate polygons.
"""

from pathlib import Path
from typing import Dict, List, Optional

from polygon_dataset.core import PathManager
from polygon_dataset.generators.binary.base import BinaryGenerator, FPGTask
from polygon_dataset.generators.registry import register_generator


@register_generator("fpg_binary")
class FPGBinaryGenerator(BinaryGenerator):
    """
    Fast Polygon Generator using external binary executables.

    This generator uses the 'fpg' executable to generate random polygons
    with various parameters such as kinetic, holes, etc.
    """

    def _get_binary_name(self) -> str:
        """
        Get the name of the binary executable.

        Returns:
            str: Name of the binary executable.
        """
        return "fpg"

    def _execute_task(self, task: FPGTask) -> Optional[Path]:
        """
        Execute a single FPG generation task.

        Args:
            task: FPG task to execute.

        Returns:
            Optional[Path]: Path to the generated file if successful, None otherwise.
        """
        # Build command
        cmd = [
            str(self.bin_dir / self._get_binary_name()),
            "-i", str(task.initial_vertices),
            "-o", "line",
            "-m", str(task.vertices)
        ]

        # Add optional parameters
        if task.holes > 0:
            cmd.extend(["--nrofholes", str(task.holes)])

        if task.kinetic:
            cmd.append("--kinetic")

        # Output file comes last
        cmd.append(str(task.output_file))

        # Run command
        if self._run_subprocess(cmd):
            # For FPG, output_file is the complete path including .line
            return task.output_file
        else:
            print(f"Error generating FPG sample {task.file_idx}")
            return None

    def _create_tasks(
            self,
            path_manager: PathManager,
            vertex_count: int,
            split_sizes: Dict[str, int]
    ) -> List[FPGTask]:
        """
        Create tasks for parallel FPG generation.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            split_sizes: Dictionary mapping split names to their sizes.

        Returns:
            List[FPGTask]: List of tasks for worker processes.
        """
        initial_vertices = self.params.get("initial_vertices", 20)
        holes = self.params.get("holes", 0)
        kinetic = self.params.get("kinetic", False)

        tasks = []

        # Create tasks for each split
        for split, count in split_sizes.items():
            # Ensure output directory exists
            split_dir = path_manager.get_raw_split_dir(split, self.get_full_name())
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create tasks
            for i in range(count):
                # Define output file path
                output_file = split_dir / f"polygon_{i:06d}.line"

                task = FPGTask(
                    output_dir=split_dir,
                    file_idx=i,
                    output_file=output_file,
                    vertices=vertex_count,
                    initial_vertices=initial_vertices,
                    holes=holes,
                    kinetic=kinetic
                )

                tasks.append(task)

        return tasks