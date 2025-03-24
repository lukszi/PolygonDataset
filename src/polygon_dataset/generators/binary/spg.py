# polygon_dataset/generators/binary/spg.py
"""
Simple Polygon Generator implementation.

This module provides an implementation of the Simple Polygon Generator (SPG)
that uses an external executable to generate polygons.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from polygon_dataset.core import PathManager
from polygon_dataset.generators.binary.base import BinaryGenerator, SPGTask
from polygon_dataset.generators.registry import register_generator


@register_generator("spg_binary")
class SPGBinaryGenerator(BinaryGenerator):
    """
    Simple Polygon Generator using external binary executables.

    This generator uses the 'spg' executable to generate simple polygons
    with various algorithms.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SPG binary generator.

        Args:
            config: Configuration parameters for the generator.
                Must include 'bin_dir' pointing to the directory containing
                the 'spg' executable and the 'randomize_pnts.py' script.

        Raises:
            ValueError: If the 'bin_dir' parameter is missing.
        """
        super().__init__(config)
        self.algorithm: str = self.params.get("algorithm", "2opt")
        self.randomize_script = self._find_randomize_script()

    def _get_binary_name(self) -> str:
        """
        Get the name of the binary executable.

        Returns:
            str: Name of the binary executable.
        """
        return "spg"

    def _find_randomize_script(self) -> Path:
        """
        Locate the randomize_pnts.py script needed for SPG.

        Returns:
            Path: Path to the randomize_pnts.py script.

        Raises:
            FileNotFoundError: If the script cannot be found.
        """
        # Possible locations for the script
        possible_paths = [
            self.bin_dir.parent / "src/genpoly-spg/testdata/randomize_pnts.py",
            self.bin_dir.parent / "testdata/randomize_pnts.py",
            self.bin_dir / "randomize_pnts.py"
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                return path.resolve()

        raise FileNotFoundError(
            "Could not find randomize_pnts.py script. Please ensure it exists in one of these locations: "
            + ", ".join(str(p) for p in possible_paths)
        )

    def _execute_task(self, task: SPGTask) -> Optional[Path]:
        """
        Execute a single SPG generation task.

        Args:
            task: SPG task to execute.

        Returns:
            Optional[Path]: Path to the generated file if successful, None otherwise.
        """
        points_file = task.output_dir / f"points_{task.file_idx}.tmp"

        try:
            # Step 1: Generate random points using the Python script
            if not self._run_subprocess([
                "python3",
                str(task.randomize_script),
                "-o", str(points_file),
                "-s", str(task.vertices)
            ], cwd=task.output_dir):
                print(f"Error generating random points for SPG sample {task.file_idx}")
                return None

            # Step 2: Generate polygon from the points
            if not self._run_subprocess([
                str(self.bin_dir / "spg"),
                "-i", str(points_file),
                "-a", task.algorithm,
                "-b", "pnt",
                "-c", "line",
                "-o", str(task.output_file)
            ], cwd=task.output_dir):
                if points_file.exists():
                    points_file.unlink()
                print(f"Error generating SPG sample {task.file_idx}")
                return None

            # Clean up temporary points file
            if points_file.exists():
                points_file.unlink()

            return task.output_file

        except Exception as e:
            # Clean up temporary file if it exists
            if points_file.exists():
                points_file.unlink()

            print(f"Error generating SPG sample {task.file_idx}: {e}")
            return None

    def _create_tasks(
            self,
            path_manager: PathManager,
            vertex_count: int,
            split_sizes: Dict[str, int]
    ) -> List[SPGTask]:
        """
        Create tasks for parallel SPG generation.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            split_sizes: Dictionary mapping split names to their sizes.

        Returns:
            List[SPGTask]: List of tasks for worker processes.
        """
        algorithm = self.params.get("algorithm", "2opt")

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

                task = SPGTask(
                    output_dir=split_dir,
                    file_idx=i,
                    output_file=output_file,
                    vertices=vertex_count,
                    algorithm=algorithm,
                    randomize_script=self.randomize_script
                )

                tasks.append(task)

        return tasks