# polygon_dataset/generators/binary/base.py
"""
Base classes for binary-based polygon generators.

This module provides the abstract base class and task classes for binary generators
that use external executables for polygon generation.
"""

import multiprocessing
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from polygon_dataset.core import PathManager
from polygon_dataset.generators.base import Generator


# Task classes for different generators
@dataclass
class BinaryGeneratorTask:
    """
    Base class for binary generator tasks.

    Attributes:
        output_dir: Directory where the output file should be saved.
        file_idx: Index of the file within the split.
        output_file: Path to the output file.
    """
    output_dir: Path
    file_idx: int
    output_file: Path


@dataclass
class RPGTask(BinaryGeneratorTask):
    """
    Task for Random Polygon Generator.

    Attributes:
        vertices: Number of vertices for the polygon.
        algorithm: Algorithm to use for generation.
        cluster: Whether to use clustering mode.
        holes: Number of holes to generate.
        smooth: Level of smoothing to apply.
    """
    vertices: int
    algorithm: str
    cluster: bool
    holes: int
    smooth: int


@dataclass
class FPGTask(BinaryGeneratorTask):
    """
    Task for Fast Polygon Generator.

    Attributes:
        vertices: Number of vertices for the polygon.
        initial_vertices: Number of initial vertices to use.
        holes: Number of holes to generate.
        kinetic: Whether to use kinetic mode.
    """
    vertices: int
    initial_vertices: int
    holes: int
    kinetic: bool


@dataclass
class SPGTask(BinaryGeneratorTask):
    """
    Task for Simple Polygon Generator.

    Attributes:
        vertices: Number of vertices for the polygon.
        algorithm: Algorithm to use for generation.
        randomize_script: Path to the randomize_pnts.py script.
    """
    vertices: int
    algorithm: str
    randomize_script: Path


@dataclass
class SRPGTask(BinaryGeneratorTask):
    """
    Task for Super Random Polygon Generator.

    Attributes:
        grid_x: Number of grid cells in the x direction.
        grid_y: Number of grid cells in the y direction.
        percent: Percentage of grid cells to include in the polygon.
        smooth: Level of smoothing to apply.
        holes: Whether to generate holes in the polygons.
    """
    grid_x: int
    grid_y: int
    percent: float
    smooth: int
    holes: bool


class BinaryGenerator(Generator):
    """
    Abstract base class for generators that use external binary executables.

    This class provides common functionality for binary-based generators
    including binary verification and multiprocessing-based execution.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the binary generator.

        Args:
            config: Configuration parameters for the generator.
                Must include 'bin_dir' pointing to the directory containing
                binaries.

        Raises:
            ValueError: If the 'bin_dir' parameter is missing.
        """
        super().__init__(config)

        if "bin_dir" not in config:
            raise ValueError("'bin_dir' parameter is required for binary generators")

        self.bin_dir: Path = Path(config["bin_dir"])

        # Verify the binary exists
        self._verify_binary()

    def _verify_binary(self) -> None:
        """
        Verify that the binary executable exists and is executable.

        Raises:
            FileNotFoundError: If the binary is not found.
            PermissionError: If the binary is not executable.
        """
        binary_name = self._get_binary_name()
        binary_path = self.bin_dir / binary_name

        if not binary_path.exists():
            raise FileNotFoundError(f"Binary '{binary_name}' not found at {binary_path}")

        if not binary_path.is_file():
            raise FileNotFoundError(f"Path {binary_path} exists but is not a file")

        # Check if the file is executable (on Unix-like systems)
        if not binary_path.stat().st_mode & 0o100:
            raise PermissionError(f"Binary '{binary_path}' is not executable")

    @staticmethod
    def _run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """
        Run a subprocess command.

        Args:
            cmd: Command to run.
            cwd: Working directory for the command.

        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _get_binary_name(self) -> str:
        """
        Get the name of the binary executable.

        Returns:
            str: Name of the binary executable.
        """
        raise NotImplementedError("Subclasses must implement _get_binary_name")

    def _execute_task(self, task: BinaryGeneratorTask) -> Optional[Path]:
        """
        Execute a single generation task.

        Args:
            task: Task to execute.

        Returns:
            Optional[Path]: Path to the generated file if successful, None otherwise.
        """
        raise NotImplementedError("Subclasses must implement _execute_task")

    def _create_tasks(
            self,
            path_manager: PathManager,
            vertex_count: int,
            split_sizes: Dict[str, int]
    ) -> List[BinaryGeneratorTask]:
        """
        Create tasks for parallel polygon generation.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            split_sizes: Dictionary mapping split names to their sizes.

        Returns:
            List[BinaryGeneratorTask]: List of tasks for worker processes.
        """
        raise NotImplementedError("Subclasses must implement _create_tasks")

    def generate(
            self,
            path_manager: PathManager,
            vertex_count: int,
            num_samples: int,
            split_ratios: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Generate polygon samples using multiprocessing.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            num_samples: Total number of samples to generate.
            split_ratios: Dictionary mapping split names to their ratios.
                Default is {"train": 0.8, "val": 0.1, "test": 0.1}.

        Raises:
            ValueError: If generation fails or if the configuration is invalid.
        """
        # Calculate number of samples per split
        split_sizes = self._get_split_sizes(num_samples, split_ratios)

        # Create tasks for parallel processing
        tasks = self._create_tasks(path_manager, vertex_count, split_sizes)

        if not tasks:
            print(f"No tasks to execute for {self.name}")
            return

        # Use available cores for parallel generation, leave one free
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        num_processes = min(num_processes, len(tasks))

        print(f"Generating {len(tasks)} polygons using {num_processes} processes...")

        # Execute tasks in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._execute_task, tasks),
                total=len(tasks),
                desc=f"Generating polygons with {self.name}"
            ))

        # Count successful and failed generations
        successful = sum(1 for r in results if r is not None)
        failed = len(tasks) - successful

        print(f"Successfully generated {successful} polygons")
        if failed > 0:
            print(f"Failed to generate {failed} polygons")