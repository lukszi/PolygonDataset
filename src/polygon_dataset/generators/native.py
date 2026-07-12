# polygon_dataset/generators/native.py
"""
Native Python-binding-based polygon generators.

This module provides implementations of polygon generators that use native
Python bindings to external C/C++ libraries for generating polygons.
"""

import gc
import logging
import multiprocessing
import random
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from polygon_dataset.utils.chunking import write_chunk_to_file, create_empty_memory_mapped_file, distribute_work

try:
    import genpoly_rpg
except ImportError:
    genpoly_rpg = None

try:
    import genpoly_fpg
except ImportError:
    genpoly_fpg = None

from polygon_dataset.core import PathManager
from polygon_dataset.generators.base import Generator
from polygon_dataset.generators.registry import register_generator

logger = logging.getLogger(__name__)

# Named tuple for RPG Native generator tasks
RPGNativeTask = namedtuple('RPGNativeTask', ['worker_id', 'count', 'vertices', 'algorithm', 'seed'])

# Named tuple for FPG Native generator tasks.
# Unlike RPG, the FPG binding exposes no seed argument: genpoly_fpg reseeds a
# fresh std::mt19937 from std::random_device for every polygon, so each worker
# (and each polygon) is independently random with no duplicate risk.
FPGNativeTask = namedtuple('FPGNativeTask', ['worker_id', 'count', 'vertices', 'initial_vertices', 'radius'])


@register_generator("rpg_native")
class RPGNativeGenerator(Generator):
    """
    Random Polygon Generator using native Python bindings.

    This generator uses the genpoly_rpg module to generate random polygons
    with various algorithms (2opt_ii, growth, space, etc.).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RPG native generator.

        Args:
            config: Configuration parameters for the generator.

        Raises:
            ImportError: If the genpoly_rpg module is not available.
        """
        if genpoly_rpg is None:
            raise ImportError(
                "genpoly_rpg module not found. Install the 'native' extra "
                "(polygon-dataset[native]) to use the native RPG generator."
            )
        super().__init__(config)

    def _generate_batch(self, task: RPGNativeTask) -> np.ndarray:
        """
        Generate a batch of polygons using the genpoly_rpg module.

        Args:
            task: Named tuple containing worker_id, count, vertices, algorithm, and seed.

        Returns:
            np.ndarray: Array of generated polygons with shape (count, vertices+1, 2).

        Raises:
            RuntimeError: If polygon generation fails.
        """
        try:
            # Generate 'count' polygons using the specified algorithm and seed
            polygons = genpoly_rpg.generate_polygons(
                vertices=task.vertices,
                num_polygons=task.count,
                algorithm=task.algorithm,
                seed=task.seed
            )
            return polygons
        except Exception as e:
            raise RuntimeError(f"Error in genpoly_rpg.generate_polygons: {e}")

    def generate(
            self,
            path_manager: PathManager,
            vertex_count: int,
            num_samples: int,
            split_ratios: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Generate polygon samples using native Python bindings.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            num_samples: Total number of samples to generate.
            split_ratios: Dictionary mapping split names to their ratios.
                Default is {"train": 0.8, "val": 0.1, "test": 0.1}.

        Raises:
            ValueError: If generation fails or if the configuration is invalid.
            RuntimeError: If the genpoly_rpg module fails.
        """
        # Calculate number of samples per split
        split_sizes = self._get_split_sizes(num_samples, split_ratios)

        # Get algorithm from parameters
        algorithm = self.params.get("algorithm", "2opt")

        # Use available cores for parallel generation, leave one free
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Determine chunk size for processing
        chunk_size = 1000  # Adjust based on memory constraints

        # Process each split
        for split, size in split_sizes.items():
            logger.debug(f"Generating {size} polygons for '{split}' split using algorithm '{algorithm}'...")

            # Create output memory-mapped array
            shape = (size, vertex_count + 1, 2)  # +1 for closing vertex
            output_file = self.create_output_file(algorithm, path_manager, shape, split)

            # Process in chunks to avoid memory issues
            for chunk_start in range(0, size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, size)
                chunk_size_actual = chunk_end - chunk_start
                logger.debug(f"  Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

                # Create tasks for parallel processing
                tasks = self._create_tasks(chunk_size_actual, num_processes, vertex_count, algorithm)

                # Execute tasks and write results to output file
                chunk_polygons = self._process_tasks(tasks, num_processes)
                write_chunk_to_file(output_file, chunk_polygons, chunk_start, shape, dtype="float64")
                logger.debug(f"    Wrote {chunk_size_actual} polygons to {output_file}")

                # Memory management
                del chunk_polygons  # Free memory
                gc.collect()  # Force garbage collection

        logger.info(f"Successfully generated {num_samples} polygons using native RPG with algorithm '{algorithm}'")


    def _create_tasks(
            self,
            chunk_size: int,
            num_processes: int,
            vertex_count: int,
            algorithm: str
    ) -> List[RPGNativeTask]:
        """
        Create tasks for parallel polygon generation.

        Args:
            chunk_size: Size of the current chunk to process.
            num_processes: Number of worker processes to use.
            vertex_count: Number of vertices for each polygon.
            algorithm: Algorithm to use for generation.

        Returns:
            List[Tuple]: List of task tuples for worker processes.
        """
        batches: List[Tuple[int, int]] = distribute_work(chunk_size, num_processes)

        tasks = []
        for i, batch in enumerate(batches):
            seed = random.randint(0, 2 ** 31 - 1)
            batch_size = batch[1] - batch[0]
            tasks.append(RPGNativeTask(i, batch_size, vertex_count, algorithm, seed))

        return tasks


    def _process_tasks(
            self,
            tasks: List[RPGNativeTask],
            num_processes: int
    ) -> np.ndarray:
        """
        Process tasks in parallel and combine results.

        Args:
            tasks: List of task tuples for worker processes.
            num_processes: Number of worker processes to use.

        Returns:
            np.ndarray: Combined array of polygons.

        Raises:
            RuntimeError: If polygon generation fails.
        """
        # Create a process pool and execute tasks
        with multiprocessing.Pool(processes=min(num_processes, len(tasks))) as pool:
            # Use map instead of imap to ensure ordered results
            partial_results = pool.map(self._generate_batch, tasks)

        # Combine partial results
        if not partial_results:
            raise RuntimeError("No polygons generated")

        return np.concatenate(partial_results, axis=0)


    def create_output_file(self, algorithm: str, path_manager: PathManager, shape: Tuple[int, ...], split: str) -> Path:
        """
        Create a memory-mapped output file for storing generated polygons.
        """
        output_file = path_manager.get_processed_path(self.get_full_name(), algorithm, split)
        create_empty_memory_mapped_file(output_file, shape, dtype="float64")
        return output_file


@register_generator("fpg_native")
class FPGNativeGenerator(Generator):
    """
    Fast Polygon Generator using native Python bindings.

    This generator uses the genpoly_fpg module to grow simple polygons from an
    initial regular polygon, mirroring the structure of RPGNativeGenerator. The
    FPG binding exposes ``generate_multiple_polygons(num_polygons, num_vertices,
    initial_vertices, radius)`` and returns a float64 array of shape
    ``(num_polygons, num_vertices + 1, 2)`` with each polygon closed (first
    vertex repeated at the end).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the FPG native generator.

        Args:
            config: Configuration parameters for the generator.

        Raises:
            ImportError: If the genpoly_fpg module is not available.
        """
        if genpoly_fpg is None:
            raise ImportError(
                "genpoly_fpg module not found. Install the 'native' extra "
                "(polygon-dataset[native]) to use the native FPG generator."
            )
        super().__init__(config)

    def _generate_batch(self, task: FPGNativeTask) -> np.ndarray:
        """
        Generate a batch of polygons using the genpoly_fpg module.

        Args:
            task: Named tuple containing worker_id, count, vertices,
                initial_vertices, and radius.

        Returns:
            np.ndarray: Array of generated polygons with shape (count, vertices+1, 2).

        Raises:
            RuntimeError: If polygon generation fails.
        """
        try:
            # Generate 'count' polygons. FPG reseeds from std::random_device on
            # every polygon, so no explicit seed is passed or needed.
            polygons = genpoly_fpg.generate_multiple_polygons(
                num_polygons=task.count,
                num_vertices=task.vertices,
                initial_vertices=task.initial_vertices,
                radius=task.radius,
            )
            return polygons
        except Exception as e:
            raise RuntimeError(f"Error in genpoly_fpg.generate_multiple_polygons: {e}")

    def generate(
            self,
            path_manager: PathManager,
            vertex_count: int,
            num_samples: int,
            split_ratios: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Generate polygon samples using native Python bindings.

        Args:
            path_manager: Path manager for the dataset.
            vertex_count: Number of vertices for each polygon (excluding closing vertex).
            num_samples: Total number of samples to generate.
            split_ratios: Dictionary mapping split names to their ratios.
                Default is {"train": 0.8, "val": 0.1, "test": 0.1}.

        Raises:
            ValueError: If generation fails or if the configuration is invalid.
            RuntimeError: If the genpoly_fpg module fails.
        """
        # Calculate number of samples per split
        split_sizes = self._get_split_sizes(num_samples, split_ratios)

        # FPG parameters. The algorithm token mirrors the binary FPG route,
        # which carries no algorithm variant and falls back to "default" during
        # extraction (params.get("algorithm", "default")); keeping it identical
        # here means native and binary FPG outputs share the same naming.
        algorithm = self.params.get("algorithm", "default")
        initial_vertices = self.params.get("initial_vertices", 20)
        radius = self.params.get("radius", 1.0)

        # Use available cores for parallel generation, leave one free
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Determine chunk size for processing
        chunk_size = 1000  # Adjust based on memory constraints

        # Process each split
        for split, size in split_sizes.items():
            logger.debug(f"Generating {size} polygons for '{split}' split using native FPG...")

            # Create output memory-mapped array
            shape = (size, vertex_count + 1, 2)  # +1 for closing vertex
            output_file = self.create_output_file(algorithm, path_manager, shape, split)

            # Process in chunks to avoid memory issues
            for chunk_start in range(0, size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, size)
                chunk_size_actual = chunk_end - chunk_start
                logger.debug(f"  Processing chunk {chunk_start}-{chunk_end} ({chunk_size_actual} polygons)")

                # Create tasks for parallel processing
                tasks = self._create_tasks(
                    chunk_size_actual, num_processes, vertex_count, initial_vertices, radius
                )

                # Execute tasks and write results to output file
                chunk_polygons = self._process_tasks(tasks, num_processes)
                write_chunk_to_file(output_file, chunk_polygons, chunk_start, shape, dtype="float64")
                logger.debug(f"    Wrote {chunk_size_actual} polygons to {output_file}")

                # Memory management
                del chunk_polygons  # Free memory
                gc.collect()  # Force garbage collection

        logger.info(f"Successfully generated {num_samples} polygons using native FPG")

    def _create_tasks(
            self,
            chunk_size: int,
            num_processes: int,
            vertex_count: int,
            initial_vertices: int,
            radius: float
    ) -> List[FPGNativeTask]:
        """
        Create tasks for parallel polygon generation.

        Args:
            chunk_size: Size of the current chunk to process.
            num_processes: Number of worker processes to use.
            vertex_count: Number of vertices for each polygon.
            initial_vertices: Number of vertices in the initial regular polygon.
            radius: Radius of the initial regular polygon.

        Returns:
            List[FPGNativeTask]: List of task tuples for worker processes.
        """
        batches: List[Tuple[int, int]] = distribute_work(chunk_size, num_processes)

        tasks = []
        for i, batch in enumerate(batches):
            batch_size = batch[1] - batch[0]
            tasks.append(FPGNativeTask(i, batch_size, vertex_count, initial_vertices, radius))

        return tasks

    def _process_tasks(
            self,
            tasks: List[FPGNativeTask],
            num_processes: int
    ) -> np.ndarray:
        """
        Process tasks in parallel and combine results.

        Args:
            tasks: List of task tuples for worker processes.
            num_processes: Number of worker processes to use.

        Returns:
            np.ndarray: Combined array of polygons.

        Raises:
            RuntimeError: If polygon generation fails.
        """
        # Some batches may be empty when chunk_size < num_processes; drop them so
        # genpoly_fpg is never asked for zero polygons (its output-shape probe
        # indexes all_polygons[0]).
        tasks = [t for t in tasks if t.count > 0]

        # Create a process pool and execute tasks
        with multiprocessing.Pool(processes=min(num_processes, len(tasks))) as pool:
            # Use map instead of imap to ensure ordered results
            partial_results = pool.map(self._generate_batch, tasks)

        # Combine partial results
        if not partial_results:
            raise RuntimeError("No polygons generated")

        return np.concatenate(partial_results, axis=0)

    def create_output_file(self, algorithm: str, path_manager: PathManager, shape: Tuple[int, ...], split: str) -> Path:
        """
        Create a memory-mapped output file for storing generated polygons.
        """
        output_file = path_manager.get_processed_path(self.get_full_name(), algorithm, split)
        create_empty_memory_mapped_file(output_file, shape, dtype="float64")
        return output_file