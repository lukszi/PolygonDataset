#!/usr/bin/env python3
"""
Tests for the polygon canonicalization transformer.

This module provides tests and benchmarks for the CanonicalizeTransformer class,
demonstrating its functionality and performance characteristics.
"""

import gc
import os
import time
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import psutil

from polygon_dataset.transformers.canonicalize import CanonicalizeTransformer


def test_canonicalization():
    """
    Test function that demonstrates polygon canonicalization by creating sample
    polygons, canonicalizing them, and printing the results.

    The test creates sample polygons, canonicalizes them using the transformer,
    and visualizes the original and canonicalized versions side by side.
    """
    # Create some sample polygons (all closed - first and last vertices are identical)
    display_polygons, names, polygons = create_example_polygons()

    # Initialize the transformer
    transformer = CanonicalizeTransformer({"name": "canonicalize"})

    # Canonicalize the polygons
    canonicalized = transformer.transform(polygons)

    # Print and visualize results
    plt.figure(figsize=(15, 10))

    for i, (poly, name) in enumerate(zip(display_polygons, names)):
        # Print original and canonicalized vertices
        print(f"\n{name}:")
        print("Original vertices:")
        for j, vertex in enumerate(poly):
            print(f"  {j}: {vertex}")

        print("Canonicalized vertices:")
        canon_poly = canonicalized[i]
        # Remove NaN values if present
        canon_poly = canon_poly[~np.isnan(canon_poly).any(axis=1)]
        for j, vertex in enumerate(canon_poly):
            print(f"  {j}: {vertex}")

        # Plot the polygon
        plt.subplot(2, 3, i + 1)
        x, y = poly[:, 0], poly[:, 1]
        plt.plot(x, y, 'b-o')
        plt.plot(x[0], y[0], 'go', markersize=10, label='Original Start')  # Original start

        # Find lexicographically smallest vertex in original
        min_x = np.min(poly[:, 0])
        min_x_indices = np.where(poly[:, 0] == min_x)[0]  # Get all indices with minimum x
        lex_min_idx = min_x_indices[np.argmin(poly[min_x_indices, 1])]  # Get index with min y among min x
        plt.plot(x[lex_min_idx], y[lex_min_idx], 'ro', markersize=10, label='Lexicographically Smallest')

        plt.title(f"Original {name}")
        plt.grid(True)
        plt.legend()

        # Plot canonicalized polygon
        plt.subplot(2, 3, i + 4)
        cx, cy = canon_poly[:, 0], canon_poly[:, 1]
        plt.plot(cx, cy, 'b-o')
        plt.plot(cx[0], cy[0], 'go', markersize=10, label='New Start')  # New start
        plt.title(f"Canonicalized {name}")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('canonicalization_test.png')
    plt.close()
    print("\nVisualization saved as 'canonicalization_test.png'")


def create_example_polygons() -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Create a batch of example polygons with the same number of vertices.

    Each polygon will have 6 vertices total (5 unique points plus the closing vertex).
    When necessary, additional vertices are added by interpolating along edges to
    maintain the original shape while ensuring all polygons have the same structure.

    Returns:
        Tuple containing:
            - display_polygons: List of individual polygons
            - names: List of polygon names
            - polygon_batch: NumPy array of shape (3, 6, 2) containing all polygons
    """
    # 1. Square with 6 vertices (including closing vertex)
    square = np.array([
        [1.0, 1.0],  # top-right
        [0.5, 1.0],  # interpolated point on top edge
        [0.0, 1.0],  # top-left
        [0.0, 0.0],  # bottom-left
        [1.0, 0.0],  # bottom-right
        [1.0, 1.0],  # closing vertex (same as first)
    ])

    # 2. Triangle with 6 vertices (including closing vertex)
    triangle = np.array([
        [1.0, 1.0],  # top
        [1.5, 0.5],  # interpolated point on right edge
        [2.0, 0.0],  # bottom-right
        [1.0, 0.0],  # interpolated point on bottom edge
        [0.0, 0.0],  # bottom-left
        [1.0, 1.0],  # closing vertex
    ])

    # 3. Pentagon with 6 vertices (already has the right number)
    pentagon = np.array([
        [3.0, 2.0],  # arbitrary starting point
        [4.0, 4.0],
        [2.0, 5.0],
        [1.0, 3.0],
        [2.0, 1.0],  # this is the lexicographically smallest vertex
        [3.0, 2.0],  # closing vertex
    ])

    # Create the batch directly without padding
    polygons = np.array([square, triangle, pentagon])

    # Create display information (no need to handle different sizes now)
    display_polygons = [square, triangle, pentagon]
    names = ["Square", "Triangle", "Pentagon"]

    return display_polygons, names, polygons


def generate_random_polygons(batch_size: int, num_vertices: int) -> np.ndarray:
    """
    Generates a batch of random closed polygons.

    Args:
        batch_size: Number of polygons to generate
        num_vertices: Number of vertices per polygon (including closing vertex)

    Returns:
        NumPy array of shape (batch_size, num_vertices, 2)
    """
    # First generate random points for each polygon (without closure)
    polygons = np.random.rand(batch_size, num_vertices - 1, 2) * 10.0

    # Add closure by duplicating the first vertex as the last vertex
    closed_polygons = np.zeros((batch_size, num_vertices, 2))
    closed_polygons[:, :-1, :] = polygons
    closed_polygons[:, -1, :] = polygons[:, 0, :]

    return closed_polygons


def benchmark_canonicalization():
    """
    Benchmarks the polygon canonicalization transformer by testing its performance
    on batches of randomly generated polygons of varying sizes.

    This function measures execution time and memory usage across different batch
    sizes to evaluate the algorithm's efficiency and scaling characteristics.
    """
    # Configuration for the benchmark
    vertex_count = 10  # Number of vertices per polygon (including closing vertex)
    batch_sizes = [10, 100, 1000, 10000, 100000]  # Different batch sizes to test
    repeat_tests = 3  # Number of times to repeat each test for more reliable results

    # Initialize results containers
    execution_times = []
    memory_usages = []

    # Memory monitoring process
    process = psutil.Process(os.getpid())

    # Initialize the transformer once to avoid counting initialization time
    transformer = CanonicalizeTransformer({"name": "canonicalize"})

    print("Starting canonicalization benchmark...")

    for batch_size in batch_sizes:
        batch_times = []
        batch_memory = []

        print(f"\nTesting batch size: {batch_size} polygons")

        for i in range(repeat_tests):
            # Generate random polygons
            print(f"  Test {i + 1}/{repeat_tests}: Generating {batch_size} random polygons...")
            polygons = generate_random_polygons(batch_size, vertex_count)

            # Force garbage collection to get clean memory measurement
            gc.collect()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

            # Time the canonicalization
            print(f"  Test {i + 1}/{repeat_tests}: Canonicalizing polygons...")
            start_time = time.time()
            canonicalized = transformer.transform(polygons)
            end_time = time.time()

            # Calculate execution time and memory usage
            execution_time = end_time - start_time
            peak_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
            memory_used = peak_memory - initial_memory

            batch_times.append(execution_time)
            batch_memory.append(memory_used)

            print(
                f"  Test {i + 1}/{repeat_tests}: Completed in {execution_time:.4f} seconds, used {memory_used:.2f} MB")

            # Verify a few results to ensure correctness
            if i == 0:
                for j in range(min(3, batch_size)):
                    # The first point should be lexicographically smallest
                    first_point = canonicalized[j, 0]
                    # Check all other points (excluding closing vertex)
                    for k in range(1, vertex_count - 1):
                        point = canonicalized[j, k]
                        # Verify lexicographic ordering
                        if point[0] < first_point[0] or (point[0] == first_point[0] and point[1] < first_point[1]):
                            print(f"  WARNING: Canonicalization error detected in polygon {j}")
                            break

            # Clean up to free memory
            del polygons
            del canonicalized
            gc.collect()

        # Record average results for this batch size
        avg_time = sum(batch_times) / len(batch_times)
        avg_memory = sum(batch_memory) / len(batch_memory)
        execution_times.append(avg_time)
        memory_usages.append(avg_memory)

        print(f"Batch size: {batch_size}, Average time: {avg_time:.4f} seconds, Average memory: {avg_memory:.2f} MB")

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Execution time plot
    plt.subplot(2, 1, 1)
    plt.plot(batch_sizes, execution_times, marker='o', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size (number of polygons)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Canonicalization Performance: Execution Time')
    plt.grid(True, which="both", ls="--")

    # Memory usage plot
    plt.subplot(2, 1, 2)
    plt.plot(batch_sizes, memory_usages, marker='o', linewidth=2, color='green')
    plt.xscale('log')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Batch Size (number of polygons)')
    plt.title('Canonicalization Performance: Memory Usage')
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('canonicalization_benchmark.png')
    plt.close()

    print("\nBenchmark complete! Results:")
    print("| Batch Size | Execution Time (s) | Memory Usage (MB) |")
    print("|------------|---------------------|-------------------|")
    for i, batch_size in enumerate(batch_sizes):
        print(f"| {batch_size:10d} | {execution_times[i]:19.4f} | {memory_usages[i]:17.2f} |")

    print("\nVisualization saved as 'canonicalization_benchmark.png'")


# If running this file directly, execute the tests
if __name__ == "__main__":
    test_canonicalization()
    benchmark_canonicalization()