# utils/chunking.py

import gc
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.lib.format import open_memmap


def create_empty_memory_mapped_file(output_file: Path, shape: Tuple, dtype: str = 'float64') -> None:
    """Create a memory-mapped file with specified shape and dtype."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    memmap = open_memmap(output_file, dtype=dtype, mode='w+', shape=shape)
    memmap.flush()
    del memmap


def write_chunk_to_file(output_file: Path, data: np.ndarray, chunk_start: int,
                        shape: Tuple, dtype: str = 'float64') -> None:
    """Write a chunk of data to a memory-mapped file."""
    memmap = open_memmap(output_file, dtype=dtype, mode='r+', shape=shape)
    memmap[chunk_start:chunk_start + len(data)] = data
    memmap.flush()
    del memmap


def distribute_work(total_items: int, num_processes: int) -> List[Tuple[int, int]]:
    """Divide work evenly among processes."""
    batch_size = total_items // num_processes
    remainder = total_items % num_processes

    distribution = []
    for i in range(num_processes):
        start = i * batch_size + min(i, remainder)
        end = min((i + 1) * batch_size + min(i + 1, remainder), total_items)
        if end > start:
            distribution.append((start, end))

    return distribution


def load_chunk(file_path: Path, chunk_start: int, chunk_size: int) -> np.ndarray:
    """Load a chunk of data from a file, with proper memory management."""
    data = np.load(file_path, mmap_mode='r')
    chunk_end = min(chunk_start + chunk_size, len(data))
    chunk_data = data[chunk_start:chunk_end].copy()
    del data
    gc.collect()
    return chunk_data
