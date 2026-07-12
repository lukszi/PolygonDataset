"""
End-to-end generation smoke test.

Generates a handful of polygons through the real RPG generation route (the
``rpg_binary`` generator driving the compiled ``rpg`` executable, which is the
default generator in ``config.yaml``), then reads the raw ``.line`` output back
and canonicalizes it. The test is skipped automatically when the generator
binary is not present, so the suite stays green on hosts without it.

Kept deliberately tiny (5 polygons, 12 vertices) to keep runtime short.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from polygon_dataset.core import PathManager
from polygon_dataset.generators import get_generator
from polygon_dataset.transformers import Transformer, CanonicalizeStrategy
from polygon_dataset.utils.read_line_file import read_line_file

NUM_SAMPLES = 5
VERTEX_COUNT = 12


@pytest.mark.integration
@pytest.mark.generator
def test_generate_read_canonicalize(rpg_binary_config: Dict[str, Any], dataset_dir: Path):
    pm = PathManager(base_path=dataset_dir, dataset_name="e2e", create_dirs=True)

    generator_cls = get_generator("rpg_binary")
    generator = generator_cls(rpg_binary_config)

    # Put every sample in the train split for a simple, deterministic layout.
    generator.generate(
        pm,
        vertex_count=VERTEX_COUNT,
        num_samples=NUM_SAMPLES,
        split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
    )

    raw_paths = pm.get_raw_paths("rpg_binary", "train")
    assert len(raw_paths) == NUM_SAMPLES, "expected one .line file per sample"

    polygons = [read_line_file(p) for p in raw_paths]
    for poly in polygons:
        assert poly.ndim == 2 and poly.shape[1] == 2
        # rpg emits the requested vertices plus a closing vertex.
        assert poly.shape[0] >= VERTEX_COUNT
        assert np.isfinite(poly).all()

    # The generated polygons must survive canonicalization.
    min_v = min(p.shape[0] for p in polygons)
    batch = np.stack([p[:min_v] for p in polygons])
    strategy = CanonicalizeStrategy({"name": "canonicalize"})
    transformer = Transformer(strategy, chunk_size=1000, min_vertices=3, num_processes=1)
    canon = transformer.transform(batch)

    assert canon.shape == batch.shape
    # First vertex is lexicographically smallest for each canonicalized polygon.
    for i in range(canon.shape[0]):
        poly = canon[i]
        open_poly = poly[:-1] if np.array_equal(poly[0], poly[-1]) else poly
        min_x = open_poly[:, 0].min()
        min_y = open_poly[open_poly[:, 0] == min_x][:, 1].min()
        assert open_poly[0, 0] == min_x
        assert open_poly[0, 1] == min_y
