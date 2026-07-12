"""
Native FPG generation smoke test.

Drives the ``fpg_native`` generator (the ``genpoly_fpg`` Python binding) through
the real ``FPGNativeGenerator.generate`` path, then loads the processed ``.npy``
back and asserts every polygon is a valid, closed polygon with exactly the
configured vertex count. The test is skipped automatically when ``genpoly_fpg``
is not installed (the ``native`` extra), mirroring the ``rpg_binary`` fixture's
skip-if-not-available behaviour so the suite stays green on hosts without it.

Kept deliberately tiny (6 polygons, 12 vertices) to keep runtime short.
"""

from pathlib import Path

import numpy as np
import pytest

from polygon_dataset.core import PathManager
from polygon_dataset.generators import get_generator

pytest.importorskip("genpoly_fpg", reason="native extra (genpoly_fpg) not installed")

NUM_SAMPLES = 6
VERTEX_COUNT = 12


@pytest.mark.integration
@pytest.mark.generator
def test_fpg_native_generates_valid_closed_polygons(dataset_dir: Path):
    pm = PathManager(base_path=dataset_dir, dataset_name="fpg_native_e2e", create_dirs=True)

    generator_cls = get_generator("fpg_native")
    generator = generator_cls(
        {
            "name": "fpg",
            "implementation": "native",
            "params": {"initial_vertices": 8, "radius": 1.0},
        }
    )

    # Put every sample in the train split for a simple, deterministic layout.
    generator.generate(
        pm,
        vertex_count=VERTEX_COUNT,
        num_samples=NUM_SAMPLES,
        split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
    )

    # FPG binary and native share the algorithm-token-free "default" naming.
    processed = pm.get_processed_path("fpg_native", "default", "train")
    assert processed.exists(), "expected a processed .npy for the train split"

    polygons = np.load(processed)
    # +1 for the closing vertex that repeats the first point.
    assert polygons.shape == (NUM_SAMPLES, VERTEX_COUNT + 1, 2)
    assert np.isfinite(polygons).all()

    for i in range(NUM_SAMPLES):
        poly = polygons[i]
        # Closed: first vertex repeated at the end.
        assert np.allclose(poly[0], poly[-1]), f"polygon {i} is not closed"

        open_poly = poly[:-1]
        assert open_poly.shape[0] == VERTEX_COUNT, f"polygon {i} has {open_poly.shape[0]} vertices"

        # Non-degenerate: shoelace area is strictly positive.
        x = open_poly[:, 0]
        y = open_poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert area > 1e-9, f"polygon {i} is degenerate (area={area})"
