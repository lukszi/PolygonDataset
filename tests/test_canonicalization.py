"""
Canonicalization correctness tests.

These fold the invariant checks from the former standalone
``transformers/validate_canonicalization.py`` script into fast assertions
(dropping its matplotlib visualisation and psutil benchmarking, which were
demonstrations rather than pass/fail checks).

Canonicalization rotates each polygon so it starts at its lexicographically
smallest vertex (min x, then min y), while preserving closure and the vertex
multiset.
"""

import numpy as np
import pytest

from polygon_dataset.transformers import Transformer, CanonicalizeStrategy


def _make_transformer() -> Transformer:
    strategy = CanonicalizeStrategy({"name": "canonicalize"})
    # Single process keeps the test deterministic and avoids pool spawn overhead.
    return Transformer(strategy, chunk_size=1000, min_vertices=3, num_processes=1)


def _example_polygons() -> np.ndarray:
    """Three closed polygons (first vertex repeated as last), 6 vertices each."""
    square = np.array(
        [[1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    )
    triangle = np.array(
        [[1.0, 1.0], [1.5, 0.5], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    )
    pentagon = np.array(
        [[3.0, 2.0], [4.0, 4.0], [2.0, 5.0], [1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]
    )
    return np.stack([square, triangle, pentagon])


def _lex_min_vertex(open_poly: np.ndarray) -> np.ndarray:
    x = open_poly[:, 0]
    y = open_poly[:, 1]
    min_x = x.min()
    min_y = y[x == min_x].min()
    return np.array([min_x, min_y])


def _assert_canonical(original: np.ndarray, canon: np.ndarray) -> None:
    # Shape is preserved.
    assert canon.shape == original.shape

    is_closed = np.array_equal(original[0], original[-1])
    if is_closed:
        # Closure is preserved.
        assert np.array_equal(canon[0], canon[-1])
        open_orig = original[:-1]
        open_canon = canon[:-1]
    else:
        open_orig = original
        open_canon = canon

    # First vertex is the lexicographically smallest one.
    assert np.array_equal(open_canon[0], _lex_min_vertex(open_orig))

    # Canonicalization is a pure rotation: the vertex multiset is unchanged.
    sorted_orig = open_orig[np.lexsort((open_orig[:, 1], open_orig[:, 0]))]
    sorted_canon = open_canon[np.lexsort((open_canon[:, 1], open_canon[:, 0]))]
    assert np.allclose(sorted_orig, sorted_canon)


@pytest.mark.transformer
def test_example_polygons_canonicalized():
    polygons = _example_polygons()
    canon = _make_transformer().transform(polygons)

    assert isinstance(canon, np.ndarray)
    for i in range(polygons.shape[0]):
        _assert_canonical(polygons[i], canon[i])

    # The pentagon's lexicographically smallest vertex is [1, 3] (min x = 1);
    # it must lead after canonicalization.
    assert np.array_equal(canon[2][0], np.array([1.0, 3.0]))


@pytest.mark.transformer
def test_random_batch_canonicalized():
    rng = np.random.default_rng(1234)
    batch_size, n_open = 25, 12
    pts = rng.random((batch_size, n_open, 2)) * 10.0
    # Close each polygon by repeating the first vertex.
    closed = np.concatenate([pts, pts[:, :1, :]], axis=1)

    canon = _make_transformer().transform(closed)
    assert canon.shape == closed.shape
    for i in range(batch_size):
        _assert_canonical(closed[i], canon[i])


@pytest.mark.transformer
def test_idempotent():
    polygons = _example_polygons()
    transformer = _make_transformer()
    once = transformer.transform(polygons)
    twice = transformer.transform(once.copy())
    assert np.allclose(once, twice)
