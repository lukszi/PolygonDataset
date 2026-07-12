"""
Tests for :class:`polygon_dataset.core.PathManager`'s legacy (old-format) mode.

Old-format datasets (e.g. the historical ``v7`` set) name their files without
the generation-algorithm token: ``extracted/train_spg.npy`` and
``transformed/train_spg_res11.npy`` rather than the new
``train_spg_2opt.npy`` / ``train_spg_2opt_res11.npy``. ``PathManager`` detects
these datasets from their ``config.json`` and constructs matching paths while
keeping the same 4-argument public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polygon_dataset.core import PathManager


# --- Fabricated old-format dataset ------------------------------------------

# Mirrors the real v7 config.json: generator_configs keyed by bare generator
# names, each carrying a top-level ``generator_type``/``algorithm`` rather than
# the new ``implementation``/``params`` structure.
_LEGACY_CONFIG = {
    "dataset_info": {
        "name": "legacy_ds",
        "size": 100,
        "dimensionality": 2,
        "dataset_state": "transformed",
        "vertex_count": 88,
        "split_ratios": {"train": 0.9, "val": 0.05, "test": 0.05},
    },
    "generator_configs": {
        "spg": {"vertices": 87, "generator_type": "spg", "algorithm": "2opt"},
        "fpg": {"vertices": 87, "generator_type": "fpg", "initial_vertices": 20},
        "rpg": {"vertices": 87, "generator_type": "rpg", "algorithm": "2opt"},
    },
    "transformation_config": {
        "algorithm": "visvalingam-whyatt",
        "resolution_steps": [11, 22, 44],
    },
}

_NEW_CONFIG = {
    "dataset_info": {
        "name": "new_ds",
        "size": 100,
        "dimensionality": 2,
        "dataset_state": "transformed",
        "vertex_count": 88,
        "split_ratios": {"train": 0.9, "val": 0.05, "test": 0.05},
    },
    "generator_configs": {
        "rpg_binary": {
            "name": "rpg",
            "implementation": "binary",
            "params": {"algorithm": "2opt"},
        }
    },
    "transformation_config": {
        "algorithm": "visvalingam",
        "resolution_steps": [11, 22, 44],
    },
}


def _build_legacy_dataset(root: Path, name: str = "legacy_ds") -> Path:
    """
    Fabricate a minimal old-format dataset directory (config + empty .npy files)
    that mimics v7's naming, returning the dataset root.
    """
    ds = root / name
    (ds / "extracted").mkdir(parents=True)
    (ds / "transformed").mkdir(parents=True)
    (ds / "config.json").write_text(json.dumps(_LEGACY_CONFIG, indent=2))

    empty = np.zeros((0,), dtype=np.float64)
    for split in ("train", "val", "test"):
        for gen in ("spg", "fpg", "rpg"):
            np.save(ds / "extracted" / f"{split}_{gen}.npy", empty)
            for res in (11, 22, 44):
                np.save(ds / "transformed" / f"{split}_{gen}_res{res}.npy", empty)
    return ds


@pytest.fixture()
def legacy_pm(dataset_dir: Path) -> PathManager:
    _build_legacy_dataset(dataset_dir)
    return PathManager(base_path=dataset_dir, dataset_name="legacy_ds")


# --- Detection ---------------------------------------------------------------


@pytest.mark.unit
def test_detects_legacy_config():
    assert PathManager._is_legacy_config(_LEGACY_CONFIG) is True


@pytest.mark.unit
def test_new_config_is_not_legacy():
    assert PathManager._is_legacy_config(_NEW_CONFIG) is False


@pytest.mark.unit
def test_empty_or_missing_generator_configs_default_new():
    assert PathManager._is_legacy_config({}) is False
    assert PathManager._is_legacy_config({"generator_configs": {}}) is False


@pytest.mark.unit
def test_pm_legacy_property_true(legacy_pm: PathManager):
    assert legacy_pm.legacy is True


@pytest.mark.unit
def test_pm_without_config_defaults_new(dataset_dir: Path):
    pm = PathManager(base_path=dataset_dir, dataset_name="missing", create_dirs=True)
    assert pm.legacy is False


# --- Legacy path construction (algorithm token omitted) ----------------------


@pytest.mark.unit
def test_legacy_extracted_path_drops_algorithm(legacy_pm: PathManager, dataset_dir: Path):
    root = dataset_dir / "legacy_ds"
    # The algorithm argument is accepted but ignored in legacy mode.
    p = legacy_pm.get_processed_path("spg", "2opt", "train")
    assert p == root / "extracted" / "train_spg.npy"


@pytest.mark.unit
def test_legacy_resolution_path_drops_algorithm(legacy_pm: PathManager, dataset_dir: Path):
    root = dataset_dir / "legacy_ds"
    p = legacy_pm.get_resolution_path("fpg", "2opt", "val", 22)
    assert p == root / "transformed" / "val_fpg_res22.npy"


@pytest.mark.unit
def test_legacy_canonical_paths_drop_algorithm(legacy_pm: PathManager, dataset_dir: Path):
    root = dataset_dir / "legacy_ds"
    assert legacy_pm.get_canonical_path("rpg", "2opt", "test") == (
        root / "canonicalized" / "test_rpg.npy"
    )
    assert legacy_pm.get_canonical_path("rpg", "2opt", "test", 16) == (
        root / "canonicalized" / "test_rpg_res16.npy"
    )


@pytest.mark.unit
def test_legacy_ignores_algorithm_value(legacy_pm: PathManager):
    # Different algorithm tokens must not change the constructed path.
    a = legacy_pm.get_resolution_path("spg", "2opt", "train", 11)
    b = legacy_pm.get_resolution_path("spg", "anything_else", "train", 11)
    assert a == b


@pytest.mark.unit
def test_legacy_constructed_paths_exist_on_disk(legacy_pm: PathManager):
    # The fabricated files were written under the legacy names.
    assert legacy_pm.get_processed_path("spg", "2opt", "train").exists()
    assert legacy_pm.get_resolution_path("rpg", "2opt", "test", 44).exists()


# --- New-format datasets keep current behaviour ------------------------------


@pytest.mark.unit
def test_new_format_dataset_keeps_algorithm_token(dataset_dir: Path):
    ds = dataset_dir / "new_ds"
    ds.mkdir(parents=True)
    (ds / "config.json").write_text(json.dumps(_NEW_CONFIG, indent=2))

    pm = PathManager(base_path=dataset_dir, dataset_name="new_ds")
    assert pm.legacy is False
    assert pm.get_resolution_path("rpg_binary", "2opt", "train", 22) == (
        ds / "transformed" / "train_rpg_binary_2opt_res22.npy"
    )


# --- Read-only assertion against the real v7 dataset (skip if absent) --------

_V7_PATH = Path("/root/datasets/v7")


@pytest.mark.unit
@pytest.mark.skipif(
    not _V7_PATH.exists(),
    reason="real v7 dataset not present on this host",
)
def test_real_v7_paths_resolve(tmp_path: Path):
    # PathManager takes (base_path, dataset_name); point it at the real v7.
    pm = PathManager(base_path=_V7_PATH.parent, dataset_name=_V7_PATH.name)
    assert pm.legacy is True

    for split in ("train", "val", "test"):
        for gen in ("spg", "fpg", "rpg"):
            assert pm.get_processed_path(gen, "2opt", split).exists()
            for res in (11, 22, 44):
                assert pm.get_resolution_path(gen, "2opt", split, res).exists()
