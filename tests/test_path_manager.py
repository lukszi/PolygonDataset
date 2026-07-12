"""
Unit tests for :class:`polygon_dataset.core.PathManager`.

Covers path construction (extracted / transformed / canonical / raw), directory
creation behaviour, raw-file discovery, and the full generator-name helper.
"""

from pathlib import Path

import pytest

from polygon_dataset.core import PathManager


@pytest.fixture()
def pm(dataset_dir: Path) -> PathManager:
    return PathManager(base_path=dataset_dir, dataset_name="ds", create_dirs=True)


@pytest.mark.unit
def test_create_dirs_makes_dataset_root(dataset_dir: Path):
    pm = PathManager(base_path=dataset_dir, dataset_name="ds", create_dirs=True)
    assert pm.dataset_path == dataset_dir / "ds"
    assert pm.dataset_path.is_dir()


@pytest.mark.unit
def test_no_create_dirs_does_not_make_root(dataset_dir: Path):
    pm = PathManager(base_path=dataset_dir, dataset_name="ds", create_dirs=False)
    assert not pm.dataset_path.exists()


@pytest.mark.unit
def test_directory_layout(pm: PathManager, dataset_dir: Path):
    root = dataset_dir / "ds"
    assert pm.get_raw_dir() == root / "raw"
    assert pm.get_extracted_dir() == root / "extracted"
    assert pm.get_transformed_dir() == root / "transformed"
    assert pm.get_canonical_dir() == root / "canonicalized"
    # Accessor methods create the directories on demand.
    assert pm.get_extracted_dir().is_dir()
    assert pm.get_transformed_dir().is_dir()


@pytest.mark.unit
def test_raw_split_dir(pm: PathManager, dataset_dir: Path):
    split_dir = pm.get_raw_split_dir("train", "rpg_binary")
    assert split_dir == dataset_dir / "ds" / "raw" / "train" / "rpg_binary"
    assert split_dir.is_dir()


@pytest.mark.unit
def test_processed_and_resolution_and_canonical_paths(pm: PathManager, dataset_dir: Path):
    root = dataset_dir / "ds"

    processed = pm.get_processed_path("rpg_binary", "2opt", "train")
    assert processed == root / "extracted" / "train_rpg_binary_2opt.npy"

    res = pm.get_resolution_path("rpg_binary", "visvalingam", "val", 32)
    assert res == root / "transformed" / "val_rpg_binary_visvalingam_res32.npy"

    canon_no_res = pm.get_canonical_path("rpg_binary", "canonicalize", "test")
    assert canon_no_res == root / "canonicalized" / "test_rpg_binary_canonicalize.npy"

    canon_res = pm.get_canonical_path("rpg_binary", "canonicalize", "test", 16)
    assert canon_res == root / "canonicalized" / "test_rpg_binary_canonicalize_res16.npy"


@pytest.mark.unit
def test_config_path(pm: PathManager, dataset_dir: Path):
    assert pm.get_config_path() == dataset_dir / "ds" / "config.json"


@pytest.mark.unit
def test_full_generator_name():
    assert PathManager.get_full_generator_name("rpg", "binary") == "rpg_binary"
    assert PathManager.get_full_generator_name("fpg", "native") == "fpg_native"


@pytest.mark.unit
def test_get_raw_paths_sorted_and_empty(pm: PathManager):
    # No files yet -> empty list (directory is created by the accessor).
    assert pm.get_raw_paths("rpg_binary", "train") == []

    split_dir = pm.get_raw_split_dir("train", "rpg_binary")
    for idx in (2, 0, 1):
        (split_dir / f"polygon_{idx:06d}.line").write_text("1\n0.0 0.0\n")
    # A non-.line file must be ignored.
    (split_dir / "notes.txt").write_text("ignore me")

    paths = pm.get_raw_paths("rpg_binary", "train")
    names = [p.name for p in paths]
    assert names == [
        "polygon_000000.line",
        "polygon_000001.line",
        "polygon_000002.line",
    ]
