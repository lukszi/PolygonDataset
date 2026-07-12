"""
Shared pytest fixtures and helpers for the polygon_dataset test suite.

The suite is intentionally minimal (a smoke / regression net), not exhaustive:
it verifies that every module imports, that the Hydra configs compose (including
each machine profile), that ``PathManager`` builds the expected paths, and that
the native RPG generation + canonicalization pipeline runs end to end.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


def _configs_dir() -> Path:
    """Return the on-disk path of the packaged ``polygon_dataset/configs`` dir."""
    import polygon_dataset.configs as configs_pkg

    return Path(configs_pkg.__file__).parent


def machine_profile_names() -> List[str]:
    """Discover the available machine profiles (``configs/machine/*.yaml``)."""
    machine_dir = _configs_dir() / "machine"
    return sorted(p.stem for p in machine_dir.glob("*.yaml"))


def compose_config(overrides: Optional[List[str]] = None):
    """
    Compose the root Hydra config from the installed ``polygon_dataset.configs``
    module, returning the resolved ``DictConfig``.

    A fresh ``initialize_config_module`` context is entered per call so tests do
    not leak global Hydra state into one another.
    """
    from hydra import compose, initialize_config_module

    # Importing the schema module registers the structured-config nodes.
    import polygon_dataset.config.config_schema  # noqa: F401

    with initialize_config_module(
        config_module="polygon_dataset.configs", version_base=None
    ):
        return compose(config_name="config", overrides=overrides or [])


def _rpg_binary_generator_config() -> Optional[Dict[str, Any]]:
    """
    Build the ``rpg_binary`` generator config from the default composed config,
    or ``None`` when the compiled ``rpg`` binary is not present on this host.
    """
    from omegaconf import OmegaConf

    cfg = compose_config()
    gen_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.generators, resolve=True)

    bin_dir = gen_cfg.get("bin_dir")
    if not bin_dir:
        return None
    if not (Path(bin_dir) / "rpg").exists():
        return None
    return gen_cfg


@pytest.fixture(scope="session")
def rpg_binary_config() -> Dict[str, Any]:
    """
    Session fixture yielding a ready-to-use ``rpg_binary`` generator config.

    Skips the depending test when the compiled ``rpg`` binary is unavailable,
    keeping the suite green on hosts without the generator binaries.
    """
    cfg = _rpg_binary_generator_config()
    if cfg is None:
        pytest.skip("rpg binary not available on this host (machine.bin_dir/rpg missing)")
    return cfg


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    """A throwaway dataset root under pytest's tmp_path."""
    root = tmp_path / "datasets"
    root.mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)
