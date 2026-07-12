"""
Hydra configuration composition tests.

Verifies that the root config composes, that machine-specific paths are wired
through interpolation, and that every committed machine profile
(``configs/machine/*.yaml``) composes and resolves without error.
"""

from omegaconf import OmegaConf
import pytest

from conftest import compose_config, machine_profile_names


@pytest.mark.unit
def test_default_config_composes():
    cfg = compose_config()

    # Core sections are present.
    assert cfg.dataset.name
    assert cfg.dataset.vertex_count > 0
    assert cfg.generators is not None
    assert cfg.transform is not None

    # output_dir is interpolated from the selected machine profile.
    assert cfg.output_dir == cfg.machine.output_dir


@pytest.mark.unit
def test_machine_profiles_exist():
    profiles = machine_profile_names()
    # These two ship with the repo; guards against an empty/misnamed directory.
    assert "default" in profiles
    assert "wsl" in profiles


@pytest.mark.unit
@pytest.mark.parametrize("machine", machine_profile_names())
def test_each_machine_profile_composes(machine):
    cfg = compose_config(overrides=[f"machine={machine}"])

    # Fully resolve interpolations (${machine.*}, ${oc.env:...}) - raises on error.
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["machine"]["output_dir"]
    assert resolved["machine"]["bin_dir"]
    # The machine paths propagate into the top-level and generator configs.
    assert resolved["output_dir"] == resolved["machine"]["output_dir"]
    assert resolved["generators"]["bin_dir"] == resolved["machine"]["bin_dir"]


@pytest.mark.unit
def test_machine_override_changes_paths():
    wsl = compose_config(overrides=["machine=wsl"])
    assert wsl.machine.output_dir == "/root/datasets"
    assert wsl.machine.bin_dir == "/root/polygons/bin"
