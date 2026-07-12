"""
Import-walk smoke test.

Walks every submodule of :mod:`polygon_dataset` and imports it, failing with a
readable report if any module cannot be imported. This catches syntax errors,
broken relative imports, and missing (non-optional) dependencies across the
whole package in one cheap test.
"""

import importlib
import pkgutil

import pytest

import polygon_dataset


def _iter_module_names():
    names = []

    def _on_error(name: str) -> None:
        # Record packages that blow up while being walked so the test still fails.
        names.append(name)

    for info in pkgutil.walk_packages(
        polygon_dataset.__path__, prefix="polygon_dataset.", onerror=_on_error
    ):
        names.append(info.name)
    return sorted(set(names))


@pytest.mark.unit
def test_all_submodules_import():
    failures = {}
    module_names = _iter_module_names()

    # Guard against the walk silently finding nothing.
    assert len(module_names) > 10, f"suspiciously few modules discovered: {module_names}"

    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want to report every failure
            failures[name] = repr(exc)

    assert not failures, "modules failed to import:\n" + "\n".join(
        f"  {name}: {err}" for name, err in sorted(failures.items())
    )
