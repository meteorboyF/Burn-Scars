"""Import every submodule. Guards the class of bug where an unanchored gitignore
(or a packaging step) silently drops a subpackage such as burnscars/data/."""
import importlib
import pkgutil

import burnscars

EXPECTED_SUBPACKAGES = {"data", "models", "train", "eval", "figures"}


def _all_modules():
    return [m.name for m in pkgutil.walk_packages(burnscars.__path__, "burnscars.")]


def test_every_submodule_imports():
    names = _all_modules()
    assert names, "no submodules found"
    for name in names:
        importlib.import_module(name)


def test_expected_subpackages_present():
    found = {m.name for m in pkgutil.iter_modules(burnscars.__path__) if m.ispkg}
    assert EXPECTED_SUBPACKAGES <= found, EXPECTED_SUBPACKAGES - found
