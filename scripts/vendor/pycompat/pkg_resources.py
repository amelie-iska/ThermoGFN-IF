"""Minimal pkg_resources compatibility shim for offline envs.

Only implements `resource_filename`, which is sufficient for ProDy's
datafile lookup in this project.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    spec = find_spec(package_or_requirement)
    if spec is None:
        raise ModuleNotFoundError(f"Cannot resolve package for resource lookup: {package_or_requirement}")

    if spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        base = Path(spec.origin).parent
    else:
        raise RuntimeError(f"Unable to determine package path for: {package_or_requirement}")

    return str((base / resource_name).resolve())

