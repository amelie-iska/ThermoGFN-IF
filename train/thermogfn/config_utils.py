"""YAML config loading helpers for orchestration scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root_from(path: Path) -> Path:
    """Find repository root from any file path under this project."""
    for parent in path.resolve().parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load YAML config file and always return a mapping."""
    if not path:
        return {}
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config path not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {p}")
    return data


def cfg_get(cfg: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested key from dict using dotted path syntax."""
    cur: Any = cfg
    for key in key_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

