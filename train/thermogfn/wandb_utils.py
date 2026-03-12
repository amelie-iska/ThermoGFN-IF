"""Lightweight Weights & Biases helpers for orchestration and training metrics."""

from __future__ import annotations

import importlib
import json
import math
import netrc
import os
from pathlib import Path
import sys
from typing import Any


def _import_real_wandb():
    """Import the installed wandb package, not the local ./wandb artifact dir."""

    try:  # pragma: no cover - depends on runtime env
        mod = importlib.import_module("wandb")
        if hasattr(mod, "init") and hasattr(mod, "log"):
            return mod
    except Exception:  # noqa: BLE001
        pass

    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    original_path = list(sys.path)
    try:  # pragma: no cover - depends on runtime env
        filtered: list[str] = []
        for entry in original_path:
            if entry in {"", "."}:
                continue
            try:
                resolved = Path(entry).resolve()
            except Exception:  # noqa: BLE001
                filtered.append(entry)
                continue
            if resolved == cwd or resolved == repo_root:
                continue
            filtered.append(entry)
        sys.path[:] = filtered
        sys.modules.pop("wandb", None)
        mod = importlib.import_module("wandb")
        if hasattr(mod, "init") and hasattr(mod, "log"):
            return mod
    except Exception:  # noqa: BLE001
        return None
    finally:
        sys.path[:] = original_path
    return None


wandb = _import_real_wandb()


def flatten_metrics(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        full = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=full))
            continue
        if isinstance(value, (list, tuple)):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            out[full] = value
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            out[full] = value
            continue
        if isinstance(value, str):
            out[full] = value
    return out


def parse_tags(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def read_history_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_wandb_credentials() -> bool:
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if api_key:
        return True
    try:
        auth = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    for host in ("api.wandb.ai", "wandb.ai"):
        if host in auth.hosts:
            return True
    return False


class WandbRun:
    """Thin wrapper that keeps wandb optional and centralized."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None,
        entity: str | None,
        mode: str | None,
        name: str | None,
        group: str | None,
        job_type: str | None,
        tags: list[str] | None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None
        self._mode = mode
        if not self.enabled:
            return
        if wandb is None:
            raise RuntimeError("wandb logging requested, but wandb is not installed in this environment")
        resolved_mode = mode
        if resolved_mode in {None, "", "auto"}:
            env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
            if env_mode in {"online", "offline", "disabled"}:
                resolved_mode = env_mode
            else:
                resolved_mode = "online" if _has_wandb_credentials() else "offline"
        self._run = wandb.init(
            project=project,
            entity=entity,
            mode=resolved_mode,
            name=name,
            group=group,
            job_type=job_type,
            tags=tags or [],
            config=config or {},
            reinit="finish_previous",
        )

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self.active:
            return
        payload = flatten_metrics(metrics)
        if not payload:
            return
        wandb.log(payload, step=step)

    def log_prefixed(self, prefix: str, metrics: dict[str, Any], *, step: int | None = None) -> None:
        self.log(flatten_metrics(metrics, prefix=prefix), step=step)

    def log_history(
        self,
        rows: list[dict[str, Any]],
        *,
        prefix: str,
        step_key: str = "step",
        base_step: int | None = None,
    ) -> None:
        if not self.active:
            return
        for idx, row in enumerate(rows):
            step = row.get(step_key, idx)
            if base_step is not None:
                try:
                    step = int(base_step) + int(step)
                except Exception:  # noqa: BLE001
                    step = base_step + idx
            self.log(flatten_metrics(row, prefix=prefix), step=int(step) if step is not None else None)

    def summary_update(self, metrics: dict[str, Any], *, prefix: str = "") -> None:
        if not self.active:
            return
        payload = flatten_metrics(metrics, prefix=prefix)
        for key, value in payload.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
