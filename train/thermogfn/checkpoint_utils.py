"""Checkpoint retention helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _round_id_from_name(path: Path, prefix: str) -> int | None:
    pat = re.compile(rf"^{re.escape(prefix)}_round_(\d+)\.ckpt$")
    m = pat.match(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:  # noqa: BLE001
        return None


def prune_round_checkpoints(
    outdir: str | Path,
    *,
    prefix: str,
    max_keep: int,
    logger: Any | None = None,
) -> list[Path]:
    """Keep at most `max_keep` latest <prefix>_round_*.ckpt checkpoints."""
    if max_keep <= 0:
        return []
    p = Path(outdir)
    ckpts = sorted(p.glob(f"{prefix}_round_*.ckpt"))
    if len(ckpts) <= max_keep:
        return []

    def sort_key(x: Path) -> tuple[int, float, str]:
        rid = _round_id_from_name(x, prefix)
        # Unknown round ids are treated as oldest.
        if rid is None:
            rid = -1
        try:
            mtime = x.stat().st_mtime
        except Exception:  # noqa: BLE001
            mtime = 0.0
        return (rid, mtime, x.name)

    ckpts_sorted = sorted(ckpts, key=sort_key)
    to_remove = ckpts_sorted[: max(0, len(ckpts_sorted) - max_keep)]
    removed: list[Path] = []
    for c in to_remove:
        try:
            c.unlink(missing_ok=True)
            removed.append(c)
            if logger is not None:
                logger.info(
                    "Pruned old checkpoint prefix=%s max_keep=%d removed=%s",
                    prefix,
                    max_keep,
                    c,
                )
        except Exception:  # noqa: BLE001
            if logger is not None:
                logger.exception("Failed to prune checkpoint: %s", c)
    return removed

