"""Shared logging and progress utilities for CLI scripts."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

try:  # pragma: no cover - import fallback behavior is environment-dependent
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def configure_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Configure timestamped process-wide logging and return a named logger."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    kwargs = {
        "level": resolved,
        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    try:
        # Python >=3.8 supports `force`, which is the cleanest way to reset
        # root handlers in subprocess-heavy orchestration.
        logging.basicConfig(force=True, **kwargs)
    except Exception:
        # Python 3.7 fallback (e.g., SPURS env): remove root handlers manually.
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        logging.basicConfig(**kwargs)
    return logging.getLogger(name)


def progress_enabled(no_progress: bool = False) -> bool:
    """Central progress toggle with env override."""
    env = os.getenv("THERMOGFN_NO_PROGRESS", "").strip().lower()
    return (not no_progress) and env not in {"1", "true", "yes", "y", "on"}


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    no_progress: bool = False,
    leave: bool = False,
) -> Iterator[T]:
    """Return an iterator wrapped with tqdm when available/enabled."""
    if not progress_enabled(no_progress):
        for item in iterable:
            yield item
        return
    if tqdm is None:
        label = desc or "progress"
        if total is None:
            count = 0
            print(f"{label}: start", flush=True)
            for item in iterable:
                count += 1
                if count == 1 or (count % 100) == 0:
                    print(f"{label}: {count}", flush=True)
                yield item
            print(f"{label}: done n={count}", flush=True)
            return
        step = max(1, int(total) // 20)
        count = 0
        print(f"{label}: start total={total}", flush=True)
        for item in iterable:
            count += 1
            if count == 1 or count == int(total) or (count % step) == 0:
                print(f"{label}: {count}/{total}", flush=True)
            yield item
        return
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave)


def reset_peak_vram_tracking() -> None:
    """Reset CUDA peak memory tracker when CUDA is available."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if not torch.cuda.is_available():
        return
    try:
        device_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception:  # noqa: BLE001
        return


def log_peak_vram(logger: logging.Logger, *, label: str = "inference") -> None:
    """Log peak VRAM usage for current CUDA device, if available."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        logger.info("%s peak_vram=unavailable torch_not_importable", label)
        return
    if not torch.cuda.is_available():
        logger.info("%s peak_vram=cpu_only", label)
        return
    try:
        device_index = torch.cuda.current_device()
        torch.cuda.synchronize(device_index)
        peak_bytes = int(torch.cuda.max_memory_allocated(device_index))
        total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
        peak_gib = peak_bytes / float(1024**3)
        total_gib = total_bytes / float(1024**3)
        peak_frac = (peak_bytes / float(total_bytes)) if total_bytes > 0 else 0.0
        device_name = torch.cuda.get_device_name(device_index)
        logger.info(
            "%s peak_vram_bytes=%d peak_vram_gib=%.3f peak_vram_frac=%.3f device=%d name=%s",
            label,
            peak_bytes,
            peak_gib,
            peak_frac,
            device_index,
            device_name,
        )
        logger.info("%s total_vram_gib=%.3f", label, total_gib)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s peak_vram_log_failed error=%s", label, exc)
