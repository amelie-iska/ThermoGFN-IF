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


class _SimpleProgressBar:
    """Fallback progress helper with a tqdm-like surface."""

    def __init__(
        self,
        *,
        total: int | None,
        desc: str | None = None,
        leave: bool = False,
        unit: str | None = None,
    ) -> None:
        self.total = total
        self.desc = desc or "progress"
        self.leave = leave
        self.unit = unit or "item"
        self.count = 0
        self._closed = False
        self._print_start()

    def _print_start(self) -> None:
        if self.total is None:
            print(f"{self.desc}: start", flush=True)
        else:
            print(f"{self.desc}: start total={self.total}", flush=True)

    def _print_update(self, *, force: bool = False) -> None:
        if self._closed:
            return
        if self.total is None:
            if force or self.count == 1 or (self.count % 25) == 0:
                print(f"{self.desc}: {self.count} {self.unit}", flush=True)
            return
        step = max(1, int(self.total) // 20)
        if force or self.count == 1 or self.count == int(self.total) or (self.count % step) == 0:
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        self._print_update()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:  # noqa: ARG002
        payload = {}
        if ordered_dict:
            payload.update(dict(ordered_dict))
        payload.update(kwargs)
        if payload:
            summary = " ".join(f"{k}={v}" for k, v in payload.items())
            print(f"{self.desc}: {summary}", flush=True)

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:  # noqa: ARG002
        if s:
            print(f"{self.desc}: {s}", flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self._print_update(force=True)
        if self.leave:
            if self.total is None:
                print(f"{self.desc}: done n={self.count}", flush=True)
            else:
                print(f"{self.desc}: done total={self.total}", flush=True)
        self._closed = True


class _NullProgressBar:
    """No-op progress helper used when progress bars are disabled."""

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        return

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:  # noqa: ARG002
        return

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:  # noqa: ARG002
        return

    def close(self) -> None:
        return


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
        pbar = _SimpleProgressBar(total=total, desc=desc, leave=leave)
        for item in iterable:
            pbar.update(1)
            yield item
        pbar.close()
        return
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave)


def make_progress(
    *,
    total: int | None,
    desc: str | None = None,
    no_progress: bool = False,
    leave: bool = True,
    unit: str | None = None,
):
    """Return a tqdm-compatible progress bar or a lightweight fallback."""
    if not progress_enabled(no_progress):
        return _NullProgressBar()
    if tqdm is None:
        return _SimpleProgressBar(total=total, desc=desc, leave=leave, unit=unit)
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=leave, unit=unit)


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
