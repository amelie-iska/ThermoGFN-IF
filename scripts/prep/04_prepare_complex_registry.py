#!/usr/bin/env python3
"""Prepare registry for mixed-modality complex splits when available."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/rfd3_splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.io_utils import write_json
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("prep.complex_registry", level=args.log_level)
    base = root / args.data_root
    patterns = ["*dimer*", "*ligand*", "*complex*"]
    entries: list[dict] = []
    logger.info("Scanning complex split patterns under %s", base)
    for pat in iter_progress(patterns, total=len(patterns), desc="registry:patterns", no_progress=args.no_progress):
        for p in sorted(base.glob(pat)):
            if not p.is_dir():
                continue
            train_dir = p / "train"
            test_dir = p / "test"
            entries.append(
                {
                    "split_name": p.name,
                    "path": str(p),
                    "has_train": train_dir.exists(),
                    "has_test": test_dir.exists(),
                    "status": "ready" if train_dir.exists() and test_dir.exists() else "pending",
                }
            )

    payload = {
        "registry_version": "v1",
        "data_root": str(base),
        "entries": entries,
    }
    write_json(root / args.output, payload)
    logger.info("Complex registry written entries=%d elapsed=%.2fs", len(entries), time.perf_counter() - t0)
    print(root / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
