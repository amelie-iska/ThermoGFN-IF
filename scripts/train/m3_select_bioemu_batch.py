#!/usr/bin/env python3
"""Select BioEmu acquisition subset from SPURS-scored pool."""

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
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.acquisition import score_bioemu_acquisition, select_top

    logger = configure_logging("train.select_bioemu", level=args.log_level)
    rows = read_records(root / args.input_path)
    logger.info("BioEmu acquisition scoring start: rows=%d budget=%d", len(rows), args.budget)
    for rec in iter_progress(rows, total=len(rows), desc="acq:bioemu", no_progress=args.no_progress):
        rec["acq_bioemu"] = score_bioemu_acquisition(rec)
    selected = select_top(rows, key="acq_bioemu", budget=args.budget)
    write_records(root / args.output_path, selected)
    logger.info("BioEmu selection complete: selected=%d elapsed=%.2fs", len(selected), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
