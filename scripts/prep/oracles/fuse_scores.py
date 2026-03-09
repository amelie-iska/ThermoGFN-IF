#!/usr/bin/env python3
"""Fuse oracle channels into risk-adjusted rewards."""

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
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.reward import compute_fused_score, compute_reliability_gates

    logger = configure_logging("oracle.fuse", level=args.log_level)
    rows = read_records(root / args.candidate_path)
    logger.info("Fusing scores for n=%d", len(rows))
    out = []
    for rec in iter_progress(rows, total=len(rows), desc="fuse:scores", no_progress=args.no_progress):
        rho_b, rho_u = compute_reliability_gates(rec)
        rec["rho_B"] = rho_b
        rec["rho_U"] = rho_u
        out.append(compute_fused_score(rec))

    write_records(root / args.output_path, out)
    logger.info("Fusion complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
