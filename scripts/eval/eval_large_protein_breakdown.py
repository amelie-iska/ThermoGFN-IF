#!/usr/bin/env python3
"""Evaluate metrics stratified by prepared atom count threshold."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.io_utils import read_records
    from train.thermogfn.eval_utils import compute_design_metrics
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("eval.large_breakdown", level=args.log_level)
    rows = read_records(root / args.input_path)
    small = [r for r in rows if int(r.get("prepared_atom_count", 0)) <= args.threshold]
    large = [r for r in rows if int(r.get("prepared_atom_count", 0)) > args.threshold]
    logger.info("Large-protein breakdown: n_total=%d n_small=%d n_large=%d threshold=%d", len(rows), len(small), len(large), args.threshold)

    payload = {
        "threshold": args.threshold,
        "small": compute_design_metrics(small),
        "large": compute_design_metrics(large),
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote large-protein metrics to %s elapsed=%.2fs", out, time.perf_counter() - t0)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
