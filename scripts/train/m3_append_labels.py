#!/usr/bin/env python3
"""Append newly labeled candidates to D_{r+1}."""

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
    parser.add_argument("--input-dr", required=True)
    parser.add_argument("--labeled-path", required=True)
    parser.add_argument("--output-dr-next", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.schemas import ensure_unique_ids, validate_records

    logger = configure_logging("train.append_labels", level=args.log_level)
    dr = read_records(root / args.input_dr)
    labeled = read_records(root / args.labeled_path)
    logger.info("Merging labels: prev=%d new=%d", len(dr), len(labeled))

    merged = {r["candidate_id"]: r for r in dr}
    for r in iter_progress(labeled, total=len(labeled), desc="merge:labels", no_progress=args.no_progress):
        merged[r["candidate_id"]] = r
    out = list(merged.values())

    ensure_unique_ids(out, "candidate_id")
    summary = validate_records(out, "candidate")
    if summary.invalid:
        for err in summary.errors[:10]:
            print(err, file=sys.stderr)
        return 3

    write_records(root / args.output_dr_next, out)
    payload = {
        "n_prev": len(dr),
        "n_new_labeled": len(labeled),
        "n_next": len(out),
    }
    s = root / args.summary_path
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Append complete: next=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_dr_next)
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
