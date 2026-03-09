#!/usr/bin/env python3
"""Materialize initial D0 dataset from baseline candidates."""

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
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--output-train", dest="output_train", required=True)
    parser.add_argument("--output-test", dest="output_test", required=True)
    parser.add_argument("--output-all", dest="output_all", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging
    from train.thermogfn.schemas import validate_records, ensure_unique_ids

    logger = configure_logging("prep.materialize_d0", level=args.log_level)
    rows = read_records(root / args.baselines)
    logger.info("Loaded baselines n=%d from %s", len(rows), root / args.baselines)
    ensure_unique_ids(rows, "candidate_id")
    summary = validate_records(rows, "candidate")
    if summary.invalid:
        for err in summary.errors[:10]:
            print(err, file=sys.stderr)
        return 3

    train_rows = [r for r in rows if r.get("split") == "train"]
    test_rows = [r for r in rows if r.get("split") == "test"]

    write_records(root / args.output_train, train_rows)
    write_records(root / args.output_test, test_rows)
    write_records(root / args.output_all, rows)
    logger.info(
        "Wrote D0 datasets: train=%d test=%d all=%d elapsed=%.2fs",
        len(train_rows),
        len(test_rows),
        len(rows),
        time.perf_counter() - t0,
    )

    print(root / args.output_train)
    print(root / args.output_test)
    print(root / args.output_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
