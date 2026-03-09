#!/usr/bin/env python3
"""Generate one-shot student candidate pool."""

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
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--input-dr", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.method3_core import load_state, generate_student_candidates
    from train.thermogfn.schemas import validate_records, ensure_unique_ids

    logger = configure_logging("train.m3_generate_pool", level=args.log_level)
    student = load_state(root / args.student_ckpt)
    rows = read_records(root / args.input_dr)
    seeds = [r for r in rows if r.get("split") == args.split and r.get("source") in {"baseline", "teacher", "student"}]
    if not seeds:
        seeds = [r for r in rows if r.get("source") == "baseline"]
    logger.info("Generating student pool: seeds=%d split=%s pool_size=%d", len(seeds), args.split, args.pool_size)
    pool = generate_student_candidates(
        student=student,
        seeds=seeds,
        pool_size=args.pool_size,
        run_id=args.run_id,
        round_id=args.round_id,
        seed=args.seed + args.round_id,
        show_progress=not args.no_progress,
        progress_desc="student:pool",
    )

    # propagate backbone file pointers where available
    seed_by_backbone = {r["backbone_id"]: r for r in seeds}
    for rec in iter_progress(pool, total=len(pool), desc="student:annotate", no_progress=args.no_progress):
        b = seed_by_backbone.get(rec["backbone_id"])
        if b:
            for key in ("cif_path", "spec_path", "split"):
                if key in b:
                    rec[key] = b[key]

    ensure_unique_ids(pool, "candidate_id")
    summary = validate_records(pool, "candidate")
    if summary.invalid:
        for err in summary.errors[:10]:
            print(err, file=sys.stderr)
        return 3

    write_records(root / args.output_path, pool)
    logger.info("Student pool complete: wrote=%d elapsed=%.2fs", len(pool), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
