#!/usr/bin/env python3
"""Generate unconditioned candidates using trained student checkpoint."""

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
    parser.add_argument("--seed-dataset", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, log_peak_vram, reset_peak_vram_tracking
    from train.thermogfn.method3_core import load_state, generate_student_candidates

    logger = configure_logging("infer.generate_unconditioned", level=args.log_level)
    reset_peak_vram_tracking()
    student = load_state(root / args.student_ckpt)
    seeds = [r for r in read_records(root / args.seed_dataset) if r.get("source") == "baseline"]
    logger.info("Generating candidates=%d from baseline seeds=%d", args.num_candidates, len(seeds))
    candidates = generate_student_candidates(
        student=student,
        seeds=seeds,
        pool_size=args.num_candidates,
        run_id=args.run_id,
        round_id=args.round_id,
        seed=args.seed,
        show_progress=True,
        progress_desc="infer:generate",
    )
    write_records(root / args.output_path, candidates)
    logger.info("Generation complete: wrote=%d elapsed=%.2fs", len(candidates), time.perf_counter() - t0)
    log_peak_vram(logger, label="infer.generate_unconditioned")
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
