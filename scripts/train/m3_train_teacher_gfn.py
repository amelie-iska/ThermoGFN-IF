#!/usr/bin/env python3
"""Train teacher policy approximation for Method III."""

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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--gamma-off", type=float, default=0.5)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records
    from train.thermogfn.progress import configure_logging
    from train.thermogfn.method3_core import train_teacher_policy, reconstructed_trajectory_stats, save_state
    from train.thermogfn.checkpoint_utils import prune_round_checkpoints

    logger = configure_logging("train.m3_teacher", level=args.log_level)
    records = read_records(root / args.input_dr)
    logger.info("Training teacher policy approximation: n_records=%d", len(records))
    teacher = train_teacher_policy(
        records,
        seed=args.seed + args.round_id,
        show_progress=not args.no_progress,
        progress_desc="teacher:records",
    )

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"teacher_round_{args.round_id}.ckpt"
    save_state(ckpt, teacher)
    prune_round_checkpoints(outdir, prefix="teacher", max_keep=args.max_checkpoints, logger=logger)

    metrics = {
        "round_id": args.round_id,
        "n_records": len(records),
        "steps": args.steps,
        "gamma_off": args.gamma_off,
        "trajectory_stats": reconstructed_trajectory_stats(records),
        "k_probs": teacher.get("k_probs", {}),
    }
    (outdir / "teacher_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("Teacher checkpoint written: %s elapsed=%.2fs", ckpt, time.perf_counter() - t0)
    print(ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
