#!/usr/bin/env python3
"""Distill one-shot student from teacher policy."""

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
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--input-dr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15000)
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
    from train.thermogfn.method3_core import load_state, distill_student_from_teacher, save_state, teacher_student_kl
    from train.thermogfn.checkpoint_utils import prune_round_checkpoints

    logger = configure_logging("train.m3_distill", level=args.log_level)
    teacher = load_state(root / args.teacher_ckpt)
    records = read_records(root / args.input_dr)
    logger.info("Distilling student from teacher: n_records=%d", len(records))
    student = distill_student_from_teacher(
        teacher,
        records,
        seed=args.seed + args.round_id,
        show_progress=not args.no_progress,
        progress_desc="student:records",
    )

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"student_round_{args.round_id}.ckpt"
    save_state(ckpt, student)
    prune_round_checkpoints(outdir, prefix="student", max_keep=args.max_checkpoints, logger=logger)

    metrics = {
        "round_id": args.round_id,
        "steps": args.steps,
        "teacher_student_kl": teacher_student_kl(teacher, student),
        "student_k_probs": student.get("k_probs", {}),
    }
    (outdir / "student_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("Student checkpoint written: %s elapsed=%.2fs", ckpt, time.perf_counter() - t0)
    print(ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
