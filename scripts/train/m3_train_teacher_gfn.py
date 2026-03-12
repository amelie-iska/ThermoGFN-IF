#!/usr/bin/env python3
"""Train the Method III trajectory-balance teacher."""

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
    parser.add_argument("--surrogate-ckpt", default=None)
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records
    from train.thermogfn.io_utils import write_json, write_jsonl
    from train.thermogfn.progress import configure_logging
    from train.thermogfn.method3_core import (
        load_state,
        reconstructed_trajectory_stats,
        save_state,
        train_teacher_policy,
    )
    from train.thermogfn.checkpoint_utils import prune_round_checkpoints

    logger = configure_logging("train.m3_teacher", level=args.log_level)
    records = read_records(root / args.input_dr)
    surrogate = load_state(root / args.surrogate_ckpt) if args.surrogate_ckpt else None
    logger.info("Training trajectory-balance teacher: n_records=%d surrogate=%s", len(records), bool(surrogate))
    history_rows: list[dict] = []

    def _on_metric(row: dict) -> None:
        payload = {"round_id": args.round_id, **row}
        history_rows.append(payload)
        logger.info(
            "teacher step=%d loss=%.6f off=%.6f on=%.6f reg=%.6f lr=%.5f grad=%.4f",
            int(payload.get("step", 0)),
            float(payload.get("loss", 0.0)),
            float(payload.get("off_loss", 0.0)),
            float(payload.get("on_loss", 0.0)),
            float(payload.get("reg_loss", 0.0)),
            float(payload.get("lr", 0.0)),
            float(payload.get("grad_norm_post_clip", 0.0)),
        )

    teacher = train_teacher_policy(
        records,
        seed=args.seed + args.round_id,
        steps=args.steps,
        gamma_off=args.gamma_off,
        surrogate=surrogate,
        show_progress=not args.no_progress,
        progress_desc="teacher:tb",
        metrics_every=args.metrics_every,
        metrics_callback=_on_metric,
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
        "surrogate_attached": bool(surrogate),
        "trajectory_stats": reconstructed_trajectory_stats(records),
        "k_probs": teacher.get("k_probs", {}),
        "teacher_mode": teacher.get("teacher_mode"),
        "implementation_note": teacher.get("implementation_note"),
        "is_true_gflownet": bool(teacher.get("is_true_gflownet", False)),
    }
    write_json(outdir / "teacher_metrics.json", metrics)
    if args.history_path:
        write_jsonl(root / args.history_path, history_rows)
    else:
        write_json(outdir / "teacher_history.json", history_rows)
    logger.info("Teacher checkpoint written: %s elapsed=%.2fs", ckpt, time.perf_counter() - t0)
    print(ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
