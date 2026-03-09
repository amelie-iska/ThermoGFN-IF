#!/usr/bin/env python3
"""Target-conditioned generation with bounded retries using student proposals."""

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


def _target_deviation(rec: dict, target_reward: float, tol: float) -> float:
    reward = float(rec.get("reward", 0.0))
    return max(0.0, abs(reward - target_reward) - tol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-reward", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=16)
    parser.add_argument("--candidates-per-retry", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram, reset_peak_vram_tracking

    logger = configure_logging("infer.target_conditioned", level=args.log_level)
    reset_peak_vram_tracking()
    rows = read_records(root / args.candidate_path)
    logger.info(
        "Target-conditioned selection: rows=%d target_reward=%.4f tol=%.4f max_retries=%d candidates_per_retry=%d",
        len(rows),
        args.target_reward,
        args.tolerance,
        args.max_retries,
        args.candidates_per_retry,
    )
    best = None
    best_dev = float("inf")
    success = False
    retries_used = 0

    retry_iter = iter_progress(
        range(1, args.max_retries + 1),
        total=args.max_retries,
        desc="infer:retries",
        no_progress=args.no_progress,
    )
    for r in retry_iter:
        retries_used = r
        window = rows[(r - 1) * args.candidates_per_retry : r * args.candidates_per_retry]
        if not window:
            break
        for rec in window:
            dev = _target_deviation(rec, args.target_reward, args.tolerance)
            if dev < best_dev:
                best_dev = dev
                best = rec
            if dev == 0.0:
                success = True
                best = rec
                logger.info("Target hit at retry=%d candidate_id=%s", r, rec.get("candidate_id"))
                break
        if success:
            break

    payload = {
        "success": success,
        "retries": retries_used,
        "best_deviation": best_dev,
        "best_candidate": best,
        "target_reward": args.target_reward,
        "tolerance": args.tolerance,
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Target-conditioned selection complete: success=%s retries=%d elapsed=%.2fs", success, retries_used, time.perf_counter() - t0)
    log_peak_vram(logger, label="infer.target_conditioned")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
