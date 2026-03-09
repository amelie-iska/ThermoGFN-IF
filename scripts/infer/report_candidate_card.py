#!/usr/bin/env python3
"""Generate a readable candidate card from selected candidate records."""

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
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.io_utils import read_records
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram, reset_peak_vram_tracking

    logger = configure_logging("infer.report_card", level=args.log_level)
    reset_peak_vram_tracking()
    rows = read_records(root / args.input_path)
    logger.info("Searching candidate_id=%s in n=%d records", args.candidate_id, len(rows))
    target = None
    for rec in iter_progress(rows, total=len(rows), desc="infer:find-candidate", no_progress=args.no_progress):
        if rec.get("candidate_id") == args.candidate_id:
            target = rec
            break
    if target is None:
        print(f"candidate_id not found: {args.candidate_id}", file=sys.stderr)
        return 2

    card = {
        "candidate_id": target.get("candidate_id"),
        "backbone_id": target.get("backbone_id"),
        "task_type": target.get("task_type"),
        "K": target.get("K"),
        "mutations": target.get("mutations"),
        "reward": target.get("reward"),
        "scores": {
            "spurs": target.get("spurs_mean"),
            "bioemu": target.get("bioemu_calibrated"),
            "uma": target.get("uma_calibrated"),
        },
        "uncertainty": {
            "spurs_std": target.get("spurs_std"),
            "bioemu_std": target.get("bioemu_std"),
            "uma_std": target.get("uma_std"),
        },
        "sequence": target.get("sequence"),
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True))
    logger.info("Candidate card written to %s elapsed=%.2fs", out, time.perf_counter() - t0)
    log_peak_vram(logger, label="infer.report_card")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
