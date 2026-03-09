#!/usr/bin/env python3
"""Rescore candidates with fused reward and select top-k."""

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
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram, reset_peak_vram_tracking
    from train.thermogfn.reward import compute_fused_score, compute_reliability_gates

    logger = configure_logging("infer.rescore_select", level=args.log_level)
    reset_peak_vram_tracking()
    rows = read_records(root / args.input_path)
    logger.info("Rescoring candidates=%d top_k=%d", len(rows), args.top_k)
    scored = []
    for rec in iter_progress(rows, total=len(rows), desc="infer:rescore", no_progress=args.no_progress):
        rho_b, rho_u = compute_reliability_gates(rec)
        rec["rho_B"] = rho_b
        rec["rho_U"] = rho_u
        scored.append(compute_fused_score(rec))

    top = sorted(scored, key=lambda r: float(r.get("reward", 0.0)), reverse=True)[: args.top_k]
    write_records(root / args.output_path, top)
    logger.info("Selection complete: selected=%d elapsed=%.2fs", len(top), time.perf_counter() - t0)
    log_peak_vram(logger, label="infer.rescore_select")
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
