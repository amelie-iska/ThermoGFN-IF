#!/usr/bin/env python3
"""Fuse KcatNet/GraphKcat channels into risk-adjusted Kcat rewards."""

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
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--w-kcatnet", type=float, default=0.65)
    parser.add_argument("--w-graphkcat", type=float, default=0.45)
    parser.add_argument("--w-agreement", type=float, default=0.15)
    parser.add_argument("--kappa-kcatnet", type=float, default=1.0)
    parser.add_argument("--kappa-graphkcat", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.kcat_reward import compute_fused_kcat_score
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("oracle.fuse_kcat", level=args.log_level)
    rows = read_records(root / args.candidate_path)
    logger.info("Kcat fusion start: n=%d", len(rows))

    out: list[dict] = []
    for rec in iter_progress(rows, total=len(rows), desc="fuse:kcat", no_progress=args.no_progress):
        out.append(
            compute_fused_kcat_score(
                rec,
                w_kcatnet=args.w_kcatnet,
                w_graphkcat=args.w_graphkcat,
                w_agreement=args.w_agreement,
                kappa_kcatnet=args.kappa_kcatnet,
                kappa_graphkcat=args.kappa_graphkcat,
            )
        )

    write_records(root / args.output_path, out)
    logger.info("Kcat fusion complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
