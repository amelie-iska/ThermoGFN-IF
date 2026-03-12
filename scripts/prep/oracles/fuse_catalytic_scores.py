#!/usr/bin/env python3
"""Fuse UMA catalytic and GraphKcat channels into Method III rewards."""

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
    parser.add_argument("--graphkcat-path", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--w-uma-cat", type=float, default=0.75)
    parser.add_argument("--w-graphkcat", type=float, default=0.45)
    parser.add_argument("--w-agreement", type=float, default=0.20)
    parser.add_argument("--kappa-uma-cat", type=float, default=1.0)
    parser.add_argument("--kappa-graphkcat", type=float, default=1.0)
    parser.add_argument("--graph-field", default="graphkcat_log_kcat")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.uma_cat_reward import compute_fused_catalytic_score

    logger = configure_logging("oracle.fuse_catalytic", level=args.log_level)
    base_rows = read_records(root / args.candidate_path)
    graph_rows = read_records(root / args.graphkcat_path) if args.graphkcat_path else []
    graph_by_id = {str(r.get("candidate_id")): r for r in graph_rows}
    logger.info("Catalytic fusion start: base=%d graph=%d", len(base_rows), len(graph_rows))

    out: list[dict] = []
    for rec in iter_progress(base_rows, total=len(base_rows), desc="fuse:catalytic", no_progress=args.no_progress):
        row = dict(rec)
        extra = graph_by_id.get(str(row.get("candidate_id")))
        if extra:
            for key, value in extra.items():
                if key == "candidate_id":
                    continue
                row[key] = value
        out.append(
            compute_fused_catalytic_score(
                row,
                w_uma_cat=args.w_uma_cat,
                w_graphkcat=args.w_graphkcat,
                w_agreement=args.w_agreement,
                kappa_uma_cat=args.kappa_uma_cat,
                kappa_graphkcat=args.kappa_graphkcat,
                graph_field=args.graph_field,
            )
        )

    write_records(root / args.output_path, out)
    logger.info("Catalytic fusion complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
