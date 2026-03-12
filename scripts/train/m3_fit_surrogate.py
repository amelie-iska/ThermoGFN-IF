#!/usr/bin/env python3
"""Fit surrogate ensemble for Method III round."""

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
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--history-path", default=None)
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
    from train.thermogfn.method3_core import fit_surrogate_ensemble, save_state
    from train.thermogfn.checkpoint_utils import prune_round_checkpoints

    logger = configure_logging("train.m3_fit_surrogate", level=args.log_level)
    records = read_records(root / args.input_dr)
    logger.info("Fitting surrogate: n_records=%d ensemble_size=%d", len(records), args.ensemble_size)
    history_rows: list[dict] = []

    def _on_metric(row: dict) -> None:
        payload = {"round_id": args.round_id, **row}
        history_rows.append(payload)

    surrogate = fit_surrogate_ensemble(
        records,
        ensemble_size=args.ensemble_size,
        seed=args.seed + args.round_id,
        show_progress=not args.no_progress,
        progress_desc="surrogate:models",
        metrics_callback=_on_metric,
    )

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"surrogate_round_{args.round_id}.ckpt"
    save_state(ckpt, surrogate)
    prune_round_checkpoints(outdir, prefix="surrogate", max_keep=args.max_checkpoints, logger=logger)

    metrics = {
        "round_id": args.round_id,
        "n_records": len(records),
        "ensemble_size": args.ensemble_size,
        "n_models": len(surrogate.get("models", [])),
    }
    write_json(outdir / "surrogate_metrics.json", metrics)
    if args.history_path:
        write_jsonl(root / args.history_path, history_rows)
    else:
        write_json(outdir / "surrogate_history.json", history_rows)
    logger.info("Surrogate checkpoint written: %s elapsed=%.2fs", ckpt, time.perf_counter() - t0)
    print(ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
