#!/usr/bin/env python3
"""Compute periodic overfitting gap and simple calibration diagnostics."""

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
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--test-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.io_utils import read_json
    from train.thermogfn.eval_utils import overfit_gap
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("eval.calibration", level=args.log_level)
    train_m = read_json(root / args.train_metrics)
    test_m = read_json(root / args.test_metrics)

    payload = {
        "train": train_m,
        "test": test_m,
        "overfit_gap_top8_mean_reward": overfit_gap(train_m, test_m, key="top8_mean_reward"),
        "overfit_gap_best_reward": overfit_gap(train_m, test_m, key="best_reward"),
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote calibration metrics to %s elapsed=%.2fs", out, time.perf_counter() - t0)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
