#!/usr/bin/env python3
"""Aggregate and compare metrics across ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.metric_files) != len(args.labels):
        print("--metric-files and --labels must have equal length", file=sys.stderr)
        return 2

    root = _repo_root()
    table = []
    for label, path in zip(args.labels, args.metric_files, strict=True):
        p = root / path
        metric = json.loads(p.read_text())
        table.append(
            {
                "label": label,
                "best_reward": metric.get("best_reward"),
                "top8_mean_reward": metric.get("top8_mean_reward"),
                "unique_fraction": metric.get("unique_fraction"),
            }
        )

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ablations": table}, indent=2, sort_keys=True))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
