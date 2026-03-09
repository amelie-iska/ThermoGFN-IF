#!/usr/bin/env python3
"""Method II stage 3: UMA refinement + fused scoring."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dispatch-script", default="scripts/env/dispatch.py")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("train.m2_stage3", level=args.log_level)
    dispatch = root / args.dispatch_script
    tmp = Path(args.output_path).with_suffix(".tmp.jsonl")
    cmd1 = [
        "python",
        str(dispatch),
        "--env-name",
        "uma-qc",
        "--cmd",
        f"python {root / 'scripts/prep/oracles/uma_md_screen.py'} --candidate-path {root / args.input_path} --output-path {root / tmp}",
    ]
    cmd2 = ["python", str(root / "scripts/prep/oracles/fuse_scores.py"), "--candidate-path", str(root / tmp), "--output-path", str(root / args.output_path)]
    logger.info("CMD1: %s", " ".join(cmd1))
    rc = subprocess.run(cmd1, check=False).returncode
    if rc != 0:
        logger.error("Stage3 UMA command failed rc=%d", rc)
        raise SystemExit(rc)
    logger.info("CMD2: %s", " ".join(cmd2))
    rc2 = subprocess.run(cmd2, check=False).returncode
    logger.info("Stage3 complete rc=%d elapsed=%.2fs", rc2, time.perf_counter() - t0)
    raise SystemExit(rc2)
