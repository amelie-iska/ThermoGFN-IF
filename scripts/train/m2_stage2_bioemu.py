#!/usr/bin/env python3
"""Method II stage 2: BioEmu refinement."""

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

    logger = configure_logging("train.m2_stage2", level=args.log_level)
    dispatch = root / args.dispatch_script
    cmd = [
        "python",
        str(dispatch),
        "--env-name",
        "bioemu",
        "--cmd",
        f"python {root / 'scripts/prep/oracles/bioemu_sample_and_features.py'} --candidate-path {root / args.input_path} --output-path {root / args.output_path}",
    ]
    logger.info("CMD: %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    logger.info("Stage2 complete rc=%d elapsed=%.2fs", rc, time.perf_counter() - t0)
    raise SystemExit(rc)
