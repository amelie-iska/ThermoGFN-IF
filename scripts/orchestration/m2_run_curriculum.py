#!/usr/bin/env python3
"""Run Method II sequential curriculum (SPURS -> BioEmu -> UMA)."""

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


def _run(cmd: list[str], logger) -> tuple[int, float]:
    logger.info("CMD: %s", " ".join(cmd))
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=False).returncode
    return rc, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("orchestrate.m2_curriculum", level=args.log_level)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    s1 = out / "m2_stage1_spurs.jsonl"
    s2 = out / "m2_stage2_bioemu.jsonl"
    s3 = out / "m2_stage3_uma_fused.jsonl"

    rc, dt = _run(["python", str(root / "scripts/train/m2_stage1_spurs.py"), "--input-path", str(root / args.input_path), "--output-path", str(s1)], logger)
    logger.info("Stage1 rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc
    rc, dt = _run(["python", str(root / "scripts/train/m2_stage2_bioemu.py"), "--input-path", str(s1), "--output-path", str(s2)], logger)
    logger.info("Stage2 rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc
    rc, dt = _run(["python", str(root / "scripts/train/m2_stage3_uma.py"), "--input-path", str(s2), "--output-path", str(s3)], logger)
    logger.info("Stage3 rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc

    logger.info("Method II complete output=%s elapsed=%.2fs", s3, time.perf_counter() - wall_t0)
    print(s3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
