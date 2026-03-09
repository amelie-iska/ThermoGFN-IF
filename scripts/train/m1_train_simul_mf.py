#!/usr/bin/env python3
"""Method I (Simul-MF) simplified training loop wrapper."""

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
    parser.add_argument("--dispatch-script", default="scripts/env/dispatch.py")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("train.m1_simul_mf", level=args.log_level)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    dispatch = root / args.dispatch_script
    spurs = out / "m1_spurs.jsonl"
    bio = out / "m1_bioemu.jsonl"
    uma = out / "m1_uma.jsonl"
    fused = out / "m1_fused.jsonl"

    rc, dt = _run(
        [
            "python",
            str(dispatch),
            "--env-name",
            "spurs",
            "--cmd",
            f"python {root / 'scripts/prep/oracles/spurs_score_single.py'} --candidate-path {root / args.input_path} --output-path {spurs}",
        ]
    )
    logger.info("Stage spurs rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc
    rc, dt = _run(
        [
            "python",
            str(dispatch),
            "--env-name",
            "bioemu",
            "--cmd",
            f"python {root / 'scripts/prep/oracles/bioemu_sample_and_features.py'} --candidate-path {spurs} --output-path {bio}",
        ]
    )
    logger.info("Stage bioemu rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc
    rc, dt = _run(
        [
            "python",
            str(dispatch),
            "--env-name",
            "uma-qc",
            "--cmd",
            f"python {root / 'scripts/prep/oracles/uma_md_screen.py'} --candidate-path {bio} --output-path {uma}",
        ]
    )
    logger.info("Stage uma rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc
    rc, dt = _run(["python", str(root / "scripts/prep/oracles/fuse_scores.py"), "--candidate-path", str(uma), "--output-path", str(fused)], logger)
    logger.info("Stage fuse rc=%d duration=%.2fs", rc, dt)
    if rc != 0:
        return rc

    logger.info("Method I complete output=%s elapsed=%.2fs", fused, time.perf_counter() - wall_t0)
    print(fused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
