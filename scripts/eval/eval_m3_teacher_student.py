#!/usr/bin/env python3
"""Evaluate teacher-student divergence and top-level agreement."""

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
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.method3_core import load_state, teacher_student_kl
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("eval.teacher_student", level=args.log_level)
    logger.info("Evaluating teacher-student KL: teacher=%s student=%s", root / args.teacher_ckpt, root / args.student_ckpt)
    teacher = load_state(root / args.teacher_ckpt)
    student = load_state(root / args.student_ckpt)
    kl = teacher_student_kl(teacher, student)

    payload = {
        "teacher_student_kl": kl,
        "teacher_k_probs": teacher.get("k_probs", {}),
        "student_k_probs": student.get("k_probs", {}),
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote teacher-student eval to %s (KL=%.6f) elapsed=%.2fs", out, kl, time.perf_counter() - t0)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
