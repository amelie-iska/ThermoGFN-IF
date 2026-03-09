#!/usr/bin/env python3
"""Validate monomer split structure and metadata integrity."""

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
    parser.add_argument("--split-root", default="data/rfd3_splits/unconditional_monomer_protrek35m")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.split_utils import discover_split, iter_split_specs, load_split_summary
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("prep.validate_split", level=args.log_level)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Validating split root: %s", root / args.split_root)

    report: dict = {
        "split_root": args.split_root,
        "ok": True,
        "errors": [],
        "counts": {},
        "summary": {},
    }

    try:
        paths = discover_split(root / args.split_root)
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["errors"].append(str(exc))
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        logger.exception("Split discovery failed")
        print(out_path)
        return 2

    for split in ("train", "test"):
        specs = list(iter_split_specs(paths, split))
        report["counts"][split] = len(specs)
        logger.info("Checking %s specs: n=%d", split, len(specs))
        missing = 0
        for spec in iter_progress(specs, total=len(specs), desc=f"validate:{split}", no_progress=args.no_progress):
            cif = spec.with_suffix(".cif")
            cif_gz = spec.with_suffix(".cif.gz")
            if not cif.exists() and not cif_gz.exists():
                missing += 1
        report["counts"][f"{split}_missing_structures"] = missing

    summary = load_split_summary(paths)
    report["summary"] = summary

    if report["counts"].get("train", 0) == 0:
        report["ok"] = False
        report["errors"].append("No train specs found")
    if report["counts"].get("test", 0) == 0:
        report["ok"] = False
        report["errors"].append("No test specs found")

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info(
        "Validation complete: ok=%s train=%d test=%d missing_train=%d missing_test=%d elapsed=%.2fs",
        report["ok"],
        report["counts"].get("train", 0),
        report["counts"].get("test", 0),
        report["counts"].get("train_missing_structures", 0),
        report["counts"].get("test_missing_structures", 0),
        time.perf_counter() - t0,
    )
    print(out_path)
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
