#!/usr/bin/env python3
"""Validate that Kcat-mode records carry substrate metadata required by Kcat oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


SUBSTRATE_KEYS = ("substrate_smiles", "Smiles", "smiles", "ligand_smiles")


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _has_substrate(rec: dict) -> bool:
    for key in SUBSTRATE_KEYS:
        val = rec.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                if item is not None and str(item).strip():
                    return True
            continue
        if str(val).strip():
            return True
    return False


def _row_label(rec: dict) -> str:
    for key in ("candidate_id", "backbone_id", "stem", "example_id", "spec_path"):
        val = rec.get(key)
        if val:
            return f"{key}={val}"
    return "<unknown>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage", default="kcat_input")
    parser.add_argument("--output", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--sample-errors", type=int, default=10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_json
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("prep.validate_kcat_metadata", level=args.log_level)
    rows = read_records(root / args.input)
    missing = [rec for rec in rows if not _has_substrate(rec)]
    ok = not missing

    report = {
        "ok": ok,
        "stage": args.stage,
        "input": str(root / args.input),
        "n_rows": len(rows),
        "n_missing_substrate": len(missing),
        "substrate_keys": list(SUBSTRATE_KEYS),
        "missing_examples": [_row_label(rec) for rec in missing[: args.sample_errors]],
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    }

    if args.output:
        out_path = root / args.output
        write_json(out_path, report)
        print(out_path)

    if ok:
        logger.info(
            "Kcat metadata validation complete: ok=True stage=%s rows=%d elapsed=%.2fs",
            args.stage,
            len(rows),
            time.perf_counter() - t0,
        )
        return 0

    logger.error(
        "Kcat metadata validation failed: stage=%s missing=%d/%d",
        args.stage,
        len(missing),
        len(rows),
    )
    for label in report["missing_examples"]:
        print(f"missing substrate metadata: {label}", file=sys.stderr)
    print(
        "Kcat training requires one of substrate_smiles/Smiles/smiles/ligand_smiles in the dataset "
        "or a metadata overlay passed at index-build time.",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
