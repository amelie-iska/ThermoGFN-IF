#!/usr/bin/env python3
"""Validate RF3 ProTrek pair split integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def _iter_specs(split_dir: Path):
    for path in sorted(split_dir.glob("*.json")):
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    split_root = Path(args.split_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    train_dir = split_root / "train"
    test_dir = split_root / "test"
    metadata_dir = split_root / "metadata"
    report = {
        "split_root": str(split_root),
        "ok": True,
        "errors": [],
        "counts": {},
        "summary": {},
        "elapsed_s": 0.0,
    }

    for required in (split_root, train_dir, test_dir, metadata_dir):
        if not required.exists():
            report["ok"] = False
            report["errors"].append(f"Missing required path: {required}")
    if not report["ok"]:
        report["elapsed_s"] = time.perf_counter() - t0
        output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(output)
        return 2

    try:
        summary = json.loads((metadata_dir / "split_summary.json").read_text())
        report["summary"] = summary
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["errors"].append(f"Failed to load split_summary.json: {exc}")
        summary = {}

    for split_name, split_dir in (("train", train_dir), ("test", test_dir)):
        specs = list(_iter_specs(split_dir))
        report["counts"][split_name] = len(specs)
        for spec_path in specs:
            try:
                spec = json.loads(spec_path.read_text())
            except Exception as exc:  # noqa: BLE001
                report["ok"] = False
                report["errors"].append(f"Invalid JSON {spec_path}: {exc}")
                continue
            for key in ("pair_id", "sequence", "reactant_complex_path", "product_complex_path", "protein_chain_id"):
                if not spec.get(key):
                    report["ok"] = False
                    report["errors"].append(f"Missing key '{key}' in {spec_path}")
            for key in ("reactant_complex_path", "product_complex_path", "representative_structure_path"):
                value = spec.get(key)
                if value and not Path(value).exists():
                    report["ok"] = False
                    report["errors"].append(f"Referenced path missing for {spec_path}: {value}")

    if summary:
        if summary.get("n_train") != report["counts"].get("train"):
            report["ok"] = False
            report["errors"].append("Summary n_train does not match actual count")
        if summary.get("n_test") != report["counts"].get("test"):
            report["ok"] = False
            report["errors"].append("Summary n_test does not match actual count")

    report["elapsed_s"] = time.perf_counter() - t0
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output)
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
