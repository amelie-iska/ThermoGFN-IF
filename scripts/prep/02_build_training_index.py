#!/usr/bin/env python3
"""Build a unified training index from one or more split spec roots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

OVERLAY_FIELDS = (
    "substrate_smiles",
    "product_smiles",
    "ligand_smiles",
    "smiles",
    "Smiles",
    "substrate",
    "product",
    "organism",
    "Organism",
    "ph",
    "pH",
    "temp",
    "Temp",
    "chain_id",
    "protein_path",
    "ligand_path",
    "complex_path",
)


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _infer_task_type(split_name: str) -> str:
    low = split_name.lower()
    if "ligand" in low:
        return "ligand"
    if any(tok in low for tok in ("ppi", "dimer", "multimer", "complex", "oligomer")):
        return "ppi"
    return "monomer"


def _overlay_matches(row: dict, overlay: dict, root: Path) -> bool:
    overlay_split = overlay.get("split")
    if overlay_split is not None and str(row.get("split", "")) != str(overlay_split):
        return False

    overlay_split_root = overlay.get("split_root")
    if overlay_split_root is not None and str(row.get("split_root", "")) != str(overlay_split_root):
        return False

    overlay_spec = overlay.get("spec_path")
    if overlay_spec:
        row_spec = str(row.get("spec_path", ""))
        if row_spec == str(overlay_spec):
            return True
        try:
            row_abs = str((root / row_spec).resolve())
            overlay_abs = str((Path(str(overlay_spec)) if Path(str(overlay_spec)).is_absolute() else (root / str(overlay_spec))).resolve())
            return row_abs == overlay_abs
        except Exception:  # noqa: BLE001
            return False

    overlay_stem = overlay.get("stem", overlay.get("backbone_id"))
    if overlay_stem is not None:
        return str(row.get("stem", "")) == str(overlay_stem)

    overlay_example = overlay.get("example_id")
    if overlay_example is not None:
        return str(row.get("example_id", "")) == str(overlay_example)

    raise ValueError(
        "metadata overlay row must include one of: spec_path, stem, backbone_id, example_id"
    )


def _apply_metadata_overlays(rows: list[dict], overlay_paths: list[str], root: Path, logger) -> None:
    from train.thermogfn.io_utils import read_records

    for overlay_path in overlay_paths:
        overlay_abs = Path(overlay_path)
        if not overlay_abs.is_absolute():
            overlay_abs = root / overlay_abs
        if not overlay_abs.exists():
            raise FileNotFoundError(f"metadata overlay not found: {overlay_abs}")

        overlay_rows = read_records(overlay_abs)
        applied = 0
        for i, overlay in enumerate(overlay_rows):
            updates = {k: v for k, v in overlay.items() if k in OVERLAY_FIELDS and v is not None}
            if not updates:
                continue
            targets = [row for row in rows if _overlay_matches(row, overlay, root)]
            if not targets:
                raise RuntimeError(
                    f"metadata overlay row {i} in {overlay_abs} did not match any index rows"
                )
            for row in targets:
                row.update(updates)
            applied += len(targets)
        logger.info(
            "Applied metadata overlay %s: overlay_rows=%d row_updates=%d",
            overlay_abs,
            len(overlay_rows),
            applied,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        action="append",
        dest="split_roots",
        default=None,
        help="Relative path to a split root. May be passed multiple times.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--metadata-overlay",
        action="append",
        dest="metadata_overlays",
        default=None,
        help="Optional JSONL/CSV overlay keyed by spec_path/stem/backbone_id/example_id.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.split_utils import discover_split, build_design_index

    logger = configure_logging("prep.build_index", level=args.log_level)
    split_roots = args.split_roots or ["data/rfd3_splits/unconditional_monomer_protrek35m"]
    rows: list[dict] = []
    logger.info("Building index from %d split roots", len(split_roots))
    for split_root in iter_progress(split_roots, total=len(split_roots), desc="index:roots", no_progress=args.no_progress):
        split_abs = root / split_root
        try:
            paths = discover_split(split_abs)
        except FileNotFoundError:
            if args.allow_missing:
                logger.warning("Skipping missing split root (allow-missing): %s", split_abs)
                continue
            raise
        chunk = build_design_index(paths)
        logger.info("Loaded split root %s (resolved=%s): n=%d", split_root, paths.root, len(chunk))
        split_name = Path(split_root).name
        task_type = _infer_task_type(split_name)
        for row in chunk:
            row["split_root"] = split_root
            row["split_name"] = split_name
            row["task_type"] = task_type
        rows.extend(chunk)

    if args.metadata_overlays:
        _apply_metadata_overlays(rows, args.metadata_overlays, root, logger)

    if not rows:
        raise RuntimeError("No rows found across requested split roots")

    write_records(root / args.output, rows)
    logger.info("Wrote index rows=%d to %s (elapsed=%.2fs)", len(rows), root / args.output, time.perf_counter() - t0)
    print(root / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
