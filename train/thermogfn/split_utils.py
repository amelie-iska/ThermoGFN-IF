"""Dataset split discovery and index building."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .io_utils import read_json


@dataclass
class SplitPaths:
    root: Path
    train_dir: Path
    test_dir: Path
    metadata_dir: Path


def _candidate_split_roots(root: Path) -> list[Path]:
    candidates = [root]
    parts = root.parts

    # Common repo-local alias: data/rfd3_splits/... now lives under rfd3-data/rfd3_splits/...
    for i in range(len(parts) - 1):
        if parts[i] == "data" and parts[i + 1] == "rfd3_splits":
            candidates.append(Path(*parts[:i], "rfd3-data", *parts[i + 1 :]))
            break

    # Also accept roots provided as rfd3_splits/... and anchor them under rfd3-data/.
    for i, part in enumerate(parts):
        if part == "rfd3_splits" and (i == 0 or parts[i - 1] != "rfd3-data"):
            candidates.append(Path(*parts[:i], "rfd3-data", *parts[i:]))
            break

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def resolve_split_root(split_root: str | Path) -> Path:
    root = Path(split_root)
    candidates = _candidate_split_roots(root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Expected split path missing: {root} (tried: {tried})")


def discover_split(split_root: str | Path) -> SplitPaths:
    root = resolve_split_root(split_root)
    train_dir = root / "train"
    test_dir = root / "test"
    metadata_dir = root / "metadata"
    for path in (root, train_dir, test_dir, metadata_dir):
        if not path.exists():
            raise FileNotFoundError(f"Expected split path missing: {path}")
    return SplitPaths(root=root, train_dir=train_dir, test_dir=test_dir, metadata_dir=metadata_dir)


def _iter_specs(split_dir: Path) -> Iterator[Path]:
    for p in sorted(split_dir.glob("*.json")):
        if p.name == "split_summary.json":
            continue
        yield p


def iter_split_specs(paths: SplitPaths, split: str) -> Iterator[Path]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train/test")
    target = paths.train_dir if split == "train" else paths.test_dir
    yield from _iter_specs(target)


def parse_spec_file(path: Path) -> dict:
    obj = read_json(path)
    extra = obj.get("specification", {}).get("extra", {})
    sampled_contig = str(extra.get("sampled_contig", "0"))
    try:
        length = int(sampled_contig.split(",")[0]) if sampled_contig.isdigit() else int(sampled_contig)
    except Exception:  # noqa: BLE001
        length = int(extra.get("num_residues") or extra.get("num_residues_in") or 0)
    cif_override = extra.get("representative_structure_path") or obj.get("representative_structure_path")
    cif_gz_override = extra.get("representative_structure_gz_path") or obj.get("representative_structure_gz_path")
    row = {
        "spec_path": str(path),
        "stem": path.stem,
        "task_name": extra.get("task_name", "unknown"),
        "example_id": extra.get("example_id", "unknown"),
        "num_residues": int(extra.get("num_residues") or 0),
        "num_atoms": int(extra.get("num_atoms") or 0),
        "sampled_contig": sampled_contig,
        "length_estimate": length,
        "cif_path": str(cif_override or path.with_suffix(".cif")),
        "cif_gz_path": str(cif_gz_override or path.with_suffix(".cif.gz")),
    }
    # Optional metadata fields used by Kcat workflows and mixed-modality datasets.
    optional_keys = (
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
        "pair_id",
        "rhea_id",
        "uniprot_id",
        "reactant_complex_path",
        "product_complex_path",
        "reactant_protein_path",
        "product_protein_path",
        "representative_structure_path",
        "representative_structure_gz_path",
        "protein_chain_id",
        "ligand_chain_id",
        "sequence",
        "sequence_length",
        "pocket_positions",
    )
    for key in optional_keys:
        if key in extra:
            row[key] = extra[key]
    # Some split generators may store these fields at the top-level.
    for key in optional_keys:
        if key in obj and key not in row:
            row[key] = obj[key]
    return row


def build_design_index(paths: SplitPaths) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "test"):
        for spec in iter_split_specs(paths, split):
            row = parse_spec_file(spec)
            row["split"] = split
            rows.append(row)
    return rows


def load_split_summary(paths: SplitPaths) -> dict:
    summary_path = paths.metadata_dir / "split_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}
