#!/usr/bin/env python3
"""
Build a robust small-molecule ligand library for RFdiffusion3 ligand-binder runs.

Outputs:
  - Per-ligand CIF structures under data/rfd3/ccd_ligands/structures/
  - Ligand list TSV compatible with scripts/run_rfd3_inference.sh
  - Metadata CSV for auditing/filtering
  - 3K allocation TSV with per-ligand design counts

This script uses Biotite's built-in CCD through AtomWorks, so it does not require
a local CCD mirror download.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from atomworks.io.utils.ccd import atom_array_from_ccd_code


DEFAULT_ALLOWED_ELEMENTS = (
    "H,C,N,O,S,P,F,CL,BR,I,B,SI,SE"
)


@dataclass
class Candidate:
    code: str
    name: str
    formula: str
    formula_weight: float
    formula_heavy_atoms: int
    formula_carbons: int
    formula_hetero_atoms: int
    weight_bin: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CCD ligand library for RFdiffusion3 ligand-binder generation."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path.",
    )
    parser.add_argument(
        "--target-ligands",
        type=int,
        default=300,
        help="Target number of ligands to include in the final library.",
    )
    parser.add_argument(
        "--total-designs",
        type=int,
        default=3000,
        help="Total target designs for allocation plan file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--min-formula-weight",
        type=float,
        default=150.0,
        help="Minimum formula weight filter.",
    )
    parser.add_argument(
        "--max-formula-weight",
        type=float,
        default=700.0,
        help="Maximum formula weight filter.",
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=12,
        help="Minimum heavy-atom count (post-CCD parse).",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=60,
        help="Maximum heavy-atom count (post-CCD parse).",
    )
    parser.add_argument(
        "--min-hetero-atoms",
        type=int,
        default=2,
        help="Minimum non-carbon heavy atoms (post-CCD parse).",
    )
    parser.add_argument(
        "--allowed-elements",
        type=str,
        default=DEFAULT_ALLOWED_ELEMENTS,
        help="Comma-separated allowed element symbols (case-insensitive).",
    )
    parser.add_argument(
        "--weight-bins",
        type=str,
        default="150,220,280,340,400,470,540,620,700",
        help="Comma-separated formula-weight bin edges.",
    )
    parser.add_argument(
        "--max-validation-attempts",
        type=int,
        default=6000,
        help="Maximum CCD validation attempts while selecting ligands.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="ccd_robust_ligands",
        help="Basename stem for output list/metadata/allocation files.",
    )
    return parser.parse_args()


def parse_formula(formula: str) -> Dict[str, int]:
    # Example: "C35 H42 F2 N2 O6"
    token_re = re.compile(r"([A-Z][a-z]?)(\d*)")
    counts: Dict[str, int] = {}
    for element, count_str in token_re.findall(formula):
        count = int(count_str) if count_str else 1
        counts[element.upper()] = counts.get(element.upper(), 0) + count
    return counts


def build_candidates(
    min_weight: float,
    max_weight: float,
    weight_edges: List[float],
) -> List[Candidate]:
    chem = struc.info.ccd.get_ccd()["chem_comp"]
    ids = chem["id"].as_array().astype(str)
    names = chem["name"].as_array().astype(str)
    types = chem["type"].as_array().astype(str)
    formulas = chem["formula"].as_array().astype(str)
    weights = chem["formula_weight"].as_array().astype(float)
    release_status = chem["pdbx_release_status"].as_array().astype(str)

    candidates: List[Candidate] = []
    for code, name, typ, formula, fw, status in zip(
        ids, names, types, formulas, weights, release_status, strict=False
    ):
        if status != "REL":
            continue
        if "non-polymer" not in typ.lower():
            continue
        if not np.isfinite(fw):
            continue
        if fw < min_weight or fw > max_weight:
            continue
        if formula == "?" or code == "?":
            continue

        composition = parse_formula(formula)
        carbons = composition.get("C", 0)
        heavy_formula = sum(v for k, v in composition.items() if k != "H")
        hetero_formula = heavy_formula - carbons
        if carbons < 6:
            continue
        if heavy_formula < 10:
            continue

        # Weight bin index for stratified sampling.
        bin_idx = None
        for i in range(len(weight_edges) - 1):
            lo = weight_edges[i]
            hi = weight_edges[i + 1]
            if lo <= fw <= hi:
                bin_idx = i
                break
        if bin_idx is None:
            continue

        candidates.append(
            Candidate(
                code=code,
                name=name.replace("\n", " ").strip(),
                formula=formula,
                formula_weight=float(fw),
                formula_heavy_atoms=int(heavy_formula),
                formula_carbons=int(carbons),
                formula_hetero_atoms=int(hetero_formula),
                weight_bin=int(bin_idx),
            )
        )

    return candidates


def split_allocation(total: int, n: int) -> List[int]:
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_atom_array_cif(atom_array, out_path: Path) -> None:
    if "chain_id" in atom_array.get_annotation_categories():
        atom_array.chain_id[:] = "L"
    if "res_id" in atom_array.get_annotation_categories():
        atom_array.res_id[:] = 1
    if "ins_code" in atom_array.get_annotation_categories():
        atom_array.ins_code[:] = ""

    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atom_array)
    ensure_parent(out_path)
    cif.write(str(out_path))


def normalize_elements(elements: Iterable[str]) -> List[str]:
    return [str(e).strip().upper() for e in elements]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.target_ligands < 1:
        raise ValueError("--target-ligands must be >= 1")
    if args.total_designs < 1:
        raise ValueError("--total-designs must be >= 1")

    allowed_elements = {x.strip().upper() for x in args.allowed_elements.split(",") if x.strip()}
    if not allowed_elements:
        raise ValueError("No allowed elements were parsed from --allowed-elements")

    weight_edges = [float(x.strip()) for x in args.weight_bins.split(",") if x.strip()]
    if len(weight_edges) < 2:
        raise ValueError("--weight-bins must contain at least two numeric edges")
    if sorted(weight_edges) != weight_edges:
        raise ValueError("--weight-bins must be sorted ascending")

    repo_root = args.repo_root.resolve()
    structures_dir = repo_root / "data" / "rfd3" / "ccd_ligands" / "structures"
    ligands_dir = repo_root / "data" / "rfd3" / "ligand_lists"
    stem = args.output_stem.strip()
    if not stem:
        raise ValueError("--output-stem must be non-empty")
    metadata_csv = ligands_dir / f"{stem}_metadata.csv"
    list_tsv = ligands_dir / f"{stem}.tsv"
    alloc_tsv = ligands_dir / f"{stem}_3k_allocation.tsv"

    print(f"[build] repo_root={repo_root}")
    print("[build] collecting CCD candidates from biotite chem_comp...")
    candidates = build_candidates(
        min_weight=args.min_formula_weight,
        max_weight=args.max_formula_weight,
        weight_edges=weight_edges,
    )
    print(f"[build] candidates after metadata filters: {len(candidates)}")

    by_bin: Dict[int, List[Candidate]] = {}
    for c in candidates:
        by_bin.setdefault(c.weight_bin, []).append(c)
    for bin_idx in by_bin:
        rng.shuffle(by_bin[bin_idx])

    bin_order = sorted(by_bin.keys())
    pool: List[Candidate] = []
    exhausted = set()
    while len(exhausted) < len(bin_order):
        for b in bin_order:
            items = by_bin.get(b, [])
            if items:
                pool.append(items.pop())
            else:
                exhausted.add(b)

    print(f"[build] stratified candidate pool size: {len(pool)}")
    selected_rows = []
    selected_codes = set()
    attempts = 0

    for cand in pool:
        if len(selected_rows) >= args.target_ligands:
            break
        if attempts >= args.max_validation_attempts:
            break
        attempts += 1

        if cand.code in selected_codes:
            continue

        try:
            aa = atom_array_from_ccd_code(cand.code)
        except Exception:
            continue

        if aa.array_length() == 0:
            continue
        if not np.isfinite(aa.coord).all():
            continue

        elements = normalize_elements(aa.element.tolist())
        element_set = set(elements)
        if not element_set.issubset(allowed_elements):
            continue
        if "C" not in element_set:
            continue

        heavy = sum(1 for e in elements if e != "H")
        carbons = sum(1 for e in elements if e == "C")
        hetero = heavy - carbons
        if heavy < args.min_heavy_atoms or heavy > args.max_heavy_atoms:
            continue
        if hetero < args.min_hetero_atoms:
            continue

        # reject degenerate coordinates
        span = float(np.max(aa.coord) - np.min(aa.coord))
        if span < 1e-3:
            continue

        out_cif = structures_dir / f"{cand.code}.cif"
        write_atom_array_cif(aa, out_cif)

        rel_cif = out_cif.relative_to(repo_root)
        selected_rows.append(
            {
                "name": f"ccd_{cand.code.lower()}",
                "input_path": f"./{rel_cif.as_posix()}",
                "ligand_code": cand.code,
                "formula_weight": f"{cand.formula_weight:.3f}",
                "heavy_atoms": str(heavy),
                "hetero_atoms": str(hetero),
                "elements": ",".join(sorted(element_set)),
                "formula": cand.formula,
                "ccd_name": cand.name,
                "weight_bin": str(cand.weight_bin),
            }
        )
        selected_codes.add(cand.code)

    if len(selected_rows) == 0:
        raise RuntimeError("No ligands selected. Relax filters and retry.")

    if len(selected_rows) < args.target_ligands:
        print(
            f"[build] warning: requested {args.target_ligands}, selected {len(selected_rows)} "
            f"after {attempts} validation attempts"
        )
    else:
        print(f"[build] selected target ligand count: {len(selected_rows)}")

    selected_rows.sort(key=lambda r: (float(r["formula_weight"]), r["ligand_code"]))

    # Write metadata CSV.
    ensure_parent(metadata_csv)
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "input_path",
                "ligand_code",
                "formula_weight",
                "heavy_atoms",
                "hetero_atoms",
                "elements",
                "formula",
                "ccd_name",
                "weight_bin",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    # Write ligand list TSV for run_rfd3_inference.sh.
    ensure_parent(list_tsv)
    with list_tsv.open("w", encoding="utf-8") as f:
        f.write("# name input_path ligand_code\n")
        for row in selected_rows:
            f.write(f"{row['name']} {row['input_path']} {row['ligand_code']}\n")

    # Write allocation TSV for total designs (defaults to 3000).
    allocations = split_allocation(args.total_designs, len(selected_rows))
    ensure_parent(alloc_tsv)
    with alloc_tsv.open("w", encoding="utf-8") as f:
        f.write("# name input_path ligand_code designs\n")
        for row, n_designs in zip(selected_rows, allocations, strict=False):
            f.write(
                f"{row['name']} {row['input_path']} {row['ligand_code']} {n_designs}\n"
            )

    per_ligand = args.total_designs / len(selected_rows)
    print(f"[build] wrote structures to: {structures_dir}")
    print(f"[build] wrote ligand list: {list_tsv}")
    print(f"[build] wrote metadata: {metadata_csv}")
    print(f"[build] wrote allocation: {alloc_tsv}")
    print(
        f"[build] allocation summary: total_designs={args.total_designs}, "
        f"ligands={len(selected_rows)}, avg_per_ligand={per_ligand:.2f}"
    )


if __name__ == "__main__":
    main()
