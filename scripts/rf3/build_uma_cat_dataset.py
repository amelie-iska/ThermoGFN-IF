#!/usr/bin/env python3
"""Build baseline catalytic Method III datasets from RF3 reactant/product outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


def _candidate_id(pair_id: str, sequence: str) -> str:
    payload = f"uma-cat:{pair_id}:{sequence}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _iter_shard_records(shard_root: Path, state: str):
    state_root = shard_root / "shards" / state
    for shard_path in sorted(state_root.glob("*.json")):
        obj = json.loads(shard_path.read_text())
        if isinstance(obj, dict):
            rows = obj.get("inputs") or obj.get("examples") or [obj]
        else:
            rows = obj
        for rec in rows:
            yield rec


def _collect_inputs(prepared_root: Path) -> dict[str, dict]:
    per_pair: dict[str, dict] = {}
    for state in ("reactant", "product"):
        for rec in _iter_shard_records(prepared_root, state):
            meta = rec.get("metadata", {})
            pair_id = str(meta.get("pair_id") or "").strip()
            if not pair_id:
                continue
            slot = per_pair.setdefault(pair_id, {})
            slot[state] = rec
    return per_pair


def _pair_id_from_job_name(job_name: str, state: str) -> str | None:
    suffix = f"__{state}"
    if not job_name.endswith(suffix):
        return None
    return job_name[: -len(suffix)]


def _collect_model_paths(state_root: Path, state: str) -> dict[str, str]:
    best: dict[str, Path] = {}
    for cif_path in sorted(state_root.glob("**/*_model.cif")):
        job_name = cif_path.parent.name
        pair_id = _pair_id_from_job_name(job_name, state)
        if not pair_id:
            continue
        incumbent = best.get(pair_id)
        if incumbent is None or len(cif_path.parts) < len(incumbent.parts):
            best[pair_id] = cif_path.resolve()
    return {k: str(v) for k, v in best.items()}


def _get_component_seq(rec: dict) -> str:
    comps = rec.get("components") or []
    if not comps:
        return ""
    return str(comps[0].get("seq") or "").strip()


def _get_component_smiles(rec: dict) -> str:
    comps = rec.get("components") or []
    if len(comps) < 2:
        return ""
    return str(comps[1].get("smiles") or "").strip()


def _decomposition(protein_chain_id: str, ligand_chain_id: str | None) -> dict:
    return {
        "type": "protein_ligand",
        "complex": ["protein", "ligand"],
        "components": {
            "protein": {"chain_ids": [protein_chain_id]},
            "ligand": {"chain_ids": [ligand_chain_id] if ligand_chain_id else []},
        },
    }


def _count_atoms_in_structure(path: str | Path) -> int:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    count = 0
    with opener(p, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM ") or line.startswith("HETATM "):
                count += 1
    if count <= 0:
        raise RuntimeError(f"No atoms found in structure: {p}")
    return count


def _rows_from_split_root(split_root: Path, split_name: str) -> list[dict]:
    from train.thermogfn.split_utils import discover_split, iter_split_specs

    paths = discover_split(split_root)
    rows: list[dict] = []
    for spec_path in iter_split_specs(paths, split_name):
        rows.append(json.loads(spec_path.read_text()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--reactant-root", default="")
    parser.add_argument("--product-root", default="")
    parser.add_argument("--split-root", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-id", default="uma_cat_bootstrap")
    parser.add_argument("--split", default="train")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import write_records
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("rf3.build_uma_cat_dataset", level=args.log_level)

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = root / output_path

    rows: list[dict] = []
    if args.split_root:
        split_root = Path(args.split_root)
        if not split_root.is_absolute():
            split_root = root / split_root
        specs = _rows_from_split_root(split_root.resolve(), args.split)
        if args.limit > 0:
            specs = specs[: int(args.limit)]
        logger.info("Collected RF3 split specs: split_root=%s split=%s rows=%d", split_root, args.split, len(specs))
        for spec in specs:
            pair_id = str(spec.get("pair_id") or "").strip()
            seq = str(spec.get("sequence") or "").strip()
            reactant_model = str(spec.get("reactant_complex_path") or "").strip()
            product_model = str(spec.get("product_complex_path") or "").strip()
            protein_chain_id = str(spec.get("protein_chain_id") or "A")
            ligand_chain_id = spec.get("ligand_chain_id")
            if not pair_id or not seq or not reactant_model or not product_model:
                logger.warning("Skipping invalid split spec missing core fields: %s", pair_id or "<unknown>")
                continue
            try:
                atom_count = int(spec.get("prepared_atom_count") or _count_atoms_in_structure(reactant_model))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping pair_id=%s due to atom-count failure: %s", pair_id, exc)
                continue
            row = {
                "candidate_id": _candidate_id(pair_id, seq),
                "run_id": args.run_id,
                "round_id": int(args.round_id),
                "task_type": "ligand",
                "backbone_id": pair_id,
                "seed_id": pair_id,
                "sequence": seq,
                "mutations": [],
                "K": 0,
                "prepared_atom_count": int(atom_count),
                "eligibility": {"bioemu": False, "uma_whole": True, "uma_local": False},
                "source": "baseline",
                "schema_version": "v1",
                "split": args.split,
                "pair_id": pair_id,
                "rhea_id": spec.get("rhea_id"),
                "uniprot_id": spec.get("uniprot_id"),
                "substrate_smiles": spec.get("substrate_smiles") or spec.get("ligand_smiles"),
                "product_smiles": spec.get("product_smiles"),
                "ligand_smiles": spec.get("ligand_smiles") or spec.get("substrate_smiles"),
                "reactant_complex_path": reactant_model,
                "product_complex_path": product_model,
                "reactant_protein_path": spec.get("reactant_protein_path") or reactant_model,
                "product_protein_path": spec.get("product_protein_path") or product_model,
                "cif_path": spec.get("representative_structure_path") or reactant_model,
                "complex_path": reactant_model,
                "protein_chain_id": protein_chain_id,
                "ligand_chain_id": ligand_chain_id,
                "pocket_positions": [int(x) for x in (spec.get("pocket_positions") or [])],
                "sequence_length": int(spec.get("sequence_length") or len(seq)),
                "novelty": 0.0,
                "pack_unc": 0.0,
                "decomposition": _decomposition(protein_chain_id, ligand_chain_id),
            }
            rows.append(row)
    else:
        if not args.prepared_input_root or not args.reactant_root or not args.product_root:
            raise RuntimeError(
                "--prepared-input-root, --reactant-root, and --product-root are required unless --split-root is used."
            )
        prepared_root = Path(args.prepared_input_root)
        if not prepared_root.is_absolute():
            prepared_root = root / prepared_root
        reactant_root = Path(args.reactant_root)
        if not reactant_root.is_absolute():
            reactant_root = root / reactant_root
        product_root = Path(args.product_root)
        if not product_root.is_absolute():
            product_root = root / product_root

        input_map = _collect_inputs(prepared_root.resolve())
        reactant_models = _collect_model_paths(reactant_root.resolve(), "reactant")
        product_models = _collect_model_paths(product_root.resolve(), "product")
        logger.info(
            "Collected RF3 records: prepared_pairs=%d reactant_models=%d product_models=%d",
            len(input_map),
            len(reactant_models),
            len(product_models),
        )
        pair_ids = sorted(set(input_map) & set(reactant_models) & set(product_models))
        if args.limit > 0:
            pair_ids = pair_ids[: int(args.limit)]

        for pair_id in pair_ids:
            reactant_input = input_map[pair_id].get("reactant")
            product_input = input_map[pair_id].get("product")
            if not reactant_input or not product_input:
                continue

            meta = reactant_input.get("metadata", {})
            seq = _get_component_seq(reactant_input)
            if not seq:
                logger.warning("Skipping pair_id=%s due to empty sequence", pair_id)
                continue
            protein_chain_id = str(meta.get("protein_chain_id") or "A")
            ligand_chain_id = meta.get("ligand_chain_id")
            pocket_positions = [int(x) for x in (meta.get("pocket_positions") or [])]
            reactant_model = reactant_models[pair_id]
            product_model = product_models[pair_id]

            try:
                atom_count = _count_atoms_in_structure(reactant_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping pair_id=%s due to atom-count failure: %s", pair_id, exc)
                continue

            row = {
                "candidate_id": _candidate_id(pair_id, seq),
                "run_id": args.run_id,
                "round_id": int(args.round_id),
                "task_type": "ligand",
                "backbone_id": pair_id,
                "seed_id": pair_id,
                "sequence": seq,
                "mutations": [],
                "K": 0,
                "prepared_atom_count": int(atom_count),
                "eligibility": {"bioemu": False, "uma_whole": True, "uma_local": False},
                "source": "baseline",
                "schema_version": "v1",
                "split": args.split,
                "pair_id": pair_id,
                "rhea_id": meta.get("rhea_id"),
                "uniprot_id": meta.get("uniprot_id"),
                "substrate_smiles": meta.get("ligand_smiles") or _get_component_smiles(reactant_input),
                "product_smiles": product_input.get("metadata", {}).get("ligand_smiles") or _get_component_smiles(product_input),
                "ligand_smiles": meta.get("ligand_smiles") or _get_component_smiles(reactant_input),
                "reactant_complex_path": reactant_model,
                "product_complex_path": product_model,
                "reactant_protein_path": reactant_model,
                "product_protein_path": product_model,
                "cif_path": reactant_model,
                "complex_path": reactant_model,
                "protein_chain_id": protein_chain_id,
                "ligand_chain_id": ligand_chain_id,
                "pocket_positions": pocket_positions,
                "sequence_length": len(seq),
                "novelty": 0.0,
                "pack_unc": 0.0,
                "decomposition": _decomposition(protein_chain_id, ligand_chain_id),
            }
            rows.append(row)

    write_records(output_path, rows)
    logger.info("Built UMA-cat dataset: rows=%d output=%s elapsed=%.2fs", len(rows), output_path, time.perf_counter() - t0)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
