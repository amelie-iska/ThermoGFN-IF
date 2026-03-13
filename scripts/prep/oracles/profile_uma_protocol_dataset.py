#!/usr/bin/env python3
"""Profile UMA endpoint protocol eligibility across a catalytic dataset without running MD."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _float_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _float_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _sequence_length(row: dict) -> int:
    return int(len(str(row.get("sequence") or "")))


def _stratified_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)
    ordered = sorted(rows, key=_sequence_length)
    if sample_size == 1:
        return [ordered[len(ordered) // 2]]
    anchors = np.linspace(0, len(ordered) - 1, sample_size, dtype=int).tolist()
    chosen: list[dict] = []
    used: set[int] = set()
    for idx in anchors:
        if idx not in used:
            chosen.append(ordered[idx])
            used.add(idx)
    if len(chosen) < sample_size:
        rng = random.Random(int(seed))
        remaining = [i for i in range(len(ordered)) if i not in used]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(chosen) >= sample_size:
                break
            chosen.append(ordered[idx])
    return chosen[:sample_size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/uma_cat_m3_default.yaml")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-mode", choices=["random", "stratified_length"], default="stratified_length")
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calculator-workers", type=int, default=1)
    parser.add_argument("--prepare-hydrogens", type=int, default=None)
    parser.add_argument("--add-first-shell-waters", type=int, default=None)
    parser.add_argument("--preparation-ph", type=float, default=None)
    parser.add_argument("--max-first-shell-waters", type=int, default=None)
    parser.add_argument("--water-shell-distance-a", type=float, default=None)
    parser.add_argument("--water-clash-distance-a", type=float, default=None)
    parser.add_argument("--water-bridge-distance-min-a", type=float, default=None)
    parser.add_argument("--water-bridge-distance-max-a", type=float, default=None)
    parser.add_argument("--relax-prepared-steps", type=int, default=None)
    parser.add_argument("--relax-prepared-fmax-eva", type=float, default=None)
    parser.add_argument("--protocol-max-reactive-bonds", type=int, default=None)
    parser.add_argument("--protocol-max-reactive-atoms", type=int, default=None)
    parser.add_argument("--protocol-max-reactive-fraction", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))
    fairchem_src = root / "models" / "fairchem" / "src"
    if fairchem_src.exists():
        sys.path.insert(0, str(fairchem_src))

    from train.thermogfn.config_utils import cfg_get, load_yaml_config
    from train.thermogfn.io_utils import read_records, write_json
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.uma_cat_runtime import (
        analyze_endpoint_protocol,
        load_structure,
        prepare_structure_for_uma,
        relax_structure_under_uma,
    )

    logger = configure_logging("profile.uma_protocol", level="INFO")
    cfg_path = _resolve_path(root, args.config)
    cfg = load_yaml_config(cfg_path)
    args.model_name = str(args.model_name or cfg_get(cfg, "oracles.uma_cat.model_name", "uma-s-1p1"))
    args.device = str(args.device or cfg_get(cfg, "oracles.uma_cat.device", "cuda:0"))
    args.calculator_workers = int(
        args.calculator_workers if args.calculator_workers is not None else cfg_get(cfg, "oracles.uma_cat.calculator_workers", 1)
    )
    args.prepare_hydrogens = int(
        args.prepare_hydrogens if args.prepare_hydrogens is not None else cfg_get(cfg, "oracles.uma_cat.preparation.hydrogens", 1)
    )
    args.add_first_shell_waters = int(
        args.add_first_shell_waters if args.add_first_shell_waters is not None else cfg_get(cfg, "oracles.uma_cat.preparation.first_shell_waters", 1)
    )
    args.preparation_ph = float(
        args.preparation_ph if args.preparation_ph is not None else cfg_get(cfg, "oracles.uma_cat.preparation.ph", 7.4)
    )
    args.max_first_shell_waters = int(
        args.max_first_shell_waters
        if args.max_first_shell_waters is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.max_first_shell_waters", 12)
    )
    args.water_shell_distance_a = float(
        args.water_shell_distance_a
        if args.water_shell_distance_a is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.water_shell_distance_a", 2.8)
    )
    args.water_clash_distance_a = float(
        args.water_clash_distance_a
        if args.water_clash_distance_a is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.water_clash_distance_a", 2.1)
    )
    args.water_bridge_distance_min_a = float(
        args.water_bridge_distance_min_a
        if args.water_bridge_distance_min_a is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.water_bridge_distance_min_a", 4.2)
    )
    args.water_bridge_distance_max_a = float(
        args.water_bridge_distance_max_a
        if args.water_bridge_distance_max_a is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.water_bridge_distance_max_a", 6.6)
    )
    args.relax_prepared_steps = int(
        args.relax_prepared_steps
        if args.relax_prepared_steps is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.relax_steps", 25)
    )
    args.relax_prepared_fmax_eva = float(
        args.relax_prepared_fmax_eva
        if args.relax_prepared_fmax_eva is not None
        else cfg_get(cfg, "oracles.uma_cat.preparation.relax_fmax_eva", 0.20)
    )
    args.protocol_max_reactive_bonds = int(
        args.protocol_max_reactive_bonds
        if args.protocol_max_reactive_bonds is not None
        else cfg_get(cfg, "oracles.uma_cat.protocol.max_reactive_bonds", 8)
    )
    args.protocol_max_reactive_atoms = int(
        args.protocol_max_reactive_atoms
        if args.protocol_max_reactive_atoms is not None
        else cfg_get(cfg, "oracles.uma_cat.protocol.max_reactive_atoms", 12)
    )
    args.protocol_max_reactive_fraction = float(
        args.protocol_max_reactive_fraction
        if args.protocol_max_reactive_fraction is not None
        else cfg_get(cfg, "oracles.uma_cat.protocol.max_reactive_fraction", 0.35)
    )

    dataset_path = _resolve_path(root, args.dataset_path)
    output_path = _resolve_path(root, args.output)
    rows = read_records(dataset_path)
    valid = [
        row
        for row in rows
        if row.get("reactant_complex_path")
        and row.get("product_complex_path")
        and row.get("protein_chain_id")
        and row.get("pocket_positions")
    ]
    if int(args.sample_size) > 0 and len(valid) > int(args.sample_size):
        if str(args.sample_mode) == "stratified_length":
            valid = _stratified_sample(valid, int(args.sample_size), int(args.seed))
        else:
            rng = random.Random(int(args.seed))
            valid = rng.sample(valid, int(args.sample_size))

    reports: list[dict] = []
    for row in iter_progress(valid, total=len(valid), desc="uma:protocol:profile", no_progress=args.no_progress):
        protein_chain_id = str(row["protein_chain_id"])
        ligand_chain_id = row.get("ligand_chain_id")
        pocket_positions = [int(x) for x in (row.get("pocket_positions") or [])]
        reactant_prepared = None
        product_prepared = None
        reactant_path = _resolve_path(root, row["reactant_complex_path"])
        product_path = _resolve_path(root, row["product_complex_path"])
        try:
            reactant_prepared = prepare_structure_for_uma(
                reactant_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(args.preparation_ph),
                add_first_shell_waters=bool(args.add_first_shell_waters),
                max_first_shell_waters=int(args.max_first_shell_waters),
                water_shell_distance_a=float(args.water_shell_distance_a),
                water_clash_distance_a=float(args.water_clash_distance_a),
                water_bridge_distance_min_a=float(args.water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(args.water_bridge_distance_max_a),
            ) if bool(args.prepare_hydrogens) else None
            product_prepared = prepare_structure_for_uma(
                product_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(args.preparation_ph),
                add_first_shell_waters=bool(args.add_first_shell_waters),
                max_first_shell_waters=int(args.max_first_shell_waters),
                water_shell_distance_a=float(args.water_shell_distance_a),
                water_clash_distance_a=float(args.water_clash_distance_a),
                water_bridge_distance_min_a=float(args.water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(args.water_bridge_distance_max_a),
            ) if bool(args.prepare_hydrogens) else None
            if reactant_prepared is not None and int(args.relax_prepared_steps) > 0:
                reactant_prepared = relax_structure_under_uma(
                    structure=reactant_prepared,
                    model_name=args.model_name,
                    device=args.device,
                    calculator_workers=int(args.calculator_workers),
                    steps=int(args.relax_prepared_steps),
                    fmax_eva=float(args.relax_prepared_fmax_eva),
                )
            if product_prepared is not None and int(args.relax_prepared_steps) > 0:
                product_prepared = relax_structure_under_uma(
                    structure=product_prepared,
                    model_name=args.model_name,
                    device=args.device,
                    calculator_workers=int(args.calculator_workers),
                    steps=int(args.relax_prepared_steps),
                    fmax_eva=float(args.relax_prepared_fmax_eva),
                )
            reactant_struct = reactant_prepared if reactant_prepared is not None else load_structure(reactant_path)
            product_struct = product_prepared if product_prepared is not None else load_structure(product_path)
            bundle = analyze_endpoint_protocol(
                reactant=reactant_struct,
                product=product_struct,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                max_reactive_bonds=int(args.protocol_max_reactive_bonds),
                max_reactive_atoms=int(args.protocol_max_reactive_atoms),
                max_reactive_fraction=float(args.protocol_max_reactive_fraction),
            )
            mapping = bundle["mapping"]
            graph_model = bundle["graph_model"]
            protocol_meta = bundle["protocol_meta"]
            prepared = reactant_prepared
            report = {
                "candidate_id": row.get("candidate_id"),
                "status": "ok",
                "sequence_length": _sequence_length(row),
                "prepared_atom_count": int(prepared.count_atoms()) if prepared is not None else None,
                "prepared_first_shell_water_molecules": int(
                    sum(
                        1
                        for res_name, atom_name in zip(prepared.residue_names, prepared.atom_names, strict=False)
                        if res_name == "HOH" and atom_name == "O"
                    )
                ) if prepared is not None else 0,
                "alignment_mode": str(mapping.get("alignment_mode", "unknown")),
                "alignment_atom_count": int(mapping.get("alignment_atom_count", 0)),
                "shared_atom_count": int(mapping.get("shared_atom_count", 0)),
                "exact_name_matches": int(mapping.get("exact_name_matches", 0)),
                "protocol_mode": str(protocol_meta.get("protocol_mode", "unknown")),
                "protocol_reason": str(protocol_meta.get("protocol_reason", "")),
                "reactive_barrier_valid": bool(protocol_meta.get("reactive_barrier_valid", False)),
                "pmf_eligible": bool(protocol_meta.get("pmf_eligible", False)),
                "steering_mode": str(bundle.get("steering_mode", "unknown")),
                "steering_confident": bool(bundle["ligand_restraints"].get("steering_confident", False)),
                "broken_bond_count": int(graph_model.get("broken_bond_count", 0)),
                "formed_bond_count": int(graph_model.get("formed_bond_count", 0)),
                "reactive_atom_count": int(graph_model.get("reactive_atom_count", 0)),
                "reactive_fraction": float(graph_model.get("reactive_fraction", 0.0)),
                "reaction_component_pair_count": int(len(graph_model.get("component_pair_indices", ()))),
                "reaction_aux_pair_count": int(len(graph_model.get("aux_pairs", ()))),
                "quality_reason": str(graph_model.get("quality_reason", "")),
            }
        except Exception as exc:  # noqa: BLE001
            report = {
                "candidate_id": row.get("candidate_id"),
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)

    ok_reports = [r for r in reports if r.get("status") == "ok"]
    protocol_counts = Counter(str(r.get("protocol_mode", "missing")) for r in ok_reports)
    reason_counts = Counter(str(r.get("protocol_reason", "")) for r in ok_reports)
    steering_counts = Counter(str(r.get("steering_mode", "missing")) for r in ok_reports)
    reactive_bonds = [float(r.get("broken_bond_count", 0)) + float(r.get("formed_bond_count", 0)) for r in ok_reports]
    reactive_atoms = [float(r.get("reactive_atom_count", 0)) for r in ok_reports]
    reactive_fractions = [float(r.get("reactive_fraction", 0.0)) for r in ok_reports]
    atom_counts = [float(r.get("prepared_atom_count", 0)) for r in ok_reports if r.get("prepared_atom_count") is not None]
    water_counts = [float(r.get("prepared_first_shell_water_molecules", 0)) for r in ok_reports]
    n_reactive_protocol = sum(1 for r in ok_reports if bool(r.get("reactive_barrier_valid", False)))
    n_pmf_eligible = sum(1 for r in ok_reports if bool(r.get("pmf_eligible", False)))
    payload = {
        "config": vars(args),
        "n_total": len(reports),
        "n_ok": len(ok_reports),
        "ok_fraction": float(len(ok_reports) / len(reports)) if reports else 0.0,
        "protocol_mode_counts": dict(protocol_counts),
        "protocol_reason_counts": dict(reason_counts),
        "steering_mode_counts": dict(steering_counts),
        "reactive_protocol_fraction": float(n_reactive_protocol / len(ok_reports)) if ok_reports else 0.0,
        "pmf_eligible_fraction": float(n_pmf_eligible / len(ok_reports)) if ok_reports else 0.0,
        "aggregate": {
            "reactive_bond_count_mean": _float_mean(reactive_bonds),
            "reactive_bond_count_median": _float_median(reactive_bonds),
            "reactive_atom_count_mean": _float_mean(reactive_atoms),
            "reactive_fraction_mean": _float_mean(reactive_fractions),
            "prepared_atom_count_mean": _float_mean(atom_counts),
            "prepared_water_count_mean": _float_mean(water_counts),
        },
        "reports": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    logger.info(
        "UMA protocol profile complete: ok=%d/%d reactive_fraction=%.3f pmf_eligible_fraction=%.3f wrote=%s",
        len(ok_reports),
        len(reports),
        float(payload["reactive_protocol_fraction"]),
        float(payload["pmf_eligible_fraction"]),
        output_path,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
