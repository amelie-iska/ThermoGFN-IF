#!/usr/bin/env python3
"""Export a multi-MODEL PDB from a fresh UMA steered MD trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-summary-json", required=True, help="Path to UMA artifact summary.json")
    parser.add_argument("--output-pdb", required=True, help="Output multi-MODEL PDB path")
    parser.add_argument("--output-summary-json", default="", help="Optional output path for fresh sMD summary")
    parser.add_argument("--reverse", type=int, default=0, help="Run reverse product->reactant pull instead")
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calculator-workers", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.05)
    parser.add_argument("--friction-ps-inv", type=float, default=2.0)
    parser.add_argument("--images", type=int, default=96)
    parser.add_argument("--steps-per-image", type=int, default=48)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--k-steer-eva2", type=float, default=0.01)
    parser.add_argument("--k-global-eva2", type=float, default=0.03)
    parser.add_argument("--k-local-eva2", type=float, default=0.15)
    parser.add_argument("--k-anchor-eva2", type=float, default=0.01)
    parser.add_argument("--ca-network-sequential-k-eva2", type=float, default=12.0)
    parser.add_argument("--ca-network-contact-k-eva2", type=float, default=0.50)
    parser.add_argument("--ca-network-contact-cutoff-a", type=float, default=8.0)
    parser.add_argument("--anchor-stride", type=int, default=6)
    parser.add_argument("--force-clip-eva", type=float, default=0.35)
    parser.add_argument("--prepare-hydrogens", type=int, default=1)
    parser.add_argument("--add-first-shell-waters", type=int, default=1)
    parser.add_argument("--preparation-ph", type=float, default=7.4)
    parser.add_argument("--max-first-shell-waters", type=int, default=12)
    parser.add_argument("--water-shell-distance-a", type=float, default=2.8)
    parser.add_argument("--water-clash-distance-a", type=float, default=2.1)
    parser.add_argument("--water-bridge-distance-min-a", type=float, default=4.2)
    parser.add_argument("--water-bridge-distance-max-a", type=float, default=6.6)
    parser.add_argument("--export-replica-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))
    fairchem_src = root / "models" / "fairchem" / "src"
    if fairchem_src.exists():
        sys.path.insert(0, str(fairchem_src))

    from train.thermogfn.uma_cat_runtime import run_steered_uma_dynamics

    summary_path = _resolve_path(root, args.artifact_summary_json)
    output_pdb = _resolve_path(root, args.output_pdb)
    summary = json.loads(summary_path.read_text())
    record = summary.get("record", {})
    if not record:
        raise ValueError(f"summary lacks record payload: {summary_path}")

    reactant_path = record.get("reactant_complex_packed_path") or record.get("reactant_complex_path")
    product_path = record.get("product_complex_packed_path") or record.get("product_complex_path")
    if not reactant_path or not product_path:
        raise ValueError("record is missing reactant/product complex paths")

    if bool(args.reverse):
        start_path = product_path
        end_path = reactant_path
    else:
        start_path = reactant_path
        end_path = product_path

    smd = run_steered_uma_dynamics(
        reactant_complex_path=_resolve_path(root, start_path),
        product_complex_path=_resolve_path(root, end_path),
        protein_chain_id=str(record["protein_chain_id"]),
        ligand_chain_id=record.get("ligand_chain_id"),
        pocket_positions=[int(x) for x in (record.get("pocket_positions") or [])],
        temperature_k=float(args.temperature_k),
        timestep_fs=float(args.timestep_fs),
        friction_ps_inv=float(args.friction_ps_inv),
        images=int(args.images),
        steps_per_image=int(args.steps_per_image),
        replicas=int(args.replicas),
        k_steer_eva2=float(args.k_steer_eva2),
        k_global_eva2=float(args.k_global_eva2),
        k_local_eva2=float(args.k_local_eva2),
        k_anchor_eva2=float(args.k_anchor_eva2),
        ca_network_sequential_k_eva2=float(args.ca_network_sequential_k_eva2),
        ca_network_contact_k_eva2=float(args.ca_network_contact_k_eva2),
        ca_network_contact_cutoff_a=float(args.ca_network_contact_cutoff_a),
        model_name=args.model_name,
        device=args.device,
        calculator_workers=int(args.calculator_workers),
        anchor_stride=int(args.anchor_stride),
        force_clip_eva=float(args.force_clip_eva),
        prepare_hydrogens=bool(args.prepare_hydrogens),
        add_first_shell_waters=bool(args.add_first_shell_waters),
        preparation_ph=float(args.preparation_ph),
        max_first_shell_waters=int(args.max_first_shell_waters),
        water_shell_distance_a=float(args.water_shell_distance_a),
        water_clash_distance_a=float(args.water_clash_distance_a),
        water_bridge_distance_min_a=float(args.water_bridge_distance_min_a),
        water_bridge_distance_max_a=float(args.water_bridge_distance_max_a),
        seed=int(args.seed),
        export_multimodel_pdb_path=output_pdb,
        export_replica_index=int(args.export_replica_index),
    )

    out_summary = args.output_summary_json.strip()
    if out_summary:
        out_summary_path = _resolve_path(root, out_summary)
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        out_summary_path.write_text(json.dumps(smd, indent=2, sort_keys=True))

    print(output_pdb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
