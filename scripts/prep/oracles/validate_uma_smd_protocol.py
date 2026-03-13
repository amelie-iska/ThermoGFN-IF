#!/usr/bin/env python3
"""Validate the UMA sMD protocol on a representative catalytic dataset sample."""

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
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-mode", choices=["random", "stratified_length"], default="stratified_length")
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calculator-workers", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=None)
    parser.add_argument("--timestep-fs", type=float, default=None)
    parser.add_argument("--friction-ps-inv", type=float, default=None)
    parser.add_argument("--images", type=int, default=None)
    parser.add_argument("--steps-per-image", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--k-steer-eva2", type=float, default=None)
    parser.add_argument("--k-global-eva2", type=float, default=None)
    parser.add_argument("--k-local-eva2", type=float, default=None)
    parser.add_argument("--k-anchor-eva2", type=float, default=0.0)
    parser.add_argument("--ca-network-sequential-k-eva2", type=float, default=None)
    parser.add_argument("--ca-network-contact-k-eva2", type=float, default=None)
    parser.add_argument("--ca-network-contact-cutoff-a", type=float, default=None)
    parser.add_argument("--force-clip-eva", type=float, default=None)
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
    parser.add_argument("--threshold-final-product-rmsd-a", type=float, default=None)
    parser.add_argument("--threshold-max-product-rmsd-a", type=float, default=None)
    parser.add_argument("--threshold-max-pocket-rmsd-a", type=float, default=None)
    parser.add_argument("--threshold-max-backbone-rmsd-a", type=float, default=None)
    parser.add_argument("--threshold-max-ca-network-rms-a", type=float, default=None)
    parser.add_argument("--threshold-max-close-contacts", type=int, default=None)
    parser.add_argument("--threshold-max-excess-bond-count", type=int, default=None)
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
        prepare_structure_for_uma,
        relax_structure_under_uma,
        run_steered_uma_dynamics,
    )

    logger = configure_logging("validate.uma_smd", level="INFO")
    cfg_path = _resolve_path(root, args.config)
    cfg = load_yaml_config(cfg_path)
    args.model_name = str(args.model_name or cfg_get(cfg, "oracles.uma_cat.model_name", "uma-s-1p1"))
    args.device = str(args.device or cfg_get(cfg, "oracles.uma_cat.device", "cuda:0"))
    args.calculator_workers = int(
        args.calculator_workers if args.calculator_workers is not None else cfg_get(cfg, "oracles.uma_cat.calculator_workers", 1)
    )
    args.temperature_k = float(
        args.temperature_k if args.temperature_k is not None else cfg_get(cfg, "oracles.uma_cat.smd.temperature_k", 300.0)
    )
    args.timestep_fs = float(
        args.timestep_fs if args.timestep_fs is not None else cfg_get(cfg, "oracles.uma_cat.smd.timestep_fs", 0.05)
    )
    args.friction_ps_inv = float(
        args.friction_ps_inv if args.friction_ps_inv is not None else cfg_get(cfg, "oracles.uma_cat.smd.friction_ps_inv", 1.0)
    )
    args.images = int(args.images if args.images is not None else cfg_get(cfg, "oracles.uma_cat.smd.images", 96))
    args.steps_per_image = int(
        args.steps_per_image if args.steps_per_image is not None else cfg_get(cfg, "oracles.uma_cat.smd.steps_per_image", 32)
    )
    args.replicas = int(args.replicas if args.replicas is not None else cfg_get(cfg, "oracles.uma_cat.smd.replicas", 2))
    args.k_steer_eva2 = float(
        args.k_steer_eva2 if args.k_steer_eva2 is not None else cfg_get(cfg, "oracles.uma_cat.smd.k_steer_eva2", 0.01)
    )
    args.k_global_eva2 = float(
        args.k_global_eva2 if args.k_global_eva2 is not None else cfg_get(cfg, "oracles.uma_cat.smd.k_global_eva2", 0.015)
    )
    args.k_local_eva2 = float(
        args.k_local_eva2 if args.k_local_eva2 is not None else cfg_get(cfg, "oracles.uma_cat.smd.k_local_eva2", 0.10)
    )
    args.ca_network_sequential_k_eva2 = float(
        args.ca_network_sequential_k_eva2
        if args.ca_network_sequential_k_eva2 is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.ca_network.sequential_k_eva2", 8.0)
    )
    args.ca_network_contact_k_eva2 = float(
        args.ca_network_contact_k_eva2
        if args.ca_network_contact_k_eva2 is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.ca_network.contact_k_eva2", 0.30)
    )
    args.ca_network_contact_cutoff_a = float(
        args.ca_network_contact_cutoff_a
        if args.ca_network_contact_cutoff_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.ca_network.contact_cutoff_a", 8.0)
    )
    args.force_clip_eva = float(
        args.force_clip_eva if args.force_clip_eva is not None else cfg_get(cfg, "oracles.uma_cat.smd.force_clip_eva", 0.50)
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
    args.threshold_final_product_rmsd_a = float(
        args.threshold_final_product_rmsd_a
        if args.threshold_final_product_rmsd_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_final_product_rmsd_a", 1.75)
    )
    args.threshold_max_product_rmsd_a = float(
        args.threshold_max_product_rmsd_a
        if args.threshold_max_product_rmsd_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_product_rmsd_a", 5.0)
    )
    args.threshold_max_pocket_rmsd_a = float(
        args.threshold_max_pocket_rmsd_a
        if args.threshold_max_pocket_rmsd_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_pocket_rmsd_a", 4.0)
    )
    args.threshold_max_backbone_rmsd_a = float(
        args.threshold_max_backbone_rmsd_a
        if args.threshold_max_backbone_rmsd_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_backbone_rmsd_a", 3.0)
    )
    args.threshold_max_ca_network_rms_a = float(
        args.threshold_max_ca_network_rms_a
        if args.threshold_max_ca_network_rms_a is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_ca_network_rms_a", 0.75)
    )
    args.threshold_max_close_contacts = int(
        args.threshold_max_close_contacts
        if args.threshold_max_close_contacts is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_close_contacts", 0)
    )
    args.threshold_max_excess_bond_count = int(
        args.threshold_max_excess_bond_count
        if args.threshold_max_excess_bond_count is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.quality.max_max_excess_bond_count", 0)
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
    if len(valid) > int(args.sample_size):
        if str(args.sample_mode) == "stratified_length":
            valid = _stratified_sample(valid, int(args.sample_size), int(args.seed))
        else:
            rng = random.Random(int(args.seed))
            valid = rng.sample(valid, int(args.sample_size))

    reports: list[dict] = []
    for idx, row in enumerate(
        iter_progress(valid, total=len(valid), desc="uma:smd:validate", no_progress=args.no_progress)
    ):
        protein_chain_id = str(row["protein_chain_id"])
        ligand_chain_id = row.get("ligand_chain_id")
        pocket_positions = [int(x) for x in (row.get("pocket_positions") or [])]
        reactant_prepared = None
        product_prepared = None
        try:
            reactant_prepared = prepare_structure_for_uma(
                _resolve_path(root, row["reactant_complex_path"]),
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
                _resolve_path(root, row["product_complex_path"]),
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
            smd = run_steered_uma_dynamics(
                reactant_complex_path=_resolve_path(root, row["reactant_complex_path"]),
                product_complex_path=_resolve_path(root, row["product_complex_path"]),
                reactant_structure=reactant_prepared,
                product_structure=product_prepared,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
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
                force_clip_eva=float(args.force_clip_eva),
                prepare_hydrogens=bool(args.prepare_hydrogens),
                add_first_shell_waters=bool(args.add_first_shell_waters),
                preparation_ph=float(args.preparation_ph),
                max_first_shell_waters=int(args.max_first_shell_waters),
                water_shell_distance_a=float(args.water_shell_distance_a),
                water_clash_distance_a=float(args.water_clash_distance_a),
                water_bridge_distance_min_a=float(args.water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(args.water_bridge_distance_max_a),
                max_reactive_bonds=int(args.protocol_max_reactive_bonds),
                max_reactive_atoms=int(args.protocol_max_reactive_atoms),
                max_reactive_fraction=float(args.protocol_max_reactive_fraction),
                seed=int(args.seed) + idx,
            )
            rep = dict(smd.get("replica_summaries", [{}])[0] or {})
            report = {
                "candidate_id": row.get("candidate_id"),
                "status": "ok",
                "mapping": smd.get("mapping", {}),
                "replica": rep,
                "protocol_mode": str((smd.get("mapping", {}) or {}).get("protocol_mode", "unknown")),
                "protocol_reason": str((smd.get("mapping", {}) or {}).get("protocol_reason", "")),
                "reactive_barrier_valid": bool((smd.get("mapping", {}) or {}).get("reactive_barrier_valid", False)),
                "pmf_eligible": bool((smd.get("mapping", {}) or {}).get("pmf_eligible", False)),
                "sequence_length": _sequence_length(row),
                "prepared_atom_count": int(reactant_prepared.count_atoms()) if reactant_prepared is not None else None,
                "prepared_first_shell_water_molecules": int(
                    sum(
                        1
                        for res_name, atom_name in zip(reactant_prepared.residue_names, reactant_prepared.atom_names, strict=False)
                        if res_name == "HOH" and atom_name == "O"
                    )
                ) if reactant_prepared is not None else 0,
                "mean_final_work_kcal_mol": float(smd.get("mean_final_work_kcal_mol", 0.0)),
                "delta_g_smd_barrier_kcal_mol": float(smd.get("delta_g_smd_barrier_kcal_mol", 0.0)),
            }
        except Exception as exc:  # noqa: BLE001
            report = {
                "candidate_id": row.get("candidate_id"),
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)

    ok_reports = [r for r in reports if r.get("status") == "ok"]
    final_product = [float(r["replica"].get("final_product_rmsd_a", 0.0)) for r in ok_reports]
    max_product = [float(r["replica"].get("max_product_rmsd_a", 0.0)) for r in ok_reports]
    max_pocket = [float(r["replica"].get("max_pocket_rmsd_a", 0.0)) for r in ok_reports]
    max_backbone = [float(r["replica"].get("max_backbone_rmsd_a", 0.0)) for r in ok_reports]
    max_ca_network = [float(r["replica"].get("max_ca_network_rms_a", 0.0)) for r in ok_reports]
    max_close_contacts = [float(r["replica"].get("max_close_contacts", 0.0)) for r in ok_reports]
    max_excess_bonds = [float(r["replica"].get("max_excess_bond_count", 0.0)) for r in ok_reports]
    max_nonpocket_backbone = [float(r["replica"].get("max_nonpocket_backbone_rmsd_a", 0.0)) for r in ok_reports]
    max_protein_rg_drift = [float(r["replica"].get("max_protein_rg_drift_a", 0.0)) for r in ok_reports]
    protocol_counts = Counter(str(r.get("protocol_mode", "missing")) for r in ok_reports)
    n_reactive_protocol = sum(1 for r in ok_reports if bool(r.get("reactive_barrier_valid", False)))
    n_pmf_eligible = sum(1 for r in ok_reports if bool(r.get("pmf_eligible", False)))
    n_pass = 0
    for row in ok_reports:
        rep = row["replica"]
        if (
            float(rep.get("final_product_rmsd_a", 1e9)) <= float(args.threshold_final_product_rmsd_a)
            and float(rep.get("max_product_rmsd_a", 1e9)) <= float(args.threshold_max_product_rmsd_a)
            and float(rep.get("max_pocket_rmsd_a", 1e9)) <= float(args.threshold_max_pocket_rmsd_a)
            and float(rep.get("max_backbone_rmsd_a", 1e9)) <= float(args.threshold_max_backbone_rmsd_a)
            and float(rep.get("max_ca_network_rms_a", 1e9)) <= float(args.threshold_max_ca_network_rms_a)
            and int(rep.get("max_close_contacts", 10**9)) <= int(args.threshold_max_close_contacts)
            and int(rep.get("max_excess_bond_count", 10**9)) <= int(args.threshold_max_excess_bond_count)
        ):
            n_pass += 1
    report_payload = {
        "config": vars(args),
        "n_total": len(reports),
        "n_ok": len(ok_reports),
        "ok_fraction": float(len(ok_reports) / len(reports)) if reports else 0.0,
        "n_quality_pass": int(n_pass),
        "quality_pass_fraction": float(n_pass / len(ok_reports)) if ok_reports else 0.0,
        "protocol_mode_counts": dict(protocol_counts),
        "reactive_protocol_fraction": float(n_reactive_protocol / len(ok_reports)) if ok_reports else 0.0,
        "pmf_eligible_fraction": float(n_pmf_eligible / len(ok_reports)) if ok_reports else 0.0,
        "aggregate": {
            "final_product_rmsd_mean_a": _float_mean(final_product),
            "final_product_rmsd_median_a": _float_median(final_product),
            "max_product_rmsd_mean_a": _float_mean(max_product),
            "max_pocket_rmsd_mean_a": _float_mean(max_pocket),
            "max_backbone_rmsd_mean_a": _float_mean(max_backbone),
            "max_nonpocket_backbone_rmsd_mean_a": _float_mean(max_nonpocket_backbone),
            "max_ca_network_rms_mean_a": _float_mean(max_ca_network),
            "max_protein_rg_drift_mean_a": _float_mean(max_protein_rg_drift),
            "max_close_contacts_mean": _float_mean(max_close_contacts),
            "max_excess_bond_count_mean": _float_mean(max_excess_bonds),
        },
        "reports": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report_payload)
    logger.info(
        "UMA sMD validation complete: ok=%d/%d final_product_rmsd_mean=%.3f max_backbone_rmsd_mean=%.3f max_nonpocket_backbone_rmsd_mean=%.3f max_ca_network_rms_mean=%.3f max_protein_rg_drift_mean=%.3f wrote=%s",
        len(ok_reports),
        len(reports),
        float(report_payload["aggregate"]["final_product_rmsd_mean_a"]),
        float(report_payload["aggregate"]["max_backbone_rmsd_mean_a"]),
        float(report_payload["aggregate"]["max_nonpocket_backbone_rmsd_mean_a"]),
        float(report_payload["aggregate"]["max_ca_network_rms_mean_a"]),
        float(report_payload["aggregate"]["max_protein_rg_drift_mean_a"]),
        output_path,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
