#!/usr/bin/env python3
"""Score catalytic candidates with whole-enzyme UMA broad screening and sMD."""

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


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calculator-workers", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--broad-timestep-fs", type=float, default=0.1)
    parser.add_argument("--broad-friction-ps-inv", type=float, default=1.0)
    parser.add_argument("--broad-steps", type=int, default=1000)
    parser.add_argument("--broad-replicas", type=int, default=3)
    parser.add_argument("--broad-save-every", type=int, default=20)
    parser.add_argument("--contact-cutoff-a", type=float, default=4.5)
    parser.add_argument("--ligand-rmsd-max-a", type=float, default=2.5)
    parser.add_argument("--pocket-rmsd-max-a", type=float, default=2.0)
    parser.add_argument("--min-contacts", type=int, default=4)
    parser.add_argument("--min-distance-safe-a", type=float, default=1.5)
    parser.add_argument("--anchor-stride", type=int, default=6)
    parser.add_argument("--protocol-max-reactive-bonds", type=int, default=8)
    parser.add_argument("--protocol-max-reactive-atoms", type=int, default=12)
    parser.add_argument("--protocol-max-reactive-fraction", type=float, default=0.35)
    parser.add_argument("--run-smd", type=int, default=1)
    parser.add_argument("--run-reverse-smd", type=int, default=1)
    parser.add_argument("--smd-temperature-k", type=float, default=300.0)
    parser.add_argument("--smd-timestep-fs", type=float, default=0.05)
    parser.add_argument("--smd-friction-ps-inv", type=float, default=2.0)
    parser.add_argument("--smd-images", type=int, default=96)
    parser.add_argument("--smd-steps-per-image", type=int, default=48)
    parser.add_argument("--smd-replicas", type=int, default=2)
    parser.add_argument("--smd-k-steer-eva2", type=float, default=0.01)
    parser.add_argument("--smd-k-global-eva2", type=float, default=0.03)
    parser.add_argument("--smd-k-local-eva2", type=float, default=0.15)
    parser.add_argument("--smd-k-anchor-eva2", type=float, default=0.01)
    parser.add_argument("--smd-ca-network-sequential-k-eva2", type=float, default=12.0)
    parser.add_argument("--smd-ca-network-contact-k-eva2", type=float, default=0.50)
    parser.add_argument("--smd-ca-network-contact-cutoff-a", type=float, default=8.0)
    parser.add_argument("--smd-force-clip-eva", type=float, default=0.35)
    parser.add_argument("--smd-production-warmup-steps", type=int, default=40)
    parser.add_argument("--quality-max-final-product-rmsd-a", type=float, default=1.75)
    parser.add_argument("--quality-max-max-product-rmsd-a", type=float, default=5.0)
    parser.add_argument("--quality-max-max-pocket-rmsd-a", type=float, default=4.0)
    parser.add_argument("--quality-max-max-backbone-rmsd-a", type=float, default=3.0)
    parser.add_argument("--quality-max-max-ca-network-rms-a", type=float, default=0.75)
    parser.add_argument("--quality-max-max-close-contacts", type=int, default=0)
    parser.add_argument("--quality-max-max-excess-bond-count", type=int, default=0)
    parser.add_argument("--quality-require-smd-pass-for-pmf", type=int, default=1)
    parser.add_argument("--prepare-hydrogens", type=int, default=1)
    parser.add_argument("--add-first-shell-waters", type=int, default=1)
    parser.add_argument("--preparation-ph", type=float, default=7.4)
    parser.add_argument("--max-first-shell-waters", type=int, default=12)
    parser.add_argument("--water-shell-distance-a", type=float, default=2.8)
    parser.add_argument("--water-clash-distance-a", type=float, default=2.1)
    parser.add_argument("--water-bridge-distance-min-a", type=float, default=4.2)
    parser.add_argument("--water-bridge-distance-max-a", type=float, default=6.6)
    parser.add_argument("--relax-prepared-steps", type=int, default=25)
    parser.add_argument("--relax-prepared-fmax-eva", type=float, default=0.20)
    parser.add_argument("--run-pmf", type=int, default=0)
    parser.add_argument("--pmf-windows", type=int, default=20)
    parser.add_argument("--pmf-steps-per-window", type=int, default=320)
    parser.add_argument("--pmf-save-every", type=int, default=10)
    parser.add_argument("--pmf-replicas", type=int, default=2)
    parser.add_argument("--pmf-k-window-eva2", type=float, default=2.0)
    parser.add_argument("--pmf-k-local-eva2", type=float, default=0.08)
    parser.add_argument("--pmf-window-relax-steps", type=int, default=32)
    parser.add_argument("--pmf-window-equilibrate-steps", type=int, default=64)
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    fairchem_src = root / "models" / "fairchem" / "src"
    if fairchem_src.exists():
        sys.path.insert(0, str(fairchem_src))

    from train.thermogfn.io_utils import read_records, write_json, write_records
    from train.thermogfn.metrics_utils import summarize_uma_cat_rows
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.uma_cat_runtime import (
        count_atoms_in_structure,
        run_broad_uma_screen,
        run_path_umbrella_pmf,
        run_steered_uma_dynamics,
        summarize_catalytic_screen,
        assess_smd_quality,
        prepare_structure_for_uma,
        relax_structure_under_uma,
        write_catalytic_artifacts,
    )

    logger = configure_logging("oracle.uma_catalytic", level=args.log_level)
    input_path = _resolve_path(root, args.candidate_path)
    output_path = _resolve_path(root, args.output_path)
    artifact_root = _resolve_path(root, args.artifact_root)
    rows = read_records(input_path)
    logger.info(
        "UMA catalytic scoring start: rows=%d artifacts=%s device=%s run_smd=%s reverse_smd=%s",
        len(rows),
        artifact_root,
        args.device,
        bool(args.run_smd),
        bool(args.run_reverse_smd),
    )

    out: list[dict] = []
    for rec in iter_progress(rows, total=len(rows), desc="uma-cat:score", no_progress=args.no_progress):
        row = dict(rec)
        candidate_id = str(row.get("candidate_id") or "")
        try:
            if row.get("packing_status") != "ok":
                raise RuntimeError(f"candidate not packed successfully: {row.get('packing_status')}")
            reactant_path = _resolve_path(root, row["reactant_complex_packed_path"])
            product_path = _resolve_path(root, row["product_complex_packed_path"])
            protein_chain_id = str(row.get("protein_chain_id") or "")
            if not protein_chain_id:
                raise ValueError("missing protein_chain_id")
            pocket_positions = [int(x) for x in (row.get("pocket_positions") or [])]
            if not pocket_positions:
                raise ValueError("missing pocket_positions")

            reactant_prepared = prepare_structure_for_uma(
                reactant_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=row.get("ligand_chain_id"),
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
                ligand_chain_id=row.get("ligand_chain_id"),
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

            atom_count = int(reactant_prepared.count_atoms()) if reactant_prepared is not None else count_atoms_in_structure(reactant_path)
            row["prepared_atom_count"] = int(atom_count)
            if reactant_prepared is not None:
                row["prepared_first_shell_water_molecules"] = int(
                    sum(1 for res_name, atom_name in zip(reactant_prepared.residue_names, reactant_prepared.atom_names, strict=False) if res_name == "HOH" and atom_name == "O")
                )
            if int(args.max_atoms) > 0 and atom_count > int(args.max_atoms):
                raise RuntimeError(f"atom budget exceeded: atoms={atom_count} max_atoms={args.max_atoms}")

            broad = run_broad_uma_screen(
                reactant_complex_path=reactant_path,
                reactant_structure=reactant_prepared,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=row.get("ligand_chain_id"),
                pocket_positions=pocket_positions,
                temperature_k=float(args.temperature_k),
                timestep_fs=float(args.broad_timestep_fs),
                friction_ps_inv=float(args.broad_friction_ps_inv),
                steps=int(args.broad_steps),
                replicas=int(args.broad_replicas),
                save_every=int(args.broad_save_every),
                model_name=args.model_name,
                device=args.device,
                calculator_workers=int(args.calculator_workers),
                contact_cutoff_a=float(args.contact_cutoff_a),
                ligand_rmsd_max_a=float(args.ligand_rmsd_max_a),
                pocket_rmsd_max_a=float(args.pocket_rmsd_max_a),
                min_contacts=int(args.min_contacts),
                min_distance_safe_a=float(args.min_distance_safe_a),
                anchor_stride=int(args.anchor_stride),
                prepare_hydrogens=bool(args.prepare_hydrogens),
                add_first_shell_waters=bool(args.add_first_shell_waters),
                preparation_ph=float(args.preparation_ph),
                max_first_shell_waters=int(args.max_first_shell_waters),
                water_shell_distance_a=float(args.water_shell_distance_a),
                water_clash_distance_a=float(args.water_clash_distance_a),
                water_bridge_distance_min_a=float(args.water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(args.water_bridge_distance_max_a),
                seed=int(args.seed),
            )

            smd_forward = None
            smd_reverse = None
            pmf = None
            smd_quality = None
            pmf_skip_reason = ""
            if bool(args.run_smd):
                smd_forward = run_steered_uma_dynamics(
                    reactant_complex_path=reactant_path,
                    product_complex_path=product_path,
                    reactant_structure=reactant_prepared,
                    product_structure=product_prepared,
                    protein_chain_id=protein_chain_id,
                    ligand_chain_id=row.get("ligand_chain_id"),
                    pocket_positions=pocket_positions,
                    temperature_k=float(args.smd_temperature_k),
                    timestep_fs=float(args.smd_timestep_fs),
                    friction_ps_inv=float(args.smd_friction_ps_inv),
                    images=int(args.smd_images),
                    steps_per_image=int(args.smd_steps_per_image),
                    replicas=int(args.smd_replicas),
                    k_steer_eva2=float(args.smd_k_steer_eva2),
                    k_global_eva2=float(args.smd_k_global_eva2),
                    k_local_eva2=float(args.smd_k_local_eva2),
                    k_anchor_eva2=float(args.smd_k_anchor_eva2),
                    ca_network_sequential_k_eva2=float(args.smd_ca_network_sequential_k_eva2),
                    ca_network_contact_k_eva2=float(args.smd_ca_network_contact_k_eva2),
                    ca_network_contact_cutoff_a=float(args.smd_ca_network_contact_cutoff_a),
                    model_name=args.model_name,
                    device=args.device,
                    calculator_workers=int(args.calculator_workers),
                    anchor_stride=int(args.anchor_stride),
                    force_clip_eva=float(args.smd_force_clip_eva),
                    production_warmup_steps=int(args.smd_production_warmup_steps),
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
                    seed=int(args.seed),
                )
                if bool(args.run_reverse_smd):
                    smd_reverse = run_steered_uma_dynamics(
                        reactant_complex_path=product_path,
                        product_complex_path=reactant_path,
                        reactant_structure=product_prepared,
                        product_structure=reactant_prepared,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=row.get("ligand_chain_id"),
                        pocket_positions=pocket_positions,
                        temperature_k=float(args.smd_temperature_k),
                        timestep_fs=float(args.smd_timestep_fs),
                        friction_ps_inv=float(args.smd_friction_ps_inv),
                        images=int(args.smd_images),
                        steps_per_image=int(args.smd_steps_per_image),
                        replicas=int(args.smd_replicas),
                        k_steer_eva2=float(args.smd_k_steer_eva2),
                        k_global_eva2=float(args.smd_k_global_eva2),
                        k_local_eva2=float(args.smd_k_local_eva2),
                        k_anchor_eva2=float(args.smd_k_anchor_eva2),
                        ca_network_sequential_k_eva2=float(args.smd_ca_network_sequential_k_eva2),
                        ca_network_contact_k_eva2=float(args.smd_ca_network_contact_k_eva2),
                        ca_network_contact_cutoff_a=float(args.smd_ca_network_contact_cutoff_a),
                        model_name=args.model_name,
                        device=args.device,
                        calculator_workers=int(args.calculator_workers),
                        anchor_stride=int(args.anchor_stride),
                        force_clip_eva=float(args.smd_force_clip_eva),
                        production_warmup_steps=int(args.smd_production_warmup_steps),
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
                        seed=int(args.seed) + 1000,
                    )
                smd_quality = assess_smd_quality(
                    smd_forward,
                    max_final_product_rmsd_a=float(args.quality_max_final_product_rmsd_a),
                    max_max_product_rmsd_a=float(args.quality_max_max_product_rmsd_a),
                    max_max_pocket_rmsd_a=float(args.quality_max_max_pocket_rmsd_a),
                    max_max_backbone_rmsd_a=float(args.quality_max_max_backbone_rmsd_a),
                    max_max_ca_network_rms_a=float(args.quality_max_max_ca_network_rms_a),
                    max_max_close_contacts=int(args.quality_max_max_close_contacts),
                    max_max_excess_bond_count=int(args.quality_max_max_excess_bond_count),
                )
                if smd_forward is not None and bool(args.run_reverse_smd) and smd_reverse is not None:
                    smd_forward["forward_reverse_gap_kcal_mol"] = float(
                        abs(float(smd_forward.get("mean_final_work_kcal_mol", 0.0)) + float(smd_reverse.get("mean_final_work_kcal_mol", 0.0)))
                    )
                pmf_protocol_eligible = bool((smd_forward or {}).get("mapping", {}).get("pmf_eligible", False))
                if bool(args.run_pmf) and not pmf_protocol_eligible:
                    protocol_mode = str((smd_forward or {}).get("mapping", {}).get("protocol_mode", "unknown"))
                    protocol_reason = str((smd_forward or {}).get("mapping", {}).get("protocol_reason", ""))
                    pmf_skip_reason = f"protocol_ineligible:{protocol_mode}:{protocol_reason}"
                elif bool(args.run_pmf) and bool(args.quality_require_smd_pass_for_pmf) and not bool(smd_quality.get("pass", False)):
                    pmf_skip_reason = "smd_quality_fail"
                elif bool(args.run_pmf):
                    pmf = run_path_umbrella_pmf(
                        reactant_complex_path=reactant_path,
                        product_complex_path=product_path,
                        reactant_structure=reactant_prepared,
                        product_structure=product_prepared,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=row.get("ligand_chain_id"),
                        pocket_positions=pocket_positions,
                        temperature_k=float(args.temperature_k),
                        timestep_fs=float(args.smd_timestep_fs),
                        friction_ps_inv=float(args.smd_friction_ps_inv),
                        windows=int(args.pmf_windows),
                        steps_per_window=int(args.pmf_steps_per_window),
                        save_every=int(args.pmf_save_every),
                        replicas=int(args.pmf_replicas),
                        k_window_eva2=float(args.pmf_k_window_eva2),
                        k_global_eva2=float(args.smd_k_global_eva2),
                        k_local_eva2=float(args.pmf_k_local_eva2),
                        k_anchor_eva2=float(args.smd_k_anchor_eva2),
                        ca_network_sequential_k_eva2=float(args.smd_ca_network_sequential_k_eva2),
                        ca_network_contact_k_eva2=float(args.smd_ca_network_contact_k_eva2),
                        ca_network_contact_cutoff_a=float(args.smd_ca_network_contact_cutoff_a),
                        model_name=args.model_name,
                        device=args.device,
                        calculator_workers=int(args.calculator_workers),
                        anchor_stride=int(args.anchor_stride),
                        path_lambdas=smd_forward.get("path_lambdas"),
                        path_targets=smd_forward.get("path_targets"),
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
                        seed=int(args.seed) + 2000,
                        window_relax_steps=int(args.pmf_window_relax_steps),
                        window_equilibrate_steps=int(args.pmf_window_equilibrate_steps),
                    )

            summary = summarize_catalytic_screen(
                broad=broad,
                smd=smd_forward,
                pmf=pmf,
                temperature_k=float(args.temperature_k),
            )
            if smd_reverse is not None:
                summary["uma_cat_reverse_delta_g_smd_barrier_kcal_mol"] = float(
                    smd_reverse.get("delta_g_smd_barrier_kcal_mol", 0.0)
                )
                summary["uma_cat_reverse_mean_work_kcal_mol"] = float(
                    smd_reverse.get("mean_final_work_kcal_mol", 0.0)
                )
                summary["uma_cat_reverse_work_std_kcal_mol"] = float(
                    smd_reverse.get("std_final_work_kcal_mol", 0.0)
                )
                summary["uma_cat_forward_reverse_gap_kcal_mol"] = float(
                    abs(
                        float(smd_forward.get("mean_final_work_kcal_mol", 0.0) if smd_forward else 0.0)
                        + float(smd_reverse.get("mean_final_work_kcal_mol", 0.0))
                    )
                )
            if smd_quality is not None:
                summary["uma_cat_smd_quality_pass"] = bool(smd_quality.get("pass", False))
                summary["uma_cat_smd_quality_fail_reasons"] = list(smd_quality.get("reasons", []))
                summary["uma_cat_smd_protocol_mode"] = str(smd_quality.get("protocol_mode", "unknown"))
                summary["uma_cat_smd_protocol_reason"] = str(smd_quality.get("protocol_reason", ""))
                summary["uma_cat_smd_reactive_barrier_valid"] = bool(smd_quality.get("reactive_barrier_valid", False))
                summary["uma_cat_smd_pmf_eligible"] = bool(smd_quality.get("pmf_eligible", False))
                summary["uma_cat_smd_quality_worst_final_product_rmsd_a"] = float(smd_quality.get("worst_final_product_rmsd_a", 0.0))
                summary["uma_cat_smd_quality_worst_max_product_rmsd_a"] = float(smd_quality.get("worst_max_product_rmsd_a", 0.0))
                summary["uma_cat_smd_quality_worst_max_pocket_rmsd_a"] = float(smd_quality.get("worst_max_pocket_rmsd_a", 0.0))
                summary["uma_cat_smd_quality_worst_max_backbone_rmsd_a"] = float(smd_quality.get("worst_max_backbone_rmsd_a", 0.0))
                summary["uma_cat_smd_quality_worst_max_ca_network_rms_a"] = float(smd_quality.get("worst_max_ca_network_rms_a", 0.0))
                summary["uma_cat_smd_quality_worst_max_close_contacts"] = int(smd_quality.get("worst_max_close_contacts", 0))
                summary["uma_cat_smd_quality_worst_max_excess_bond_count"] = int(smd_quality.get("worst_max_excess_bond_count", 0))
                if not bool(smd_quality.get("pass", False)):
                    summary["uma_cat_status"] = "quality_fail"
                    summary["uma_cat_error"] = "smd_quality_fail:" + ",".join(str(x) for x in smd_quality.get("reasons", []))
            if bool(args.run_pmf) and pmf is None and pmf_skip_reason:
                summary["uma_cat_pmf_skipped"] = True
                summary["uma_cat_pmf_skip_reason"] = str(pmf_skip_reason)
                if pmf_skip_reason == "smd_quality_fail":
                    summary["uma_cat_pmf_skipped_due_to_quality"] = True
                if pmf_skip_reason.startswith("protocol_ineligible:"):
                    summary["uma_cat_pmf_skipped_due_to_protocol"] = True

            artifact_dir = artifact_root / candidate_id
            artifacts = write_catalytic_artifacts(
                artifact_dir,
                record=row,
                broad=broad,
                smd=smd_forward,
                pmf=pmf,
                summary=summary,
            )
            if smd_reverse is not None:
                reverse_summary_path = artifact_dir / "smd_reverse_summary.json"
                reverse_summary_path.write_text(
                    json.dumps(
                        {
                            "replicate_summaries": smd_reverse.get("replica_summaries", []),
                            "mapping": smd_reverse.get("mapping", {}),
                            "lambdas": smd_reverse.get("lambdas", []),
                            "jarzynski_free_profile_kcal_mol": smd_reverse.get("jarzynski_free_profile_kcal_mol", []),
                            "delta_g_smd_barrier_kcal_mol": smd_reverse.get("delta_g_smd_barrier_kcal_mol"),
                            "delta_g_smd_barrier_std_kcal_mol": smd_reverse.get("delta_g_smd_barrier_std_kcal_mol"),
                            "delta_g_react_to_prod_kcal_mol": smd_reverse.get("delta_g_react_to_prod_kcal_mol"),
                            "delta_g_react_to_prod_std_kcal_mol": smd_reverse.get("delta_g_react_to_prod_std_kcal_mol"),
                            "mean_final_work_kcal_mol": smd_reverse.get("mean_final_work_kcal_mol"),
                            "std_final_work_kcal_mol": smd_reverse.get("std_final_work_kcal_mol"),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                artifacts["smd_reverse_summary_json"] = str(reverse_summary_path)

            row.update(summary)
            row.update(artifacts)
            row["uma_cat_elapsed_s"] = float(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            row["uma_cat_status"] = "error"
            row["uma_cat_error"] = str(exc)
            logger.error("UMA catalytic scoring failed candidate_id=%s error=%s", candidate_id or "<unknown>", exc)
        out.append(row)

    write_records(output_path, out)
    summary_metrics = summarize_uma_cat_rows(out)
    if args.summary_path:
        write_json(_resolve_path(root, args.summary_path), summary_metrics)
    logger.info(
        "UMA catalytic summary: ok_fraction=%.3f ok=%d/%d mean_log10_rate_proxy=%.4f",
        float(summary_metrics.get("ok_fraction", 0.0)),
        int(summary_metrics.get("status_counts", {}).get("ok", 0)),
        int(summary_metrics.get("n", 0)),
        float(summary_metrics.get("log10_rate_proxy", {}).get("mean", 0.0)),
    )
    logger.info("UMA catalytic scoring complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
