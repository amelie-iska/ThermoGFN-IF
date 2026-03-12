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
    parser.add_argument("--broad-friction-per-fs", type=float, default=0.01)
    parser.add_argument("--broad-steps", type=int, default=500)
    parser.add_argument("--broad-replicas", type=int, default=2)
    parser.add_argument("--broad-save-every", type=int, default=10)
    parser.add_argument("--contact-cutoff-a", type=float, default=4.5)
    parser.add_argument("--ligand-rmsd-max-a", type=float, default=2.5)
    parser.add_argument("--pocket-rmsd-max-a", type=float, default=2.0)
    parser.add_argument("--min-contacts", type=int, default=4)
    parser.add_argument("--min-distance-safe-a", type=float, default=1.5)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--run-smd", type=int, default=1)
    parser.add_argument("--run-reverse-smd", type=int, default=1)
    parser.add_argument("--smd-timestep-fs", type=float, default=0.1)
    parser.add_argument("--smd-friction-per-fs", type=float, default=0.01)
    parser.add_argument("--smd-images", type=int, default=24)
    parser.add_argument("--smd-steps-per-image", type=int, default=25)
    parser.add_argument("--smd-replicas", type=int, default=2)
    parser.add_argument("--smd-k-steer-eva2", type=float, default=2.0)
    parser.add_argument("--smd-k-anchor-eva2", type=float, default=0.05)
    parser.add_argument("--smd-force-clip-eva", type=float, default=25.0)
    parser.add_argument("--run-pmf", type=int, default=0)
    parser.add_argument("--pmf-windows", type=int, default=16)
    parser.add_argument("--pmf-steps-per-window", type=int, default=100)
    parser.add_argument("--pmf-save-every", type=int, default=5)
    parser.add_argument("--pmf-replicas", type=int, default=1)
    parser.add_argument("--pmf-k-window-eva2", type=float, default=2.0)
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

            atom_count = count_atoms_in_structure(reactant_path)
            row["prepared_atom_count"] = int(atom_count)
            if int(args.max_atoms) > 0 and atom_count > int(args.max_atoms):
                raise RuntimeError(f"atom budget exceeded: atoms={atom_count} max_atoms={args.max_atoms}")

            broad = run_broad_uma_screen(
                reactant_complex_path=reactant_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=row.get("ligand_chain_id"),
                pocket_positions=pocket_positions,
                temperature_k=float(args.temperature_k),
                timestep_fs=float(args.broad_timestep_fs),
                friction_per_fs=float(args.broad_friction_per_fs),
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
                seed=int(args.seed),
            )

            smd_forward = None
            smd_reverse = None
            pmf = None
            if bool(args.run_smd):
                smd_forward = run_steered_uma_dynamics(
                    reactant_complex_path=reactant_path,
                    product_complex_path=product_path,
                    protein_chain_id=protein_chain_id,
                    ligand_chain_id=row.get("ligand_chain_id"),
                    pocket_positions=pocket_positions,
                    temperature_k=float(args.temperature_k),
                    timestep_fs=float(args.smd_timestep_fs),
                    friction_per_fs=float(args.smd_friction_per_fs),
                    images=int(args.smd_images),
                    steps_per_image=int(args.smd_steps_per_image),
                    replicas=int(args.smd_replicas),
                    k_steer_eva2=float(args.smd_k_steer_eva2),
                    k_anchor_eva2=float(args.smd_k_anchor_eva2),
                    model_name=args.model_name,
                    device=args.device,
                    calculator_workers=int(args.calculator_workers),
                    anchor_stride=int(args.anchor_stride),
                    force_clip_eva=float(args.smd_force_clip_eva),
                    seed=int(args.seed),
                )
                if bool(args.run_reverse_smd):
                    smd_reverse = run_steered_uma_dynamics(
                        reactant_complex_path=product_path,
                        product_complex_path=reactant_path,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=row.get("ligand_chain_id"),
                        pocket_positions=pocket_positions,
                        temperature_k=float(args.temperature_k),
                        timestep_fs=float(args.smd_timestep_fs),
                        friction_per_fs=float(args.smd_friction_per_fs),
                        images=int(args.smd_images),
                        steps_per_image=int(args.smd_steps_per_image),
                        replicas=int(args.smd_replicas),
                        k_steer_eva2=float(args.smd_k_steer_eva2),
                        k_anchor_eva2=float(args.smd_k_anchor_eva2),
                        model_name=args.model_name,
                        device=args.device,
                        calculator_workers=int(args.calculator_workers),
                        anchor_stride=int(args.anchor_stride),
                        force_clip_eva=float(args.smd_force_clip_eva),
                        seed=int(args.seed) + 1000,
                    )
                if smd_forward is not None and bool(args.run_reverse_smd) and smd_reverse is not None:
                    smd_forward["forward_reverse_gap_kcal_mol"] = float(
                        abs(float(smd_forward.get("mean_final_work_kcal_mol", 0.0)) + float(smd_reverse.get("mean_final_work_kcal_mol", 0.0)))
                    )
                if bool(args.run_pmf):
                    pmf = run_path_umbrella_pmf(
                        reactant_complex_path=reactant_path,
                        product_complex_path=product_path,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=row.get("ligand_chain_id"),
                        pocket_positions=pocket_positions,
                        temperature_k=float(args.temperature_k),
                        timestep_fs=float(args.smd_timestep_fs),
                        friction_per_fs=float(args.smd_friction_per_fs),
                        windows=int(args.pmf_windows),
                        steps_per_window=int(args.pmf_steps_per_window),
                        save_every=int(args.pmf_save_every),
                        replicas=int(args.pmf_replicas),
                        k_window_eva2=float(args.pmf_k_window_eva2),
                        k_anchor_eva2=float(args.smd_k_anchor_eva2),
                        model_name=args.model_name,
                        device=args.device,
                        calculator_workers=int(args.calculator_workers),
                        anchor_stride=int(args.anchor_stride),
                        path_lambdas=smd_forward.get("path_lambdas"),
                        path_targets=smd_forward.get("path_targets"),
                        seed=int(args.seed) + 2000,
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
