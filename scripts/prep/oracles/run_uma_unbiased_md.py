#!/usr/bin/env python3
"""Run unbiased UMA Langevin MD on one reactant-bound enzyme-ligand complex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


DEFAULT_DATASET_PATH = "runs/tmp/uma_cat_rf3_train_real.jsonl"
DEFAULT_CANDIDATE_ID = "83985134209fb7f8"
DEFAULT_OUTPUT_PDB = f"./reactant_bound_{DEFAULT_CANDIDATE_ID}_uma_md_multimodel.pdb"
DEFAULT_OUTPUT_SUMMARY = f"./reactant_bound_{DEFAULT_CANDIDATE_ID}_uma_md_summary.json"


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


def _load_candidate_record(dataset_path: Path, candidate_id: str) -> dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if str(row.get("candidate_id", "")) == str(candidate_id):
                return row
    raise ValueError(f"Candidate {candidate_id} not found in {dataset_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--structure-path", default="", help="Override direct reactant-bound complex path")
    parser.add_argument("--protein-chain-id", default="", help="Override protein chain ID")
    parser.add_argument("--ligand-chain-id", default="", help="Override ligand chain ID")
    parser.add_argument(
        "--pocket-positions",
        default="",
        help="Comma-separated pocket residue positions. Overrides dataset record if supplied.",
    )
    parser.add_argument("--output-pdb", default=DEFAULT_OUTPUT_PDB)
    parser.add_argument("--output-summary-json", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calculator-workers", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--friction-ps-inv", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--production-steps", type=int, default=96)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--prepare-hydrogens", type=int, default=1)
    parser.add_argument("--add-first-shell-waters", type=int, default=1)
    parser.add_argument("--preparation-ph", type=float, default=7.4)
    parser.add_argument("--max-first-shell-waters", type=int, default=12)
    parser.add_argument("--water-shell-distance-a", type=float, default=2.8)
    parser.add_argument("--water-clash-distance-a", type=float, default=2.1)
    parser.add_argument("--water-bridge-distance-min-a", type=float, default=4.2)
    parser.add_argument("--water-bridge-distance-max-a", type=float, default=6.6)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))
    fairchem_src = root / "models" / "fairchem" / "src"
    if fairchem_src.exists():
        sys.path.insert(0, str(fairchem_src))

    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from tqdm import tqdm

    from train.thermogfn.uma_cat_runtime import (
        _friction_ps_inv_to_ase,
        _get_uma_calculator,
        load_structure,
        prepare_structure_for_uma,
        write_multimodel_pdb,
    )

    dataset_path = _resolve_path(root, args.dataset_path)
    stage_bar = tqdm(
        total=6,
        desc="uma-md:stages",
        unit="stage",
        dynamic_ncols=True,
    )
    stage_bar.set_postfix_str("load-record")
    record = _load_candidate_record(dataset_path, str(args.candidate_id))
    stage_bar.update(1)

    structure_path = (
        _resolve_path(root, args.structure_path)
        if args.structure_path.strip()
        else _resolve_path(root, record["reactant_complex_path"])
    )
    protein_chain_id = args.protein_chain_id.strip() or str(record["protein_chain_id"])
    ligand_chain_id = args.ligand_chain_id.strip() or record.get("ligand_chain_id")
    if args.pocket_positions.strip():
        pocket_positions = [int(tok.strip()) for tok in args.pocket_positions.split(",") if tok.strip()]
    else:
        pocket_positions = [int(x) for x in (record.get("pocket_positions") or [])]

    tqdm.write(
        "UMA MD target: "
        f"candidate={record.get('candidate_id')} "
        f"seq_len={len(str(record.get('sequence', '')))} "
        f"protein_chain={protein_chain_id} ligand_chain={ligand_chain_id or '-'}"
    )
    tqdm.write(f"Reactant complex: {structure_path}")

    stage_bar.set_postfix_str("prepare-structure")
    prep_start = time.time()
    if bool(args.prepare_hydrogens):
        prepared = prepare_structure_for_uma(
            structure_path,
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
        )
    else:
        prepared = load_structure(structure_path)
    stage_bar.update(1)
    tqdm.write(
        "Prepared structure: "
        f"n_atoms={len(prepared.symbols)} "
        f"elapsed={time.time() - prep_start:.1f}s "
        f"hydrogens={bool(args.prepare_hydrogens)} "
        f"waters={bool(args.add_first_shell_waters)}"
    )

    stage_bar.set_postfix_str("init-uma")
    init_start = time.time()
    atoms = prepared.to_ase_atoms(charge=int(args.charge), spin=int(args.spin))
    calc = _get_uma_calculator(args.model_name, args.device, workers=int(args.calculator_workers))
    atoms.calc = calc

    rng = np.random.RandomState(int(args.seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(args.temperature_k), rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(
        atoms,
        float(args.timestep_fs) * units.fs,
        temperature_K=float(args.temperature_k),
        friction=_friction_ps_inv_to_ase(float(args.friction_ps_inv)),
    )
    stage_bar.update(1)
    tqdm.write(
        "UMA calculator ready: "
        f"model={args.model_name} device={args.device} workers={int(args.calculator_workers)} "
        f"elapsed={time.time() - init_start:.1f}s"
    )

    warmup_steps = max(0, int(args.warmup_steps))
    production_steps = max(1, int(args.production_steps))
    record_every = max(1, int(args.record_every))

    stage_bar.set_postfix_str("warmup")
    if warmup_steps:
        warmup_bar = tqdm(
            total=warmup_steps,
            desc="uma-md:warmup",
            unit="step",
            dynamic_ncols=True,
        )
        warmup_start = time.time()
        for step in range(1, warmup_steps + 1):
            dyn.run(1)
            if step == 1 or step == warmup_steps or step % max(1, min(25, warmup_steps // 10 or 1)) == 0:
                warmup_bar.set_postfix_str(
                    f"E={atoms.get_potential_energy():.3f} eV "
                    f"t_fs={step * float(args.timestep_fs):.2f}"
                )
            warmup_bar.update(1)
        warmup_bar.close()
        tqdm.write(f"Warmup complete: elapsed={time.time() - warmup_start:.1f}s")
    else:
        tqdm.write("Warmup skipped: warmup_steps=0")
    stage_bar.update(1)

    frames: list[np.ndarray] = [np.asarray(atoms.positions, dtype=np.float64).copy()]
    frame_meta: list[dict[str, float | int]] = [
        {
            "frame": 0,
            "step": 0,
            "time_fs": 0.0,
            "energy_eV": float(atoms.get_potential_energy()),
        }
    ]

    if args.device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    stage_bar.set_postfix_str("production")
    prod_bar = tqdm(
        total=production_steps,
        desc="uma-md:production",
        unit="step",
        dynamic_ncols=True,
    )
    prod_start = time.time()
    last_postfix_step = 0
    for step in range(1, production_steps + 1):
        dyn.run(1)
        if step % record_every == 0 or step == production_steps:
            energy = float(atoms.get_potential_energy())
            frames.append(np.asarray(atoms.positions, dtype=np.float64).copy())
            frame_meta.append(
                {
                    "frame": len(frames) - 1,
                    "step": int(step),
                    "time_fs": float(step) * float(args.timestep_fs),
                    "energy_eV": energy,
                }
            )
            last_postfix_step = step
            peak_vram_gb_live = None
            if args.device.startswith("cuda"):
                try:
                    import torch

                    if torch.cuda.is_available():
                        peak_vram_gb_live = float(torch.cuda.max_memory_allocated() / (1024**3))
                except Exception:
                    peak_vram_gb_live = None
            postfix = (
                f"frames={len(frames)}/{1 + (production_steps // record_every) + int(production_steps % record_every != 0)} "
                f"E={energy:.3f}eV "
                f"t_fs={step * float(args.timestep_fs):.1f}"
            )
            if peak_vram_gb_live is not None:
                postfix += f" vram={peak_vram_gb_live:.2f}GB"
            prod_bar.set_postfix_str(postfix)
        elif step == 1 or step - last_postfix_step >= max(1, record_every):
            prod_bar.set_postfix_str(
                f"frames={len(frames)} "
                f"t_fs={step * float(args.timestep_fs):.1f}"
            )
            last_postfix_step = step
        prod_bar.update(1)
    prod_bar.close()
    stage_bar.update(1)
    tqdm.write(f"Production complete: elapsed={time.time() - prod_start:.1f}s frames={len(frames)}")

    stage_bar.set_postfix_str("write-output")
    output_pdb = _resolve_path(root, args.output_pdb)
    output_summary = _resolve_path(root, args.output_summary_json)
    write_bar = tqdm(total=2, desc="uma-md:write", unit="file", dynamic_ncols=True)
    write_multimodel_pdb(prepared, frames, output_pdb, model_metadata=frame_meta)
    write_bar.set_postfix_str(output_pdb.name)
    write_bar.update(1)

    peak_vram_gb = None
    if args.device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        except Exception:
            peak_vram_gb = None

    summary = {
        "candidate_id": record.get("candidate_id"),
        "sequence_length": len(str(record.get("sequence", ""))),
        "reactant_complex_path": str(structure_path),
        "prepared_structure_path": prepared.path,
        "protein_chain_id": protein_chain_id,
        "ligand_chain_id": ligand_chain_id,
        "pocket_positions": pocket_positions,
        "model_name": args.model_name,
        "device": args.device,
        "calculator_workers": int(args.calculator_workers),
        "temperature_k": float(args.temperature_k),
        "timestep_fs": float(args.timestep_fs),
        "friction_ps_inv": float(args.friction_ps_inv),
        "warmup_steps": warmup_steps,
        "production_steps": production_steps,
        "record_every": record_every,
        "n_frames": len(frames),
        "n_atoms_prepared": int(len(prepared.symbols)),
        "prepare_hydrogens": bool(args.prepare_hydrogens),
        "add_first_shell_waters": bool(args.add_first_shell_waters),
        "peak_vram_gb": peak_vram_gb,
        "output_pdb": str(output_pdb),
        "final_energy_eV": float(frame_meta[-1]["energy_eV"]),
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_bar.set_postfix_str(output_summary.name)
    write_bar.update(1)
    write_bar.close()
    stage_bar.update(1)
    stage_bar.set_postfix_str("done")
    stage_bar.close()

    print(output_pdb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
