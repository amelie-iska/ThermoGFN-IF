#!/usr/bin/env python3
"""Generate a candidate pool from a fine-tuned LigandMPNN sequence checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _copy_seed_metadata(seed_rec: dict, rec: dict) -> None:
    for key in (
        "split",
        "spec_path",
        "cif_path",
        "complex_path",
        "decomposition",
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
        "pair_id",
        "ligand_chain_id",
        "protein_chain_id",
        "reactant_complex_path",
        "reactant_protein_path",
        "product_complex_path",
        "product_protein_path",
        "pocket_positions",
        "rhea_id",
        "uniprot_id",
    ):
        if key in seed_rec:
            rec[key] = seed_rec[key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-ckpt", required=True)
    parser.add_argument("--input-dr", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--ligandmpnn-root", default="models/LigandMPNN")
    parser.add_argument("--model-type", default="ligand_mpnn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--structure-key", default="reactant_complex_path")
    parser.add_argument("--sequence-key", default="sequence")
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mutable-mode", choices=["full_chain", "random", "pocket_random"], default="pocket_random")
    parser.add_argument("--min-mutations", type=int, default=3)
    parser.add_argument("--max-mutations", type=int, default=12)
    parser.add_argument("--pocket-fraction", type=float, default=0.75)
    parser.add_argument("--use-atom-context", type=int, default=1)
    parser.add_argument("--use-side-chain-context", type=int, default=0)
    parser.add_argument("--parse-atoms-with-zero-occupancy", type=int, default=1)
    parser.add_argument("--ligand-cutoff-a", type=float, default=8.0)
    parser.add_argument("--max-attempts-multiplier", type=int, default=20)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.features import deterministic_hash
    from train.thermogfn.io_utils import read_records, write_json, write_records
    from train.thermogfn.ligandmpnn_generator import (
        design_chain_sequence_from_tensor,
        load_ligandmpnn_sequence_model,
        make_ligandmpnn_feature_dict,
        mutation_list,
    )
    from train.thermogfn.metrics_utils import summarize_candidate_records
    from train.thermogfn.progress import configure_logging, make_progress
    from train.thermogfn.schemas import ensure_unique_ids, validate_records

    import torch

    logger = configure_logging("train.ligandmpnn_pool", level=args.log_level)
    rng = random.Random(args.seed + args.round_id)
    torch.manual_seed(args.seed + args.round_id)
    model, checkpoint, torch_device, atom_context_num, ckpt_path = load_ligandmpnn_sequence_model(
        repo_root=root,
        ligandmpnn_root=args.ligandmpnn_root,
        checkpoint_path=args.generator_ckpt,
        device=args.device,
        model_type=args.model_type,
        use_side_chain_context=args.use_side_chain_context,
    )
    generator_training = checkpoint.get("thermogfn_generator_training", {}) if isinstance(checkpoint, dict) else {}
    generator_objective = str(generator_training.get("objective") or checkpoint.get("thermogfn_gflownet_state", {}).get("objective", "unknown"))
    model.eval()

    rows = read_records(root / args.input_dr)
    seeds = [
        r for r in rows
        if str(r.get("split", args.split)) == str(args.split)
        and r.get(args.sequence_key)
        and (r.get(args.structure_key) or r.get("complex_path") or r.get("cif_path"))
    ]
    if not seeds:
        seeds = [
            r for r in rows
            if r.get(args.sequence_key) and (r.get(args.structure_key) or r.get("complex_path") or r.get("cif_path"))
        ]
    if not seeds:
        raise RuntimeError(f"No LigandMPNN sampling seeds found in {args.input_dr}")
    logger.info(
        "Generating LigandMPNN pool: seeds=%d pool_size=%d checkpoint=%s temperature=%.3f batch_size=%d",
        len(seeds),
        args.pool_size,
        ckpt_path,
        args.temperature,
        args.sample_batch_size,
    )

    feature_root = (root / args.output_path).parent / f"ligandmpnn_sampling_cache_round_{args.round_id}"
    feature_root.mkdir(parents=True, exist_ok=True)
    pool_by_id: dict[str, dict] = {}
    attempts = 0
    max_attempts = max(int(args.pool_size), 1) * max(int(args.max_attempts_multiplier), 1)
    pbar = make_progress(total=args.pool_size, desc="ligandmpnn:pool", no_progress=args.no_progress, leave=True, unit="candidate")
    seed_index = 0
    skipped: list[dict] = []
    feature_cache: dict[str, tuple[dict, dict, list[int]]] = {}

    def _apply_mutable_mask(feature_dict: dict, seed_rec: dict, design_idx: list[int]) -> list[int]:
        if args.mutable_mode == "full_chain":
            return list(range(len(design_idx)))
        seq_len = len(str(seed_rec[args.sequence_key]).strip())
        all_positions = list(range(seq_len))
        pocket_positions = []
        for raw in seed_rec.get("pocket_positions") or []:
            try:
                pos = int(raw) - 1
            except Exception:  # noqa: BLE001
                continue
            if 0 <= pos < seq_len:
                pocket_positions.append(pos)
        pocket_positions = sorted(set(pocket_positions))
        non_pocket = [pos for pos in all_positions if pos not in set(pocket_positions)]
        k_min = max(1, int(args.min_mutations))
        k_max = max(k_min, int(args.max_mutations))
        k = min(seq_len, rng.randint(k_min, k_max))
        if args.mutable_mode == "pocket_random" and pocket_positions:
            n_pocket = min(len(pocket_positions), max(1, int(round(k * max(0.0, min(1.0, args.pocket_fraction))))))
            chosen = rng.sample(pocket_positions, n_pocket)
            remaining = max(0, k - len(chosen))
            if remaining > 0:
                source = non_pocket if len(non_pocket) >= remaining else [pos for pos in all_positions if pos not in set(chosen)]
                chosen.extend(rng.sample(source, min(remaining, len(source))))
        else:
            chosen = rng.sample(all_positions, min(k, len(all_positions)))
        chosen = sorted(set(chosen))
        import torch

        mask = torch.zeros_like(feature_dict["chain_mask"])
        for seq_pos in chosen:
            if seq_pos < len(design_idx):
                mask[0, design_idx[seq_pos]] = 1
        feature_dict["chain_mask"] = mask
        return chosen

    with torch.no_grad():
        while len(pool_by_id) < int(args.pool_size) and attempts < max_attempts:
            seed_rec = seeds[seed_index % len(seeds)]
            seed_index += 1
            attempts += 1
            try:
                cache_key = str(seed_rec.get("candidate_id") or seed_rec.get("backbone_id") or seed_index)
                if cache_key not in feature_cache:
                    feature_dict, protein_dict, design_idx = make_ligandmpnn_feature_dict(
                        repo_root=root,
                        record=seed_rec,
                        ligandmpnn_root=args.ligandmpnn_root,
                        work_dir=feature_root,
                        device=torch_device,
                        atom_context_num=atom_context_num,
                        model_type=args.model_type,
                        structure_key=args.structure_key,
                        sequence_key=args.sequence_key,
                        use_atom_context=args.use_atom_context,
                        parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
                        ligand_cutoff_a=args.ligand_cutoff_a,
                        batch_size=args.sample_batch_size,
                        temperature=args.temperature,
                    )
                    feature_cache[cache_key] = (feature_dict, protein_dict, design_idx)
                feature_dict, protein_dict, design_idx = feature_cache[cache_key]
                mutable_positions = _apply_mutable_mask(feature_dict, seed_rec, design_idx)
                length = int(feature_dict["S"].shape[1])
                feature_dict["randn"] = torch.randn((int(args.sample_batch_size), length), dtype=torch.float32, device=torch_device)
                output = model.sample(feature_dict)
                for sampled in output["S"]:
                    sampled_sequence = design_chain_sequence_from_tensor(
                        protein_dict,
                        sampled,
                        seed_rec.get("protein_chain_id") or seed_rec.get("chain_id"),
                    )
                    seed_sequence = str(seed_rec[args.sequence_key]).strip().upper()
                    muts = mutation_list(seed_sequence, sampled_sequence)
                    cid = deterministic_hash(
                        f"{args.run_id}:{args.round_id}:ligandmpnn:{seed_rec.get('backbone_id')}:{sampled_sequence}"
                    )
                    if cid in pool_by_id:
                        continue
                    rec = {
                        "candidate_id": cid,
                        "run_id": args.run_id,
                        "round_id": args.round_id,
                        "task_type": seed_rec.get("task_type", "ligand"),
                        "backbone_id": seed_rec["backbone_id"],
                        "seed_id": seed_rec.get("seed_id", seed_rec["backbone_id"]),
                        "sequence": sampled_sequence,
                        "mutations": muts,
                        "K": len(muts),
                        "prepared_atom_count": int(seed_rec.get("prepared_atom_count", 0)),
                        "eligibility": dict(seed_rec.get("eligibility", {"bioemu": False, "uma_whole": True, "uma_local": False})),
                        "source": "student",
                        "source_model": "LigandMPNN",
                        "generator_backend": "ligandmpnn",
                        "generator_training_objective": generator_objective,
                        "generator_checkpoint": str(ckpt_path),
                        "generator_temperature": float(args.temperature),
                        "generator_mutable_mode": str(args.mutable_mode),
                        "generator_mutable_positions": [int(p) + 1 for p in mutable_positions],
                        "schema_version": "v1",
                        "sequence_length": len(sampled_sequence),
                    }
                    _copy_seed_metadata(seed_rec, rec)
                    pool_by_id[cid] = rec
                    pbar.update(1)
                    if len(pool_by_id) >= int(args.pool_size):
                        break
            except Exception as exc:  # noqa: BLE001
                skipped.append({"candidate_id": seed_rec.get("candidate_id"), "error": str(exc)})
                logger.warning("Skipping LigandMPNN sampling seed candidate_id=%s error=%s", seed_rec.get("candidate_id"), exc)
    pbar.close()

    pool = list(pool_by_id.values())
    ensure_unique_ids(pool, "candidate_id")
    summary = validate_records(pool, "candidate")
    if summary.invalid:
        for err in summary.errors[:10]:
            print(err, file=sys.stderr)
        return 3
    if len(pool) < int(args.pool_size):
        logger.warning("LigandMPNN pool shortfall: requested=%d wrote=%d attempts=%d", args.pool_size, len(pool), attempts)

    write_records(root / args.output_path, pool)
    metrics = summarize_candidate_records(pool)
    metrics.update(
        {
            "generator_backend": "ligandmpnn",
            "generator_training_objective": generator_objective,
            "generator_checkpoint": str(ckpt_path),
            "mutable_mode": str(args.mutable_mode),
            "min_mutations": int(args.min_mutations),
            "max_mutations": int(args.max_mutations),
            "attempts": attempts,
            "max_attempts": max_attempts,
            "skipped_seeds": len(skipped),
            "skipped_examples": skipped[:10],
            "elapsed_s": time.perf_counter() - t0,
        }
    )
    if args.metrics_path:
        write_json(root / args.metrics_path, metrics)
    logger.info("LigandMPNN pool complete: wrote=%d attempts=%d elapsed=%.2fs", len(pool), attempts, time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
