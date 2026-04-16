#!/usr/bin/env python3
"""Fine-tune the real LigandMPNN sequence generator for a Method III round."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--ligandmpnn-root", default="models/LigandMPNN")
    parser.add_argument("--base-checkpoint", default="models/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
    parser.add_argument("--previous-checkpoint", default=None)
    parser.add_argument("--model-type", default="ligand_mpnn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--structure-key", default="reactant_complex_path")
    parser.add_argument("--sequence-key", default="sequence")
    parser.add_argument("--objective", choices=["trajectory_balance", "reward_weighted_nll"], default="trajectory_balance")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-scope", choices=["decoder", "output", "full"], default="decoder")
    parser.add_argument("--reward-weight-mode", choices=["reward", "log_reward", "score_exp"], default="reward")
    parser.add_argument("--baseline-weight", type=float, default=0.1)
    parser.add_argument("--reward-power", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--anchor-l2", type=float, default=1e-6)
    parser.add_argument("--supervised-nll-weight", type=float, default=0.05)
    parser.add_argument("--tb-max-mutations", type=int, default=64)
    parser.add_argument("--reward-floor", type=float, default=1e-3)
    parser.add_argument("--reward-clip-min", type=float, default=1e-6)
    parser.add_argument("--reward-clip-max", type=float, default=1e6)
    parser.add_argument("--tb-labeled-fraction", type=float, default=0.75)
    parser.add_argument("--stop-init-logit", type=float, default=-2.0)
    parser.add_argument("--stop-len-weight-init", type=float, default=4.0)
    parser.add_argument("--log-z-init", type=float, default=0.0)
    parser.add_argument("--use-atom-context", type=int, default=1)
    parser.add_argument("--use-side-chain-context", type=int, default=0)
    parser.add_argument("--parse-atoms-with-zero-occupancy", type=int, default=1)
    parser.add_argument("--ligand-cutoff-a", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.checkpoint_utils import prune_round_checkpoints
    from train.thermogfn.io_utils import read_records, write_json, write_jsonl
    from train.thermogfn.ligandmpnn_generator import (
        build_gflownet_trajectory_entries,
        compute_record_weight,
        ensure_ligandmpnn_imports,
        load_ligandmpnn_sequence_model,
        make_ligandmpnn_feature_dict,
        make_design_chain_mask,
        normalize_weights,
        set_trainable_scope,
        set_design_sequence,
    )
    from train.thermogfn.progress import configure_logging, iter_progress, make_progress

    import torch

    ensure_ligandmpnn_imports(root, args.ligandmpnn_root)
    from data_utils import get_score
    from train.thermogfn.constants import AMINO_ACIDS
    from data_utils import restype_str_to_int

    logger = configure_logging("train.ligandmpnn_generator", level=args.log_level)
    random.seed(args.seed + args.round_id)
    torch.manual_seed(args.seed + args.round_id)

    start_ckpt = args.previous_checkpoint or args.base_checkpoint
    model, checkpoint, torch_device, atom_context_num, ckpt_path = load_ligandmpnn_sequence_model(
        repo_root=root,
        ligandmpnn_root=args.ligandmpnn_root,
        checkpoint_path=start_ckpt,
        device=args.device,
        model_type=args.model_type,
        use_side_chain_context=args.use_side_chain_context,
    )
    trainable_names = set_trainable_scope(model, args.train_scope)
    model.train()
    logger.info(
        "LigandMPNN generator training start: round=%d checkpoint=%s device=%s scope=%s trainable_tensors=%d",
        args.round_id,
        ckpt_path,
        torch_device,
        args.train_scope,
        len(trainable_names),
    )

    rows_all = read_records(root / args.input_dr)
    rows = [
        r for r in rows_all
        if str(r.get("split", args.split)) == str(args.split) and r.get(args.sequence_key)
    ]
    if not rows:
        rows = [r for r in rows_all if r.get(args.sequence_key)]
    if args.max_records and args.max_records > 0:
        labeled = [r for r in rows if r.get("reward") is not None]
        baseline = [r for r in rows if r.get("reward") is None]
        rows = (sorted(labeled, key=lambda r: float(r.get("reward") or 0.0), reverse=True) + baseline)[: args.max_records]
    if not rows:
        raise RuntimeError(f"No trainable LigandMPNN records found in {args.input_dr}")

    if args.objective == "trajectory_balance":
        gfn_entries, seed_records = build_gflownet_trajectory_entries(
            rows,
            max_mutations=args.tb_max_mutations,
            reward_floor=args.reward_floor,
            reward_clip_min=args.reward_clip_min,
            reward_clip_max=args.reward_clip_max,
        )
        if not gfn_entries:
            raise RuntimeError(
                f"No GFlowNet trajectories could be reconstructed from {args.input_dr}; "
                f"check sequence lengths and tb_max_mutations={args.tb_max_mutations}"
            )
    else:
        gfn_entries = []
        seed_records = {}
        raw_weights = [
            compute_record_weight(
                r,
                mode=args.reward_weight_mode,
                baseline_weight=args.baseline_weight,
                reward_power=args.reward_power,
                score_temperature=args.score_temperature,
                max_weight=args.max_weight,
            )
            for r in rows
        ]
        weights = normalize_weights(raw_weights)

    feature_root = (root / args.output_dir) / f"ligandmpnn_feature_cache_round_{args.round_id}"
    feature_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    feature_by_seed: dict[str, dict] = {}
    skipped: list[dict] = []
    if args.objective == "trajectory_balance":
        feature_items = [(key, rec, None) for key, rec in sorted(seed_records.items())]
    else:
        feature_items = [
            (str(rec.get("candidate_id") or rec.get("backbone_id") or i), rec, float(weights[i]))
            for i, rec in enumerate(rows)
        ]
    for key, rec, weight in iter_progress(
        feature_items,
        total=len(feature_items),
        desc="ligandmpnn:features",
        no_progress=args.no_progress,
    ):
        try:
            feature_dict, _protein_dict, design_idx = make_ligandmpnn_feature_dict(
                repo_root=root,
                record=rec,
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
                batch_size=args.batch_size,
                temperature=args.temperature,
            )
            design_mask = make_design_chain_mask(feature_dict, design_idx)
            feature_payload = {
                "record": rec,
                "feature": feature_dict,
                "design_idx": design_idx,
                "design_mask": design_mask,
                "design_len": len(design_idx),
            }
            if weight is not None:
                feature_payload["weight"] = float(weight)
            feature_by_seed[key] = feature_payload
            if args.objective != "trajectory_balance":
                entries.append(feature_payload)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"candidate_id": rec.get("candidate_id"), "error": str(exc)})
            logger.warning("Skipping LigandMPNN training record candidate_id=%s error=%s", rec.get("candidate_id"), exc)
    if args.objective == "trajectory_balance":
        entries = [entry for entry in gfn_entries if entry["seed_key"] in feature_by_seed]
    if not entries:
        raise RuntimeError(f"No records could be featurized for LigandMPNN training; skipped={skipped[:5]}")
    priority_entries = [
        entry
        for entry in entries
        if args.objective == "trajectory_balance"
        and entry.get("record", {}).get("reward") is not None
        and len(entry.get("mutations", [])) > 0
    ]

    model_params = [p for p in model.parameters() if p.requires_grad]
    gfn_state_in = checkpoint.get("thermogfn_gflownet_state", {}) if isinstance(checkpoint, dict) else {}
    stop_bias = torch.nn.Parameter(
        torch.tensor(float(gfn_state_in.get("stop_bias", args.stop_init_logit)), dtype=torch.float32, device=torch_device)
    )
    stop_len_weight = torch.nn.Parameter(
        torch.tensor(float(gfn_state_in.get("stop_len_weight", args.stop_len_weight_init)), dtype=torch.float32, device=torch_device)
    )
    seed_ids = sorted({str(entry["seed_key"]) for entry in entries}) if args.objective == "trajectory_balance" else []
    seed_index = {seed_id: i for i, seed_id in enumerate(seed_ids)}
    prev_log_z = gfn_state_in.get("log_z_by_seed", {}) if isinstance(gfn_state_in, dict) else {}
    log_z_init = [
        float(prev_log_z.get(seed_id, args.log_z_init)) if isinstance(prev_log_z, dict) else float(args.log_z_init)
        for seed_id in seed_ids
    ]
    log_z = torch.nn.Parameter(torch.tensor(log_z_init, dtype=torch.float32, device=torch_device)) if seed_ids else None
    extra_params = [stop_bias, stop_len_weight] + ([log_z] if log_z is not None else [])
    params = model_params + (extra_params if args.objective == "trajectory_balance" else [])
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    anchor_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and float(args.anchor_l2) > 0.0
    }

    aa_to_idx = {aa: int(restype_str_to_int[aa]) for aa in AMINO_ACIDS}

    def _anchor_loss() -> torch.Tensor:
        if not anchor_params:
            return torch.zeros((), device=torch_device)
        anchor_terms = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in anchor_params:
                anchor_terms.append(torch.mean((param - anchor_params[name]) ** 2))
        if not anchor_terms:
            return torch.zeros((), device=torch_device)
        return torch.stack(anchor_terms).mean()

    def _supervised_nll(feature_payload: dict, target_sequence: str) -> torch.Tensor:
        feature_dict = feature_payload["feature"]
        design_idx = feature_payload["design_idx"]
        set_design_sequence(feature_dict, target_sequence, design_idx)
        feature_dict["chain_mask"] = feature_payload["design_mask"]
        length = int(feature_dict["S"].shape[1])
        feature_dict["batch_size"] = 1
        feature_dict["randn"] = torch.randn((1, length), dtype=torch.float32, device=torch_device)
        out = model.score(feature_dict, use_sequence=True)
        avg_loss, _loss_per_residue = get_score(out["S"], out["log_probs"], feature_dict["mask"] * feature_dict["chain_mask"])
        return avg_loss.mean()

    def _action_logits_for_state(
        *,
        feature_payload: dict,
        current_seq: list[str],
        edited: set[int],
        allowed_positions: list[int],
        step_idx: int,
    ) -> tuple[torch.Tensor, list[tuple[str, int | None, str | None]]]:
        feature_dict = feature_payload["feature"]
        design_idx = feature_payload["design_idx"]
        set_design_sequence(feature_dict, "".join(current_seq), design_idx)
        feature_dict["chain_mask"] = feature_payload["design_mask"]
        length = int(feature_dict["S"].shape[1])
        feature_dict["batch_size"] = 1
        feature_dict["randn"] = torch.randn((1, length), dtype=torch.float32, device=torch_device)
        out = model.score(feature_dict, use_sequence=True)
        residue_log_probs = out["log_probs"][0]
        seq_len = max(len(current_seq), 1)
        stop_logit = stop_bias + stop_len_weight * torch.tensor(float(step_idx) / float(seq_len), device=torch_device)
        logits = [stop_logit]
        labels: list[tuple[str, int | None, str | None]] = [("STOP", None, None)]
        for pos in allowed_positions:
            if pos in edited or pos < 0 or pos >= len(current_seq) or pos >= len(design_idx):
                continue
            old_aa = current_seq[pos]
            full_idx = design_idx[pos]
            for aa in AMINO_ACIDS:
                if aa == old_aa:
                    continue
                logits.append(residue_log_probs[full_idx, aa_to_idx[aa]])
                labels.append(("EDIT", pos, aa))
        return torch.stack(logits), labels

    def _trajectory_balance_loss(entry: dict) -> tuple[torch.Tensor, dict]:
        if log_z is None:
            raise RuntimeError("log_z table was not initialized for trajectory-balance training")
        feature_payload = feature_by_seed[str(entry["seed_key"])]
        current_seq = list(str(entry["seed_sequence"]))
        edited: set[int] = set()
        allowed_positions = list(entry["allowed_positions"])
        log_pf = torch.zeros((), dtype=torch.float32, device=torch_device)
        for step_idx, (pos, new_aa) in enumerate(entry["mutations"]):
            logits, labels = _action_logits_for_state(
                feature_payload=feature_payload,
                current_seq=current_seq,
                edited=edited,
                allowed_positions=allowed_positions,
                step_idx=step_idx,
            )
            selected_idx = None
            for idx, label in enumerate(labels):
                if label == ("EDIT", int(pos), str(new_aa)):
                    selected_idx = idx
                    break
            if selected_idx is None:
                raise RuntimeError(f"target edit not in LigandMPNN action space: pos={pos} aa={new_aa}")
            log_pf = log_pf + torch.log_softmax(logits, dim=0)[selected_idx]
            current_seq[int(pos)] = str(new_aa)
            edited.add(int(pos))
        logits, _labels = _action_logits_for_state(
            feature_payload=feature_payload,
            current_seq=current_seq,
            edited=edited,
            allowed_positions=allowed_positions,
            step_idx=len(entry["mutations"]),
        )
        log_pf = log_pf + torch.log_softmax(logits, dim=0)[0]
        k = len(entry["mutations"])
        log_pb = -torch.lgamma(torch.tensor(float(k + 1), dtype=torch.float32, device=torch_device))
        log_reward = torch.log(torch.tensor(float(entry["reward"]), dtype=torch.float32, device=torch_device))
        delta = log_z[seed_index[str(entry["seed_key"])]] + log_pf - log_reward - log_pb
        tb_loss = delta.pow(2)
        sup_loss = _supervised_nll(feature_payload, str(entry["target_sequence"])) if args.supervised_nll_weight > 0 else torch.zeros((), device=torch_device)
        return tb_loss + float(args.supervised_nll_weight) * sup_loss, {
            "tb_loss": float(tb_loss.detach().cpu()),
            "tb_delta": float(delta.detach().cpu()),
            "tb_delta_abs": float(delta.detach().abs().cpu()),
            "log_pf": float(log_pf.detach().cpu()),
            "log_pb": float(log_pb.detach().cpu()),
            "log_reward": float(log_reward.detach().cpu()),
            "reward": float(entry["reward"]),
            "nll": float(sup_loss.detach().cpu()),
            "K": int(k),
        }

    rng = random.Random(args.seed + args.round_id)
    history_rows: list[dict] = []
    pbar_desc = "ligandmpnn:gfn_tb" if args.objective == "trajectory_balance" else "ligandmpnn:train"
    pbar = make_progress(total=args.steps, desc=pbar_desc, no_progress=args.no_progress, leave=True, unit="step")
    final_row: dict = {}
    for step in range(1, int(args.steps) + 1):
        if (
            args.objective == "trajectory_balance"
            and priority_entries
            and rng.random() < max(0.0, min(1.0, float(args.tb_labeled_fraction)))
        ):
            entry = priority_entries[rng.randrange(len(priority_entries))]
        else:
            entry = entries[rng.randrange(len(entries))]
        optimizer.zero_grad(set_to_none=True)
        if args.objective == "trajectory_balance":
            loss_main, train_stats = _trajectory_balance_loss(entry)
            anchor_loss = _anchor_loss()
            loss = loss_main + float(args.anchor_l2) * anchor_loss
        else:
            feature_dict = entry["feature"]
            length = int(feature_dict["S"].shape[1])
            feature_dict["randn"] = torch.randn((int(args.batch_size), length), dtype=torch.float32, device=torch_device)
            out = model.score(feature_dict, use_sequence=True)
            mask = feature_dict["mask"] * feature_dict["chain_mask"]
            avg_loss, _loss_per_residue = get_score(out["S"], out["log_probs"], mask)
            nll = avg_loss.mean()
            anchor_loss = _anchor_loss()
            loss = nll * float(entry.get("weight", 1.0)) + float(args.anchor_l2) * anchor_loss
            train_stats = {
                "tb_loss": 0.0,
                "tb_delta": 0.0,
                "tb_delta_abs": 0.0,
                "log_pf": 0.0,
                "log_pb": 0.0,
                "log_reward": 0.0,
                "reward": 0.0,
                "nll": float(nll.detach().cpu()),
                "K": int(entry["record"].get("K", 0) or 0),
            }
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
        optimizer.step()

        pbar.update(1)
        if step == 1 or step == args.steps or (args.metrics_every > 0 and step % args.metrics_every == 0):
            row = {
                "round_id": args.round_id,
                "step": step,
                "objective": args.objective,
                "loss": float(loss.detach().cpu()),
                "tb_loss": float(train_stats["tb_loss"]),
                "tb_delta": float(train_stats["tb_delta"]),
                "tb_delta_abs": float(train_stats["tb_delta_abs"]),
                "log_pf": float(train_stats["log_pf"]),
                "log_pb": float(train_stats["log_pb"]),
                "log_reward": float(train_stats["log_reward"]),
                "reward": float(train_stats["reward"]),
                "nll": float(train_stats["nll"]),
                "K": int(train_stats["K"]),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "record_weight": float(entry.get("weight", 1.0)),
                "grad_norm_pre_clip": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "candidate_id": str(entry.get("record", {}).get("candidate_id", "")),
            }
            if args.objective == "trajectory_balance":
                row.update(
                    {
                        "seed_key": str(entry.get("seed_key", "")),
                        "stop_bias": float(stop_bias.detach().cpu()),
                        "stop_len_weight": float(stop_len_weight.detach().cpu()),
                        "mean_log_z": float(log_z.detach().mean().cpu()) if log_z is not None and log_z.numel() else 0.0,
                    }
                )
            history_rows.append(row)
            final_row = row
            logger.info(
                "ligandmpnn %s step=%d loss=%.6f tb=%.6f delta=%.4f nll=%.6f K=%d anchor=%.6f grad=%.4f",
                args.objective,
                step,
                row["loss"],
                row["tb_loss"],
                row["tb_delta"],
                row["nll"],
                row["K"],
                row["anchor_loss"],
                row["grad_norm_pre_clip"],
            )
            pbar.set_postfix_str(f"loss={row['loss']:.4f} tb={row['tb_loss']:.4f} nll={row['nll']:.4f}")
    pbar.close()

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_ckpt = outdir / (
        f"ligandmpnn_gflownet_round_{args.round_id}.pt"
        if args.objective == "trajectory_balance"
        else f"ligandmpnn_generator_round_{args.round_id}.pt"
    )
    checkpoint_out = dict(checkpoint)
    checkpoint_out["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint_out["thermogfn_generator_training"] = {
        "round_id": args.round_id,
        "source_checkpoint": str(ckpt_path),
        "model_type": args.model_type,
        "objective": args.objective,
        "train_scope": args.train_scope,
        "steps": int(args.steps),
        "n_records_input": len(rows),
        "n_records_featurized": len(entries),
        "n_records_skipped": len(skipped),
        "reward_weight_mode": args.reward_weight_mode,
        "baseline_weight": float(args.baseline_weight),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "anchor_l2": float(args.anchor_l2),
        "supervised_nll_weight": float(args.supervised_nll_weight),
    }
    if args.objective == "trajectory_balance":
        log_z_cpu = log_z.detach().cpu() if log_z is not None else None
        checkpoint_out["thermogfn_gflownet_state"] = {
            "objective": "trajectory_balance",
            "stop_bias": float(stop_bias.detach().cpu()),
            "stop_len_weight": float(stop_len_weight.detach().cpu()),
            "seed_ids": seed_ids,
            "log_z_by_seed": {
                seed_id: float(log_z.detach().cpu()[idx]) if log_z is not None else float(args.log_z_init)
                for seed_id, idx in seed_index.items()
            },
        }
    torch.save(checkpoint_out, str(out_ckpt))
    prune_round_checkpoints(outdir, prefix="ligandmpnn_generator", max_keep=args.max_checkpoints, logger=logger)

    metrics = {
        "round_id": args.round_id,
        "checkpoint": str(out_ckpt),
        "source_checkpoint": str(ckpt_path),
        "model_type": args.model_type,
        "objective": args.objective,
        "train_scope": args.train_scope,
        "steps": int(args.steps),
        "n_records_input": len(rows),
        "n_records_featurized": len(entries),
        "n_gflownet_trajectories": len(entries) if args.objective == "trajectory_balance" else 0,
        "n_priority_gflownet_trajectories": len(priority_entries) if args.objective == "trajectory_balance" else 0,
        "n_seed_records": len(seed_ids),
        "n_records_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": int(sum(p.numel() for p in model_params)),
        "gflownet_parameter_count": int(sum(p.numel() for p in extra_params)) if args.objective == "trajectory_balance" else 0,
        "gflownet_state": {
            "stop_bias": float(stop_bias.detach().cpu()) if args.objective == "trajectory_balance" else None,
            "stop_len_weight": float(stop_len_weight.detach().cpu()) if args.objective == "trajectory_balance" else None,
            "log_z_mean": float(log_z_cpu.mean()) if args.objective == "trajectory_balance" and log_z_cpu is not None and log_z_cpu.numel() else None,
            "log_z_std": float(log_z_cpu.std(unbiased=False)) if args.objective == "trajectory_balance" and log_z_cpu is not None and log_z_cpu.numel() else None,
            "log_z_min": float(log_z_cpu.min()) if args.objective == "trajectory_balance" and log_z_cpu is not None and log_z_cpu.numel() else None,
            "log_z_max": float(log_z_cpu.max()) if args.objective == "trajectory_balance" and log_z_cpu is not None and log_z_cpu.numel() else None,
        },
        "final": final_row,
        "elapsed_s": time.perf_counter() - t0,
    }
    write_json(outdir / "ligandmpnn_generator_metrics.json", metrics)
    if args.history_path:
        write_jsonl(root / args.history_path, history_rows)
    else:
        write_jsonl(outdir / f"ligandmpnn_generator_history_round_{args.round_id}.jsonl", history_rows)
    logger.info("LigandMPNN generator checkpoint written: %s elapsed=%.2fs", out_ckpt, time.perf_counter() - t0)
    print(out_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
