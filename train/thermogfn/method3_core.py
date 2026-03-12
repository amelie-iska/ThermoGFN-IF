"""Method III core utilities: surrogate, trajectory-balance teacher, student, and candidate generation."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .constants import AMINO_ACIDS
from .features import candidate_feature_vector, deterministic_hash
from .progress import iter_progress


FEATURE_ORDER = ["length", "K", "prepared_atom_count"] + [f"frac_{aa}" for aa in AMINO_ACIDS]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
EPS = 1e-8


def _vec_from_record(rec: dict) -> np.ndarray:
    feats = candidate_feature_vector(rec)
    return np.array([float(feats.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)


def _bootstrap_indices(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(n) for _ in range(n)]


def fit_surrogate_ensemble(
    records: list[dict],
    ensemble_size: int,
    seed: int,
    *,
    show_progress: bool = False,
    progress_desc: str = "surrogate:bootstrap",
    metrics_callback=None,
) -> dict:
    labeled = [r for r in records if r.get("reward") is not None]
    if not labeled:
        return {"models": [], "feature_order": FEATURE_ORDER, "prior_mean": 1.0, "prior_std": 0.2}

    x = np.stack([_vec_from_record(r) for r in labeled], axis=0)
    y = np.array([float(r.get("reward", 1.0)) for r in labeled], dtype=float)

    rng = random.Random(seed)
    models: list[dict[str, Any]] = []
    for model_idx in iter_progress(
        range(ensemble_size),
        total=ensemble_size,
        desc=progress_desc,
        no_progress=not show_progress,
    ):
        idx = _bootstrap_indices(len(labeled), rng)
        xb = x[idx]
        yb = y[idx]
        xb_aug = np.concatenate([xb, np.ones((xb.shape[0], 1))], axis=1)
        coef, *_ = np.linalg.lstsq(xb_aug, yb, rcond=None)
        models.append({"coef": coef.tolist()})
        if metrics_callback is not None:
            metrics_callback(
                {
                    "step": int(model_idx) + 1,
                    "bootstrap_idx": int(model_idx),
                    "bootstrap_size": int(len(idx)),
                    "target_mean": float(np.mean(yb)),
                    "target_std": float(np.std(yb) if len(yb) > 1 else 0.0),
                    "coef_l2": float(np.linalg.norm(coef)),
                }
            )

    return {
        "models": models,
        "feature_order": FEATURE_ORDER,
        "prior_mean": float(np.mean(y)),
        "prior_std": float(np.std(y) if len(y) > 1 else 0.2),
    }


def predict_surrogate(surrogate: dict, rec: dict) -> tuple[float, float]:
    models = surrogate.get("models", [])
    if not models:
        mean = surrogate.get("prior_mean", 1.0)
        std = surrogate.get("prior_std", 0.2)
        return float(mean), float(std)

    x = _vec_from_record(rec)
    x_aug = np.concatenate([x, np.ones(1)], axis=0)
    preds = []
    for m in models:
        coef = np.array(m["coef"], dtype=float)
        preds.append(float(np.dot(coef, x_aug)))
    return float(np.mean(preds)), float(np.std(preds))


def _safe_log(x: float) -> float:
    return math.log(max(float(x), EPS))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    out = np.where(x_arr >= 0.0, 1.0 / (1.0 + np.exp(-x_arr)), np.exp(x_arr) / (1.0 + np.exp(x_arr)))
    if np.isscalar(x):
        return float(out)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros_like(logits, dtype=float)
    z = logits - float(np.max(logits))
    exp_z = np.exp(z)
    denom = max(float(np.sum(exp_z)), EPS)
    return exp_z / denom


def _sample_discrete(prob_map: dict[int, float] | dict[str, float], rng: random.Random):
    keys = list(prob_map.keys())
    vals = np.array([max(float(prob_map[k]), 0.0) for k in keys], dtype=float)
    total = float(np.sum(vals))
    if total <= 0.0:
        vals = np.ones(len(keys), dtype=float) / max(len(keys), 1)
    else:
        vals = vals / total
    idx = int(np.searchsorted(np.cumsum(vals), rng.random(), side="right"))
    idx = min(idx, len(keys) - 1)
    return keys[idx]


def _seed_key(rec: dict) -> str:
    return str(rec.get("seed_id") or rec.get("backbone_id") or rec.get("candidate_id") or "")


def _build_seed_records(records: list[dict]) -> dict[str, dict]:
    best: dict[str, tuple[tuple[int, int, str], dict]] = {}
    for rec in records:
        seq = str(rec.get("sequence", ""))
        if not seq:
            continue
        key = _seed_key(rec)
        source = str(rec.get("source", ""))
        score = (
            0 if source == "baseline" else 1,
            int(rec.get("K", 0)),
            str(rec.get("candidate_id", "")),
        )
        current = best.get(key)
        if current is None or score < current[0]:
            best[key] = (score, rec)
    return {k: v[1] for k, v in best.items()}


def _reconstruct_mutations(seed_seq: str, target_seq: str) -> list[tuple[int, str]]:
    if len(seed_seq) != len(target_seq):
        raise ValueError("Canonical edit reconstruction requires equal-length sequences")
    muts: list[tuple[int, str]] = []
    for pos, (old, new) in enumerate(zip(seed_seq, target_seq)):
        if old != new:
            muts.append((pos, new))
    return muts


def _mutation_count_prior(max_k: int) -> np.ndarray:
    raw = np.array([math.exp(-0.35 * k) for k in range(max_k + 1)], dtype=float)
    boosts = {
        0: 0.28,
        1: 1.50,
        2: 1.75,
        3: 1.45,
        4: 1.20,
        5: 1.00,
        6: 0.85,
        7: 0.75,
    }
    for k, factor in boosts.items():
        if k <= max_k:
            raw[k] *= factor
    raw = np.clip(raw, 1e-6, None)
    return raw / max(float(np.sum(raw)), EPS)


def _terminal_to_stop_probs(k_probs: np.ndarray) -> np.ndarray:
    tail = float(np.sum(k_probs))
    stop = np.zeros_like(k_probs, dtype=float)
    for k, pk in enumerate(k_probs):
        if tail <= EPS:
            stop[k] = 1.0
        else:
            stop[k] = min(max(float(pk) / tail, 1e-4), 1.0 - 1e-4)
        tail -= float(pk)
    if stop.size:
        stop[-1] = min(max(stop[-1], 0.5), 1.0 - 1e-4)
    return stop


def _init_teacher_params(seed_records: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seed_ids = list(seed_records.keys())
    if not seed_ids:
        raise ValueError("Cannot initialize teacher without seed records")
    max_len = max(len(str(seed_records[s]["sequence"])) for s in seed_ids)
    seed_lengths = {seed_id: len(str(seed_records[seed_id]["sequence"])) for seed_id in seed_ids}
    prior_k = _mutation_count_prior(max_len)
    prior_stop = _terminal_to_stop_probs(prior_k)
    stop_logits = np.log(prior_stop / np.clip(1.0 - prior_stop, EPS, None))
    params = {
        "log_z": np.zeros(len(seed_ids), dtype=float),
        "stop_logits": stop_logits.astype(float),
        "pos_logits": np.zeros((len(seed_ids), max_len), dtype=float),
        "aa_logits": np.zeros((len(AMINO_ACIDS), len(AMINO_ACIDS)), dtype=float),
    }
    meta = {
        "seed_ids": seed_ids,
        "seed_index": {seed_id: i for i, seed_id in enumerate(seed_ids)},
        "seed_lengths": seed_lengths,
        "max_len": max_len,
        "prior_stop_probs": prior_stop,
        "prior_k_probs": prior_k,
    }
    return params, meta


def _zero_like_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.zeros_like(v, dtype=float) for k, v in params.items()}


def _masked_aa_probs(params: dict[str, np.ndarray], old_idx: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(AMINO_ACIDS), dtype=bool)
    mask[old_idx] = False
    probs = _softmax(params["aa_logits"][old_idx, mask])
    return mask, probs


def _trajectory_tb_grad(
    params: dict[str, np.ndarray],
    seed_idx: int,
    seed_seq: str,
    mutations: list[tuple[int, str]],
    reward: float,
) -> tuple[float, dict[str, np.ndarray], float]:
    grads = _zero_like_params(params)
    log_pf = 0.0
    step = 0
    last_pos = -1
    seq_len = len(seed_seq)

    for pos, new_aa in mutations:
        stop_logit = float(params["stop_logits"][min(step, len(params["stop_logits"]) - 1)])
        p_stop = float(_sigmoid(stop_logit))
        log_pf += _safe_log(1.0 - p_stop)
        grads["stop_logits"][min(step, len(params["stop_logits"]) - 1)] += -p_stop

        eligible = np.arange(last_pos + 1, seq_len, dtype=int)
        pos_probs = _softmax(params["pos_logits"][seed_idx, eligible])
        rel_idx = int(pos - (last_pos + 1))
        log_pf += _safe_log(float(pos_probs[rel_idx]))
        grads["pos_logits"][seed_idx, eligible] -= pos_probs
        grads["pos_logits"][seed_idx, pos] += 1.0

        old_aa = seed_seq[pos]
        old_idx = AA_TO_IDX[old_aa]
        new_idx = AA_TO_IDX[new_aa]
        mask, aa_probs = _masked_aa_probs(params, old_idx)
        masked_indices = np.flatnonzero(mask)
        chosen = int(np.where(masked_indices == new_idx)[0][0])
        log_pf += _safe_log(float(aa_probs[chosen]))
        grads["aa_logits"][old_idx, masked_indices] -= aa_probs
        grads["aa_logits"][old_idx, new_idx] += 1.0

        step += 1
        last_pos = pos

    if last_pos < seq_len - 1:
        stop_logit = float(params["stop_logits"][min(step, len(params["stop_logits"]) - 1)])
        p_stop = float(_sigmoid(stop_logit))
        log_pf += _safe_log(p_stop)
        grads["stop_logits"][min(step, len(params["stop_logits"]) - 1)] += 1.0 - p_stop

    delta = float(params["log_z"][seed_idx]) + log_pf - _safe_log(reward)
    scale = 2.0 * delta
    grads["log_z"][seed_idx] += scale
    grads["stop_logits"] *= scale
    grads["pos_logits"] *= scale
    grads["aa_logits"] *= scale
    return float(delta * delta), grads, delta


def _sample_position(weights: np.ndarray, rng: random.Random) -> int:
    probs = np.clip(weights.astype(float), 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        probs = np.ones_like(probs, dtype=float) / max(len(probs), 1)
    else:
        probs = probs / total
    idx = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
    return min(idx, len(probs) - 1)


def _sample_teacher_trajectory(
    params: dict[str, np.ndarray],
    meta: dict[str, Any],
    seed_rec: dict,
    rng: random.Random,
) -> dict[str, Any]:
    seed_key = _seed_key(seed_rec)
    seed_idx = int(meta["seed_index"][seed_key])
    seed_seq = str(seed_rec["sequence"])
    seq = list(seed_seq)
    step = 0
    last_pos = -1
    muts: list[str] = []
    mutation_pairs: list[tuple[int, str]] = []

    while last_pos < len(seed_seq) - 1 and step < len(seed_seq):
        stop_logit = float(params["stop_logits"][min(step, len(params["stop_logits"]) - 1)])
        p_stop = float(_sigmoid(stop_logit))
        if rng.random() < p_stop:
            break

        eligible = np.arange(last_pos + 1, len(seed_seq), dtype=int)
        rel_idx = _sample_position(_softmax(params["pos_logits"][seed_idx, eligible]), rng)
        pos = int(eligible[rel_idx])
        old_aa = seed_seq[pos]
        old_idx = AA_TO_IDX[old_aa]
        mask, aa_probs = _masked_aa_probs(params, old_idx)
        masked_indices = np.flatnonzero(mask)
        aa_idx = int(masked_indices[_sample_position(aa_probs, rng)])
        new_aa = AMINO_ACIDS[aa_idx]

        seq[pos] = new_aa
        muts.append(f"{old_aa}{pos+1}{new_aa}")
        mutation_pairs.append((pos, new_aa))
        last_pos = pos
        step += 1

    return {
        "sequence": "".join(seq),
        "mutations": muts,
        "mutation_pairs": mutation_pairs,
        "K": len(muts),
        "seed_id": seed_key,
        "backbone_id": seed_rec.get("backbone_id", seed_key),
        "task_type": seed_rec.get("task_type", "monomer"),
        "prepared_atom_count": int(seed_rec.get("prepared_atom_count", 0)),
    }


def _trajectory_record_from_sample(seed_rec: dict, sample: dict[str, Any]) -> dict[str, Any]:
    rec = dict(seed_rec)
    rec["sequence"] = sample["sequence"]
    rec["mutations"] = list(sample["mutations"])
    rec["K"] = int(sample["K"])
    rec["seed_id"] = sample["seed_id"]
    rec["backbone_id"] = sample["backbone_id"]
    rec["task_type"] = sample["task_type"]
    rec["prepared_atom_count"] = sample["prepared_atom_count"]
    return rec


def _apply_regularization(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    prior_stop_probs: np.ndarray,
    *,
    stop_reg: float = 0.05,
    weight_decay: float = 1e-4,
) -> float:
    reg_loss = 0.0
    stop_probs = _sigmoid(params["stop_logits"])
    stop_diff = stop_probs - prior_stop_probs
    reg_loss += float(stop_reg * np.mean(stop_diff**2))
    grads["stop_logits"] += stop_reg * (2.0 / max(len(stop_diff), 1)) * stop_diff * stop_probs * (1.0 - stop_probs)

    for key in ("log_z", "pos_logits", "aa_logits"):
        reg_loss += float(weight_decay * np.mean(params[key] ** 2))
        grads[key] += 2.0 * weight_decay * params[key]
    return reg_loss


def _gradient_norm(grads: dict[str, np.ndarray]) -> float:
    return float(math.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))


def _clip_grads(grads: dict[str, np.ndarray], max_norm: float) -> None:
    norm = _gradient_norm(grads)
    if norm <= max_norm or norm <= EPS:
        return
    scale = max_norm / norm
    for key in grads:
        grads[key] *= scale


def _positive_reward(value: float | None) -> float:
    if value is None:
        return 1.0
    return max(float(value), 1e-3)


def _build_off_policy_trajectories(records: list[dict], seed_records: dict[str, dict], meta: dict[str, Any]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for rec in records:
        seed_key = _seed_key(rec)
        seed_rec = seed_records.get(seed_key)
        if seed_rec is None:
            continue
        seed_seq = str(seed_rec.get("sequence", ""))
        target_seq = str(rec.get("sequence", ""))
        if not seed_seq or not target_seq or len(seed_seq) != len(target_seq):
            continue
        mutations = _reconstruct_mutations(seed_seq, target_seq)
        trajectories.append(
            {
                "seed_key": seed_key,
                "seed_idx": int(meta["seed_index"][seed_key]),
                "seed_seq": seed_seq,
                "mutations": mutations,
                "reward": _positive_reward(rec.get("reward")),
            }
        )
    return trajectories


def _sample_marginals(
    params: dict[str, np.ndarray],
    meta: dict[str, Any],
    seed_records: dict[str, dict],
    *,
    seed: int,
    n_samples: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    seed_ids = list(seed_records.keys())
    k_counts: Counter[int] = Counter()
    aa_counts: Counter[str] = Counter()
    aa_conditional: dict[str, Counter[str]] = {aa: Counter() for aa in AMINO_ACIDS}
    pos_counts: dict[str, list[float]] = {
        seed_id: [0.0] * len(str(seed_records[seed_id]["sequence"])) for seed_id in seed_ids
    }

    for i in range(max(n_samples, len(seed_ids))):
        seed_id = seed_ids[i % len(seed_ids)]
        sample = _sample_teacher_trajectory(params, meta, seed_records[seed_id], rng)
        k_counts[int(sample["K"])] += 1
        for pos, new_aa in sample["mutation_pairs"]:
            old_aa = str(seed_records[seed_id]["sequence"])[pos]
            pos_counts[seed_id][pos] += 1.0
            aa_counts[new_aa] += 1
            aa_conditional[old_aa][new_aa] += 1

    total_k = max(sum(k_counts.values()), 1)
    k_probs = {str(k): float(v) / total_k for k, v in sorted(k_counts.items())}
    total_aa = max(sum(aa_counts.values()), 1)
    aa_probs = {aa: float(aa_counts.get(aa, 0)) / total_aa for aa in AMINO_ACIDS}

    aa_conditional_probs: dict[str, dict[str, float]] = {}
    for old_aa in AMINO_ACIDS:
        row = {new_aa: float(aa_conditional[old_aa].get(new_aa, 0.0)) for new_aa in AMINO_ACIDS if new_aa != old_aa}
        total = sum(row.values())
        if total <= 0.0:
            uniform = 1.0 / max(len(AMINO_ACIDS) - 1, 1)
            aa_conditional_probs[old_aa] = {new_aa: uniform for new_aa in AMINO_ACIDS if new_aa != old_aa}
        else:
            aa_conditional_probs[old_aa] = {new_aa: val / total for new_aa, val in row.items()}

    position_probs_by_seed: dict[str, list[float]] = {}
    for seed_id, counts in pos_counts.items():
        total = float(sum(counts))
        if total <= 0.0:
            n = max(len(counts), 1)
            position_probs_by_seed[seed_id] = [1.0 / n] * len(counts)
        else:
            position_probs_by_seed[seed_id] = [c / total for c in counts]

    return {
        "k_probs": k_probs,
        "aa_probs": aa_probs,
        "aa_conditional": aa_conditional_probs,
        "position_probs_by_seed": position_probs_by_seed,
    }


def train_teacher_policy(
    records: list[dict],
    seed: int,
    *,
    steps: int = 30000,
    gamma_off: float = 0.5,
    surrogate: dict | None = None,
    show_progress: bool = False,
    progress_desc: str = "teacher:tb",
    metrics_every: int = 25,
    metrics_callback=None,
) -> dict:
    seed_records = _build_seed_records(records)
    params, meta = _init_teacher_params(seed_records)
    off_trajs = _build_off_policy_trajectories(records, seed_records, meta)

    rng = random.Random(seed)
    n_steps = max(int(steps), 1)
    gamma_off = min(max(float(gamma_off), 0.0), 1.0)
    off_batch_size = min(max(len(off_trajs), 1), 64)
    on_batch_size = 0 if surrogate is None else min(max(len(seed_records), 1), 32)

    metrics = {
        "mean_loss": 0.0,
        "mean_off_loss": 0.0,
        "mean_on_loss": 0.0,
        "mean_delta": 0.0,
        "n_off_trajectories": len(off_trajs),
        "n_seed_records": len(seed_records),
    }
    running_steps = 0

    for step_idx in iter_progress(
        range(n_steps),
        total=n_steps,
        desc=progress_desc,
        no_progress=not show_progress,
    ):
        grads = _zero_like_params(params)
        total_loss = 0.0
        total_delta = 0.0
        off_loss = 0.0
        on_loss = 0.0

        if off_trajs and gamma_off > 0.0:
            for _ in range(off_batch_size):
                traj = off_trajs[rng.randrange(len(off_trajs))]
                loss_i, grad_i, delta_i = _trajectory_tb_grad(
                    params,
                    int(traj["seed_idx"]),
                    str(traj["seed_seq"]),
                    list(traj["mutations"]),
                    _positive_reward(traj.get("reward")),
                )
                weight = gamma_off / max(off_batch_size, 1)
                for key in grads:
                    grads[key] += weight * grad_i[key]
                off_loss += weight * loss_i
                total_delta += weight * abs(delta_i)

        if surrogate is not None and on_batch_size > 0 and gamma_off < 1.0:
            seed_ids = list(seed_records.keys())
            for _ in range(on_batch_size):
                seed_id = seed_ids[rng.randrange(len(seed_ids))]
                sample = _sample_teacher_trajectory(params, meta, seed_records[seed_id], rng)
                rec = _trajectory_record_from_sample(seed_records[seed_id], sample)
                mean, std = predict_surrogate(surrogate, rec)
                reward = max(float(mean) - 0.25 * float(std), 1e-3)
                loss_i, grad_i, delta_i = _trajectory_tb_grad(
                    params,
                    int(meta["seed_index"][seed_id]),
                    str(seed_records[seed_id]["sequence"]),
                    list(sample["mutation_pairs"]),
                    reward,
                )
                weight = (1.0 - gamma_off) / max(on_batch_size, 1)
                for key in grads:
                    grads[key] += weight * grad_i[key]
                on_loss += weight * loss_i
                total_delta += weight * abs(delta_i)

        reg_loss = _apply_regularization(params, grads, meta["prior_stop_probs"])
        total_loss = off_loss + on_loss + reg_loss

        grad_norm_pre_clip = _gradient_norm(grads)
        _clip_grads(grads, max_norm=5.0)
        grad_norm_post_clip = _gradient_norm(grads)
        lr = 0.08 / math.sqrt(1.0 + float(step_idx) / 200.0)
        for key in params:
            params[key] -= lr * grads[key]

        metrics["mean_loss"] += total_loss
        metrics["mean_off_loss"] += off_loss
        metrics["mean_on_loss"] += on_loss
        metrics["mean_delta"] += total_delta
        running_steps += 1

        if metrics_callback is not None:
            step_no = int(step_idx) + 1
            if step_no == 1 or step_no == n_steps or (step_no % max(int(metrics_every), 1)) == 0:
                metrics_callback(
                    {
                        "step": step_no,
                        "loss": float(total_loss),
                        "off_loss": float(off_loss),
                        "on_loss": float(on_loss),
                        "reg_loss": float(reg_loss),
                        "delta_abs": float(total_delta),
                        "lr": float(lr),
                        "grad_norm_pre_clip": float(grad_norm_pre_clip),
                        "grad_norm_post_clip": float(grad_norm_post_clip),
                        "mean_stop_prob": float(np.mean(_sigmoid(params["stop_logits"]))),
                        "mean_log_z": float(np.mean(params["log_z"])),
                        "n_off_trajectories": int(len(off_trajs)),
                        "n_seed_records": int(len(seed_records)),
                    }
                )

    if running_steps:
        for key in ("mean_loss", "mean_off_loss", "mean_on_loss", "mean_delta"):
            metrics[key] /= float(running_steps)

    marginals = _sample_marginals(
        params,
        meta,
        seed_records,
        seed=seed + 997,
        n_samples=max(512, min(4096, len(seed_records) * 128)),
    )

    return {
        "teacher_mode": "trajectory_balance_gflownet",
        "implementation_note": "trajectory_balance_teacher_over_canonical_edit_dag",
        "is_true_gflownet": True,
        "seed": seed,
        "temperature_range": [0.75, 1.75],
        "seed_ids": list(meta["seed_ids"]),
        "seed_lengths": dict(meta["seed_lengths"]),
        "params": {
            "log_z": params["log_z"].tolist(),
            "stop_logits": params["stop_logits"].tolist(),
            "pos_logits": params["pos_logits"].tolist(),
            "aa_logits": params["aa_logits"].tolist(),
        },
        "prior_k_probs": meta["prior_k_probs"].tolist(),
        "prior_stop_probs": meta["prior_stop_probs"].tolist(),
        "training_metrics": metrics,
        **marginals,
    }


def _teacher_params_and_meta(teacher: dict, seed_records: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    params = {
        "log_z": np.array(teacher["params"]["log_z"], dtype=float),
        "stop_logits": np.array(teacher["params"]["stop_logits"], dtype=float),
        "pos_logits": np.array(teacher["params"]["pos_logits"], dtype=float),
        "aa_logits": np.array(teacher["params"]["aa_logits"], dtype=float),
    }
    seed_ids = list(teacher.get("seed_ids", []))
    seed_lengths = teacher.get("seed_lengths", {})
    return params, {
        "seed_ids": seed_ids,
        "seed_index": {seed_id: i for i, seed_id in enumerate(seed_ids)},
        "seed_lengths": seed_lengths,
    }


def distill_student_from_teacher(
    teacher: dict,
    records: list[dict],
    seed: int,
    *,
    steps: int = 15000,
    show_progress: bool = False,
    progress_desc: str = "student:distill",
    metrics_every: int = 100,
    metrics_callback=None,
) -> dict:
    seed_records = _build_seed_records(records)
    params, meta = _teacher_params_and_meta(teacher, seed_records)
    rng = random.Random(seed)
    n_samples = max(512, min(max(int(steps), 1), 4096))

    k_counts: Counter[int] = Counter()
    aa_counts: Counter[str] = Counter()
    aa_conditional_counts: dict[str, Counter[str]] = {aa: Counter() for aa in AMINO_ACIDS}
    pos_counts: dict[str, list[float]] = {
        seed_id: [0.0] * len(str(seed_records[seed_id]["sequence"])) for seed_id in seed_records
    }

    seed_ids = list(seed_records.keys())
    for i in iter_progress(
        range(n_samples),
        total=n_samples,
        desc=progress_desc,
        no_progress=not show_progress,
    ):
        seed_id = seed_ids[i % len(seed_ids)]
        sample = _sample_teacher_trajectory(params, meta, seed_records[seed_id], rng)
        k_counts[int(sample["K"])] += 1
        for pos, new_aa in sample["mutation_pairs"]:
            old_aa = str(seed_records[seed_id]["sequence"])[pos]
            pos_counts[seed_id][pos] += 1.0
            aa_counts[new_aa] += 1
            aa_conditional_counts[old_aa][new_aa] += 1

        if metrics_callback is not None:
            step_no = int(i) + 1
            if step_no == 1 or step_no == n_samples or (step_no % max(int(metrics_every), 1)) == 0:
                total_k = max(sum(k_counts.values()), 1)
                mean_k = sum(float(k) * float(v) for k, v in k_counts.items()) / float(total_k)
                probs = [float(v) / float(total_k) for v in k_counts.values()]
                entropy = -sum(p * math.log(max(p, EPS)) for p in probs)
                metrics_callback(
                    {
                        "step": step_no,
                        "sampled_fraction": float(step_no) / float(max(n_samples, 1)),
                        "sampled_k_mean": float(mean_k),
                        "sampled_k_entropy": float(entropy),
                        "distinct_k": int(len(k_counts)),
                        "distinct_seed_ids": int(len({seed_ids[j % len(seed_ids)] for j in range(step_no)})),
                    }
                )

    teacher_k = {str(k): float(v) for k, v in teacher.get("k_probs", {}).items()}
    if teacher_k:
        k_probs = dict(teacher_k)
    else:
        total_sampled_k = max(sum(k_counts.values()), 1)
        k_probs = {str(k): float(v) / total_sampled_k for k, v in sorted(k_counts.items())}

    teacher_aa = {aa: float(teacher.get("aa_probs", {}).get(aa, 0.0)) for aa in AMINO_ACIDS}
    total_aa = max(sum(aa_counts.values()), 1)
    sampled_aa = {aa: float(aa_counts.get(aa, 0)) / total_aa for aa in AMINO_ACIDS}
    aa_probs = {}
    for aa in AMINO_ACIDS:
        aa_probs[aa] = 0.85 * teacher_aa.get(aa, 0.0) + 0.15 * sampled_aa.get(aa, 0.0)
    aa_norm = sum(aa_probs.values()) or 1.0
    aa_probs = {aa: v / aa_norm for aa, v in aa_probs.items()}

    aa_conditional: dict[str, dict[str, float]] = {}
    for old_aa in AMINO_ACIDS:
        teacher_row = {
            new_aa: float(teacher.get("aa_conditional", {}).get(old_aa, {}).get(new_aa, 0.0))
            for new_aa in AMINO_ACIDS
            if new_aa != old_aa
        }
        sampled_row = {
            new_aa: float(aa_conditional_counts[old_aa].get(new_aa, 0.0))
            for new_aa in AMINO_ACIDS
            if new_aa != old_aa
        }
        sampled_total = sum(sampled_row.values())
        if sampled_total > 0.0:
            sampled_row = {new_aa: val / sampled_total for new_aa, val in sampled_row.items()}
        teacher_total = sum(teacher_row.values())
        if teacher_total <= 0.0 and sampled_total <= 0.0:
            uniform = 1.0 / max(len(AMINO_ACIDS) - 1, 1)
            aa_conditional[old_aa] = {new_aa: uniform for new_aa in AMINO_ACIDS if new_aa != old_aa}
            continue
        if teacher_total > 0.0:
            teacher_row = {new_aa: val / teacher_total for new_aa, val in teacher_row.items()}
        row = {
            new_aa: 0.85 * teacher_row.get(new_aa, 0.0) + 0.15 * sampled_row.get(new_aa, 0.0)
            for new_aa in AMINO_ACIDS
            if new_aa != old_aa
        }
        total = sum(row.values()) or 1.0
        aa_conditional[old_aa] = {new_aa: val / total for new_aa, val in row.items()}

    position_probs_by_seed: dict[str, list[float]] = {}
    for seed_id, counts in pos_counts.items():
        total = float(sum(counts))
        if total <= 0.0:
            n = max(len(counts), 1)
            position_probs_by_seed[seed_id] = [1.0 / n] * len(counts)
        else:
            position_probs_by_seed[seed_id] = [c / total for c in counts]

    return {
        "teacher_mode": "trajectory_balance_gflownet",
        "implementation_note": "one_shot_student_distilled_from_tb_teacher",
        "is_true_gflownet": True,
        "seed": seed,
        "k_probs": k_probs,
        "aa_probs": aa_probs,
        "aa_conditional": aa_conditional,
        "position_probs_by_seed": position_probs_by_seed,
        "seed_lengths": {seed_id: len(str(seed_records[seed_id]["sequence"])) for seed_id in seed_records},
        "distill_samples": n_samples,
    }


def _sample_positions_without_replacement(
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    k = min(max(k, 0), len(weights))
    available = list(range(len(weights)))
    chosen: list[int] = []
    working = [max(float(w), 0.0) for w in weights]
    for _ in range(k):
        if not available:
            break
        probs = np.array([working[i] for i in available], dtype=float)
        if float(np.sum(probs)) <= 0.0:
            probs = np.ones(len(available), dtype=float) / max(len(available), 1)
        else:
            probs = probs / float(np.sum(probs))
        pick = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
        pick = min(pick, len(available) - 1)
        pos = available.pop(pick)
        chosen.append(pos)
    chosen.sort()
    return chosen


def _mutate_sequence_with_student(student: dict, seed_rec: dict, k: int, rng: random.Random) -> tuple[str, list[str]]:
    seed_seq = str(seed_rec.get("sequence", ""))
    if not seed_seq:
        return seed_seq, []
    seed_key = _seed_key(seed_rec)
    position_probs_by_seed = student.get("position_probs_by_seed", {})
    weights = list(position_probs_by_seed.get(seed_key, []))
    if len(weights) != len(seed_seq):
        n = max(len(seed_seq), 1)
        weights = [1.0 / n] * len(seed_seq)
    positions = _sample_positions_without_replacement(weights, k, rng)

    aa_probs = dict(student.get("aa_probs", {}))
    aa_conditional = student.get("aa_conditional", {})
    seq = list(seed_seq)
    muts: list[str] = []
    for pos in positions:
        old = seed_seq[pos]
        cond = aa_conditional.get(old, {})
        if cond:
            new = _sample_discrete(cond, rng)
        else:
            filtered = {aa: float(aa_probs.get(aa, 0.0)) for aa in AMINO_ACIDS if aa != old}
            new = _sample_discrete(filtered, rng)
        seq[pos] = str(new)
        muts.append(f"{old}{pos+1}{new}")
    return "".join(seq), muts


def generate_student_candidates(
    student: dict,
    seeds: list[dict],
    pool_size: int,
    run_id: str,
    round_id: int,
    seed: int,
    *,
    show_progress: bool = False,
    progress_desc: str = "student:generate",
) -> list[dict]:
    rng = random.Random(seed)
    if not seeds:
        return []
    k_probs = {int(k): float(v) for k, v in student.get("k_probs", {}).items()}
    if not k_probs:
        k_probs = {0: 0.08, 1: 0.24, 2: 0.26, 3: 0.20, 4: 0.12, 5: 0.10}

    records: list[dict] = []
    for i in iter_progress(
        range(pool_size),
        total=pool_size,
        desc=progress_desc,
        no_progress=not show_progress,
    ):
        seed_rec = seeds[i % len(seeds)]
        k = int(_sample_discrete(k_probs, rng))
        new_seq, muts = _mutate_sequence_with_student(student, seed_rec, k, rng)
        cid = deterministic_hash(f"{run_id}:{round_id}:{seed_rec['backbone_id']}:{new_seq}")
        rec = {
            "candidate_id": cid,
            "run_id": run_id,
            "round_id": round_id,
            "task_type": seed_rec.get("task_type", "monomer"),
            "backbone_id": seed_rec["backbone_id"],
            "seed_id": seed_rec.get("seed_id", seed_rec["backbone_id"]),
            "sequence": new_seq,
            "mutations": muts,
            "K": len(muts),
            "prepared_atom_count": int(seed_rec.get("prepared_atom_count", 0)),
            "eligibility": dict(seed_rec.get("eligibility", {"bioemu": True, "uma_whole": True, "uma_local": True})),
            "source": "student",
            "schema_version": "v1",
            "novelty": rng.random(),
            "pack_unc": rng.random() * 0.4,
            "sequence_length": len(new_seq),
        }
        for key in (
            "split",
            "spec_path",
            "cif_path",
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
            "complex_path",
            "pair_id",
            "ligand_chain_id",
            "protein_chain_id",
            "reactant_complex_path",
            "reactant_protein_path",
            "product_complex_path",
            "product_protein_path",
            "pocket_positions",
            "rhea_id",
        ):
            if key in seed_rec:
                rec[key] = seed_rec[key]
        records.append(rec)

    dedup = {r["candidate_id"]: r for r in records}
    return list(dedup.values())


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True))


def load_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def reconstructed_trajectory_stats(records: list[dict]) -> dict[str, float]:
    bins = defaultdict(int)
    for r in records:
        k = int(r.get("K", 0))
        if k == 0:
            bins["x0"] += 1
        elif k == 1:
            bins["k1"] += 1
        elif k == 2:
            bins["k2"] += 1
        else:
            bins["k3p"] += 1
    total = max(len(records), 1)
    return {f"frac_{k}": v / total for k, v in bins.items()}


def teacher_student_kl(teacher: dict, student: dict, max_k: int = 16) -> float:
    eps = 1e-12
    kl = 0.0
    t_probs = {int(k): float(v) for k, v in teacher.get("k_probs", {}).items()}
    s_probs = {int(k): float(v) for k, v in student.get("k_probs", {}).items()}
    for k in range(max_k + 1):
        p = float(t_probs.get(k, eps)) + eps
        q = float(s_probs.get(k, eps)) + eps
        kl += p * math.log(p / q)
    return float(kl)
