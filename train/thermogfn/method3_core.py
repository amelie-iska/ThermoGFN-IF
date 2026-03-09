"""Method III core utilities: surrogate, teacher, student, and candidate generation."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import AMINO_ACIDS
from .features import candidate_feature_vector, deterministic_hash
from .progress import iter_progress


FEATURE_ORDER = ["length", "K", "prepared_atom_count"] + [f"frac_{aa}" for aa in AMINO_ACIDS]


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
) -> dict:
    labeled = [r for r in records if r.get("reward") is not None]
    if not labeled:
        return {"models": [], "feature_order": FEATURE_ORDER, "prior_mean": 1.0, "prior_std": 0.2}

    x = np.stack([_vec_from_record(r) for r in labeled], axis=0)
    y = np.array([float(r.get("reward", 1.0)) for r in labeled], dtype=float)

    rng = random.Random(seed)
    models: list[dict[str, Any]] = []
    for _ in iter_progress(
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


def train_teacher_policy(
    records: list[dict],
    seed: int,
    *,
    show_progress: bool = False,
    progress_desc: str = "teacher:fit",
) -> dict:
    rng = random.Random(seed)
    # Reward-weighted K distribution.
    k_weight = Counter()
    aa_weight = Counter()
    for r in iter_progress(records, total=len(records), desc=progress_desc, no_progress=not show_progress):
        rew = float(r.get("reward", 1.0))
        k = int(r.get("K", 0))
        k_weight[k] += max(rew, 1e-6)
        for aa in r.get("sequence", ""):
            aa_weight[aa] += max(rew, 1e-6)

    total_k = sum(k_weight.values()) or 1.0
    observed_k = {k: v / total_k for k, v in sorted(k_weight.items()) if k >= 0}
    # Mutation-order exploration prior from plan (compressed for first implementation).
    prior_k = {1: 0.30, 2: 0.35, 3: 0.20, 4: 0.15}

    if not observed_k:
        k_probs = dict(prior_k)
    elif set(observed_k.keys()) == {0}:
        # Avoid immediate-collapse behavior when D_r has mostly baselines.
        k_probs = {0: 0.10, 1: 0.28, 2: 0.32, 3: 0.18, 4: 0.12}
    else:
        # Blend empirical with prior to keep higher-order support alive.
        k_probs = {}
        keys = set(observed_k) | set(prior_k)
        for k in keys:
            k_probs[k] = 0.8 * observed_k.get(k, 0.0) + 0.2 * prior_k.get(k, 0.0)
        norm = sum(k_probs.values()) or 1.0
        k_probs = {k: v / norm for k, v in k_probs.items()}

    total_aa = sum(aa_weight.values()) or 1.0
    aa_probs = {aa: aa_weight.get(aa, 1.0) / total_aa for aa in AMINO_ACIDS}

    return {
        "k_probs": {str(k): float(v) for k, v in k_probs.items()},
        "aa_probs": aa_probs,
        "temperature_range": [0.75, 1.75],
        "seed": seed,
        "nonce": rng.random(),
    }


def distill_student_from_teacher(
    teacher: dict,
    records: list[dict],
    seed: int,
    *,
    show_progress: bool = False,
    progress_desc: str = "student:distill",
) -> dict:
    # Keep same structure, with mild smoothing toward observed empirical counts.
    rng = random.Random(seed)
    teacher_k = teacher.get("k_probs", {})
    k_probs = {int(k): float(v) for k, v in teacher_k.items()}
    if not k_probs:
        k_probs = {0: 0.10, 1: 0.28, 2: 0.32, 3: 0.18, 4: 0.12}

    observed = Counter(int(r.get("K", 0)) for r in iter_progress(records, total=len(records), desc=progress_desc, no_progress=not show_progress))
    total_obs = sum(observed.values()) or 1
    all_keys = set(k_probs) | set(observed)
    updated: dict[int, float] = {}
    for k in all_keys:
        empirical = observed.get(k, 0) / total_obs
        updated[k] = 0.8 * float(k_probs.get(k, 0.0)) + 0.2 * empirical
    norm_k = sum(updated.values()) or 1.0
    k_probs = {k: v / norm_k for k, v in updated.items()}

    aa_probs = dict(teacher.get("aa_probs", {}))
    for aa in AMINO_ACIDS:
        aa_probs[aa] = max(1e-6, float(aa_probs.get(aa, 1.0 / len(AMINO_ACIDS))))
    norm = sum(aa_probs.values())
    aa_probs = {k: v / norm for k, v in aa_probs.items()}

    return {
        "k_probs": {str(k): float(v) for k, v in k_probs.items()},
        "aa_probs": aa_probs,
        "seed": seed,
        "nonce": rng.random(),
    }


def _sample_discrete(prob_map: dict[int, float] | dict[str, float], rng: random.Random):
    keys = list(prob_map.keys())
    vals = np.array([float(prob_map[k]) for k in keys], dtype=float)
    vals = vals / max(np.sum(vals), 1e-12)
    idx = int(np.searchsorted(np.cumsum(vals), rng.random(), side="right"))
    idx = min(idx, len(keys) - 1)
    return keys[idx]


def mutate_sequence(seed_seq: str, k: int, aa_probs: dict[str, float], rng: random.Random) -> tuple[str, list[str]]:
    if not seed_seq:
        return seed_seq, []
    positions = list(range(len(seed_seq)))
    rng.shuffle(positions)
    positions = positions[: min(k, len(positions))]
    seq = list(seed_seq)
    muts: list[str] = []
    for pos in sorted(positions):
        old = seq[pos]
        new = old
        attempts = 0
        while new == old and attempts < 10:
            new = _sample_discrete(aa_probs, rng)
            attempts += 1
        seq[pos] = new
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
    records: list[dict] = []
    for i in iter_progress(
        range(pool_size),
        total=pool_size,
        desc=progress_desc,
        no_progress=not show_progress,
    ):
        seed_rec = seeds[i % len(seeds)]
        k = int(_sample_discrete(student["k_probs"], rng))
        new_seq, muts = mutate_sequence(seed_rec["sequence"], k, student["aa_probs"], rng)
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
        ):
            if key in seed_rec:
                rec[key] = seed_rec[key]
        records.append(rec)

    # de-duplicate by candidate_id
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
