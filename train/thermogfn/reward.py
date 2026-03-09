"""Reward harmonization and risk-adjusted fused scoring."""

from __future__ import annotations

import math

from .constants import DEFAULT_SCORE_WEIGHTS


def risk_adjusted(mean: float | None, std: float | None, kappa: float = 1.0) -> float:
    if mean is None:
        return 0.0
    return float(mean) - kappa * float(std or 0.0)


def compute_reliability_gates(candidate: dict) -> tuple[float, float]:
    task = candidate.get("task_type", "monomer")
    atom_count = int(candidate.get("prepared_atom_count", 0))
    rho_b = 1.0 if task == "monomer" else 0.5
    if atom_count <= 8000:
        rho_u = 1.0
    elif atom_count <= 10000:
        rho_u = 0.5
    else:
        rho_u = 0.35
    return rho_b, rho_u


def compute_fused_score(record: dict, weights: dict | None = None) -> dict:
    w = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        w.update(weights)

    z_s = risk_adjusted(record.get("spurs_mean"), record.get("spurs_std"), kappa=1.0)
    z_b = risk_adjusted(record.get("bioemu_calibrated"), record.get("bioemu_std"), kappa=1.0)
    z_u = risk_adjusted(record.get("uma_calibrated"), record.get("uma_std"), kappa=1.0)

    rho_b = float(record.get("rho_B", 0.0))
    rho_u = float(record.get("rho_U", 0.0))

    k = int(record.get("K", 0))
    z_ord = -1.0 if k >= 3 and record.get("bioemu_calibrated") is None and record.get("uma_calibrated") is None else 0.0
    z_pack = -float(record.get("pack_unc", 0.0))

    score = (
        w["w_S"] * z_s
        + rho_b * w["w_B"] * z_b
        + rho_u * w["w_U"] * z_u
        + w["w_pack"] * z_pack
        + w["w_ord"] * z_ord
    )
    score = max(-8.0, min(8.0, score))
    reward = 1e-6 + math.exp(score)

    record.update(
        {
            "z_S": z_s,
            "z_B": z_b,
            "z_U": z_u,
            "z_pack": z_pack,
            "z_ord": z_ord,
            "score": score,
            "reward": reward,
        }
    )
    return record
