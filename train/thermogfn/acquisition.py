"""Acquisition score helpers for active-learning oracle scheduling."""

from __future__ import annotations

from .constants import (
    DEFAULT_ACQ_ALPHA,
    DEFAULT_ACQ_BETA,
    DEFAULT_ACQ_GRAPHKCAT,
    DEFAULT_ACQ_KCATNET,
)


def score_bioemu_acquisition(rec: dict, alpha: tuple[float, ...] = DEFAULT_ACQ_ALPHA) -> float:
    z_s = float(rec.get("z_S", 0.0))
    s_std = float(rec.get("spurs_std", 0.0))
    novelty = float(rec.get("novelty", 0.0))
    pack_unc = float(rec.get("pack_unc", 0.0))
    k = int(rec.get("K", 0))
    l = max(1, int(rec.get("sequence_length", len(rec.get("sequence", "")) or 1)))
    indicators = 1.0 if k == 2 else 0.0
    return (
        alpha[0] * z_s
        + alpha[1] * s_std
        + alpha[2] * novelty
        + alpha[3] * pack_unc
        + alpha[4] * indicators
        + alpha[5] * (k / l)
    )


def score_uma_acquisition(rec: dict, beta: tuple[float, ...] = DEFAULT_ACQ_BETA) -> float:
    z_b = float(rec.get("z_B", 0.0))
    b_std = float(rec.get("bioemu_std", 0.0))
    z_s = float(rec.get("z_S", 0.0))
    novelty = float(rec.get("novelty", 0.0))
    atom_count = int(rec.get("prepared_atom_count", 0))
    k = int(rec.get("K", 0))
    atom_indicator = 1.0 if atom_count <= 10000 else 0.2
    k_indicator = 1.0 if k >= 3 else 0.0
    return (
        beta[0] * z_b
        + beta[1] * b_std
        + beta[2] * abs(z_b - z_s)
        + beta[3] * novelty
        + beta[4] * atom_indicator
        + beta[5] * k_indicator
    )


def score_kcatnet_acquisition(rec: dict, alpha: tuple[float, ...] = DEFAULT_ACQ_KCATNET) -> float:
    novelty = float(rec.get("novelty", 0.0))
    pack_unc = float(rec.get("pack_unc", 0.0))
    k = int(rec.get("K", 0))
    l = max(1, int(rec.get("sequence_length", len(rec.get("sequence", "")) or 1)))
    return (
        alpha[0] * novelty
        + alpha[1] * pack_unc
        + alpha[2] * (k / l)
        + alpha[3] * (1.0 if k >= 2 else 0.0)
    )


def score_graphkcat_acquisition(
    rec: dict,
    beta: tuple[float, ...] = DEFAULT_ACQ_GRAPHKCAT,
    *,
    risk_kappa: float = 0.5,
) -> float:
    kn_mean = rec.get("kcatnet_log10", rec.get("mmkcat_log10"))
    kn_std = float(rec.get("kcatnet_std", rec.get("mmkcat_std", 0.0)))
    kn_risk = float(kn_mean) - float(risk_kappa) * kn_std if kn_mean is not None else 0.0
    novelty = float(rec.get("novelty", 0.0))
    k = int(rec.get("K", 0))
    return (
        beta[0] * kn_risk
        + beta[1] * novelty
        + beta[2] * (1.0 if k >= 2 else 0.0)
    )


# Backward-compatible alias.
def score_mmkcat_acquisition(rec: dict, alpha: tuple[float, ...] = DEFAULT_ACQ_KCATNET) -> float:
    return score_kcatnet_acquisition(rec, alpha=alpha)


def select_top(records: list[dict], key: str, budget: int) -> list[dict]:
    if budget <= 0:
        return []
    return sorted(records, key=lambda r: float(r.get(key, 0.0)), reverse=True)[:budget]
