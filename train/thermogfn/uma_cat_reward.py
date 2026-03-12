"""Catalytic reward helpers for UMA broad/sMD screening and GraphKcat fusion."""

from __future__ import annotations

import math

KB_KCAL_MOL_K = 0.00198720425864083
PLANCK_KCAL_S = 1.583981e-37  # kcal * s


def _safe_probability(value: float | None, eps: float = 1e-6) -> float:
    if value is None:
        return float(eps)
    return float(min(1.0 - eps, max(eps, float(value))))


def gating_free_energy_kcal_mol(p_gnac: float | None, temperature_k: float = 300.0) -> float:
    """Convert productive-pose occupancy into a gating free energy."""
    p = _safe_probability(p_gnac)
    rt = KB_KCAL_MOL_K * float(temperature_k)
    return float(-rt * math.log(p / (1.0 - p)))


def log10_tst_prefactor(temperature_k: float = 300.0) -> float:
    """Return log10(k_B T / h) in s^-1 with kcal/mol units."""
    kb = KB_KCAL_MOL_K
    return float(math.log10((kb * float(temperature_k)) / PLANCK_KCAL_S))


def log10_rate_proxy(
    delta_g_gate_kcal_mol: float | None,
    delta_g_barrier_kcal_mol: float | None,
    *,
    temperature_k: float = 300.0,
) -> float:
    """Transition-state-style log10 rate proxy from gating + steering barrier."""
    dg_gate = float(delta_g_gate_kcal_mol or 0.0)
    dg_bar = float(delta_g_barrier_kcal_mol or 0.0)
    rt = KB_KCAL_MOL_K * float(temperature_k)
    return float(log10_tst_prefactor(temperature_k) - (dg_gate + dg_bar) / (rt * math.log(10.0)))


def risk_adjusted(mean: float | None, std: float | None, kappa: float = 1.0) -> float:
    if mean is None:
        return 0.0
    return float(mean) - float(kappa) * float(std or 0.0)


def compute_fused_catalytic_score(
    record: dict,
    *,
    w_uma_cat: float = 0.75,
    w_graphkcat: float = 0.45,
    w_agreement: float = 0.20,
    kappa_uma_cat: float = 1.0,
    kappa_graphkcat: float = 1.0,
    graph_field: str = "graphkcat_log_kcat",
) -> dict:
    """Fuse UMA catalytic dynamics with GraphKcat into a single positive reward."""
    uma_mean = record.get("uma_cat_log10_rate_proxy")
    uma_std = record.get("uma_cat_log10_rate_std", record.get("uma_cat_std"))
    graph_mean = record.get(graph_field)
    graph_std = record.get("graphkcat_std")

    z_uma = risk_adjusted(uma_mean, uma_std, kappa=kappa_uma_cat)
    z_graph = risk_adjusted(graph_mean, graph_std, kappa=kappa_graphkcat)

    has_uma = 1.0 if str(record.get("uma_cat_status", "")).strip() == "ok" and uma_mean is not None else 0.0
    has_graph = 1.0 if str(record.get("graphkcat_status", "")).strip() == "ok" and graph_mean is not None else 0.0

    if has_uma and has_graph:
        z_agree = -abs(float(uma_mean) - float(graph_mean))
    else:
        z_agree = 0.0

    score = (
        float(w_uma_cat) * float(has_uma) * z_uma
        + float(w_graphkcat) * float(has_graph) * z_graph
        + float(w_agreement) * float(has_uma) * float(has_graph) * z_agree
    )
    # Failed UMA evaluations should not silently receive a neutral reward.
    if not has_uma:
        score = min(score, -8.0)

    score = max(-8.0, min(8.0, score))
    reward = 1e-6 + math.exp(score)

    record.update(
        {
            "rho_UCAT": float(has_uma),
            "rho_G": float(has_graph),
            "z_UCAT": float(z_uma),
            "z_GK": float(z_graph),
            "z_agree": float(z_agree),
            "score": float(score),
            "reward": float(reward),
        }
    )
    return record
