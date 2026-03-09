"""Kcat-stage reward harmonization and fusion helpers."""

from __future__ import annotations

import math


def risk_adjusted(mean: float | None, std: float | None, kappa: float = 1.0) -> float:
    if mean is None:
        return 0.0
    return float(mean) - float(kappa) * float(std or 0.0)


def compute_fused_kcat_score(
    record: dict,
    *,
    w_kcatnet: float = 0.65,
    w_graphkcat: float = 0.45,
    w_agreement: float = 0.15,
    kappa_kcatnet: float = 1.0,
    kappa_graphkcat: float = 1.0,
) -> dict:
    kn_mean = record.get("kcatnet_log10", record.get("mmkcat_log10"))
    kn_std = record.get("kcatnet_std", record.get("mmkcat_std"))
    graph_mean = record.get("graphkcat_log_kcat")
    graph_std = record.get("graphkcat_std")

    z_kn = risk_adjusted(kn_mean, kn_std, kappa=kappa_kcatnet)
    z_graph = risk_adjusted(graph_mean, graph_std, kappa=kappa_graphkcat)

    has_graph = graph_mean is not None
    rho_graph = 1.0 if has_graph else 0.0

    if kn_mean is not None and graph_mean is not None:
        z_agree = -abs(float(kn_mean) - float(graph_mean))
    else:
        z_agree = 0.0

    score = (
        float(w_kcatnet) * z_kn
        + float(w_graphkcat) * float(rho_graph) * z_graph
        + float(w_agreement) * float(rho_graph) * z_agree
    )
    score = max(-8.0, min(8.0, score))
    reward = 1e-6 + math.exp(score)

    record.update(
        {
            "rho_G": float(rho_graph),
            "z_KN": float(z_kn),
            "z_GK": float(z_graph),
            "z_agree": float(z_agree),
            "score": float(score),
            "reward": float(reward),
        }
    )
    return record
