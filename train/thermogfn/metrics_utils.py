"""Shared metrics summarization helpers for orchestration and oracle stages."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_candidate_records(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(rows),
        "unique_sequences": len({row.get("sequence") for row in rows if row.get("sequence")}),
    }
    out["unique_fraction"] = float(out["unique_sequences"]) / max(int(out["n"]), 1)

    ks = [_safe_float(row.get("K")) for row in rows]
    ks_clean = [float(v) for v in ks if v is not None]
    out["K"] = _summary_stats(ks_clean)

    rewards = [_safe_float(row.get("reward")) for row in rows]
    reward_clean = [float(v) for v in rewards if v is not None]
    if reward_clean:
        out["reward"] = _summary_stats(reward_clean)

    sources = Counter(str(row.get("source", "")) for row in rows)
    if sources:
        out["source_counts"] = dict(sources)
    return out


def summarize_graphkcat_rows(rows: list[dict]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("graphkcat_status", "missing")) for row in rows)
    valid = [row for row in rows if str(row.get("graphkcat_status", "")) == "ok"]
    log_kcat = [float(v) for v in (_safe_float(row.get("graphkcat_log_kcat")) for row in valid) if v is not None]
    stds = [float(v) for v in (_safe_float(row.get("graphkcat_std")) for row in valid) if v is not None]
    error_counts: Counter[str] = Counter()
    fragment_policy_counts = Counter(str(row.get("graphkcat_fragment_policy", "missing")) for row in rows)
    ligand_source_counts = Counter(str(row.get("graphkcat_ligand_source", "missing")) for row in rows)
    for row in rows:
        if str(row.get("graphkcat_status", "")) == "ok":
            continue
        msg = str(row.get("graphkcat_error", "")).lower()
        if "rdkit embedding failed" in msg:
            key = "rdkit_embedding_failed"
        elif "missing prediction" in msg:
            key = "missing_prediction"
        elif "unsupported ligand atom" in msg:
            key = "unsupported_ligand_atom"
        elif "bonded substrate" in msg or "no bond" in msg:
            key = "unbonded_ligand"
        elif "invalid substrate smiles" in msg:
            key = "invalid_smiles"
        elif "pocket" in msg:
            key = "pocket_parse_or_empty"
        elif "protein" in msg:
            key = "protein_parse_or_empty"
        else:
            key = "other"
        error_counts[key] += 1
    return {
        "n": len(rows),
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "fragment_policy_counts": dict(fragment_policy_counts),
        "ligand_source_counts": dict(ligand_source_counts),
        "ok_fraction": float(len(valid)) / max(len(rows), 1),
        "log_kcat": _summary_stats(log_kcat),
        "std": _summary_stats(stds),
    }


def summarize_uma_cat_rows(rows: list[dict]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("uma_cat_status", "missing")) for row in rows)
    valid = [row for row in rows if str(row.get("uma_cat_status", "")) == "ok"]
    rate_proxy = [
        float(v)
        for v in (_safe_float(row.get("uma_cat_log10_rate_proxy")) for row in valid)
        if v is not None
    ]
    p_gnac = [float(v) for v in (_safe_float(row.get("uma_cat_p_gnac")) for row in valid) if v is not None]
    barrier = [
        float(v)
        for v in (_safe_float(row.get("uma_cat_delta_g_chem_kcal_mol")) for row in valid)
        if v is not None
    ]
    quality_flags = []
    for row in valid:
        value = row.get("uma_cat_smd_quality_pass")
        if value is None:
            continue
        if isinstance(value, bool):
            quality_flags.append(bool(value))
        else:
            quality_flags.append(str(value).strip().lower() in {"1", "true", "yes", "ok", "pass"})
    quality_pass_count = sum(1 for value in quality_flags if value)
    return {
        "n": len(rows),
        "status_counts": dict(status_counts),
        "ok_fraction": float(len(valid)) / max(len(rows), 1),
        "smd_quality_evaluated_count": int(len(quality_flags)),
        "smd_quality_pass_count": int(quality_pass_count),
        "smd_quality_pass_fraction": float(quality_pass_count) / max(len(quality_flags), 1),
        "log10_rate_proxy": _summary_stats(rate_proxy),
        "p_gnac": _summary_stats(p_gnac),
        "delta_g_chem_kcal_mol": _summary_stats(barrier),
    }
