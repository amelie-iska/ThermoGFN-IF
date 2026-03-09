"""Evaluation utilities for periodic monitoring and overfitting checks."""

from __future__ import annotations

from collections import defaultdict


def topk_mean(records: list[dict], key: str, k: int) -> float:
    if not records:
        return 0.0
    vals = sorted((float(r.get(key, 0.0)) for r in records), reverse=True)[:k]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def mutation_hist(records: list[dict]) -> dict[str, int]:
    out = defaultdict(int)
    for r in records:
        kval = int(r.get("K", 0))
        if kval == 0:
            out["x0"] += 1
        elif kval == 1:
            out["K=1"] += 1
        elif kval == 2:
            out["K=2"] += 1
        else:
            out["K>=3"] += 1
    return dict(out)


def overfit_gap(train_metrics: dict, test_metrics: dict, key: str = "top8_mean_reward") -> float:
    return float(train_metrics.get(key, 0.0) - test_metrics.get(key, 0.0))


def compute_design_metrics(records: list[dict]) -> dict:
    metrics = {
        "n": len(records),
        "best_reward": max((float(r.get("reward", 0.0)) for r in records), default=0.0),
        "top8_mean_reward": topk_mean(records, "reward", 8),
        "top64_mean_reward": topk_mean(records, "reward", 64),
        "mutation_hist": mutation_hist(records),
        "unique_sequences": len({r.get("sequence") for r in records}),
    }
    metrics["unique_fraction"] = metrics["unique_sequences"] / max(metrics["n"], 1)
    return metrics
