"""Sequence and candidate feature extraction."""

from __future__ import annotations

from collections import Counter
import hashlib
import random

from .constants import AMINO_ACIDS


def deterministic_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def random_sequence(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(max(length, 1)))


def aa_composition(sequence: str) -> dict[str, float]:
    n = max(len(sequence), 1)
    counts = Counter(sequence)
    return {f"frac_{aa}": counts.get(aa, 0) / n for aa in AMINO_ACIDS}


def candidate_feature_vector(rec: dict) -> dict[str, float]:
    seq = rec["sequence"]
    comp = aa_composition(seq)
    out: dict[str, float] = {
        "length": float(len(seq)),
        "K": float(rec.get("K", 0)),
        "prepared_atom_count": float(rec.get("prepared_atom_count", 0)),
    }
    out.update(comp)
    return out
