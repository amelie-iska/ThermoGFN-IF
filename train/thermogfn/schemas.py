"""Data schemas and validation utilities for candidate/oracle/round artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK_TYPES = {"monomer", "ppi", "ligand"}
SOURCE_TYPES = {"teacher", "student", "replay", "oracle_promoted", "baseline"}
SPURS_MODES = {"single", "double", "higher"}


class SchemaError(ValueError):
    """Schema validation error."""


@dataclass
class ValidationSummary:
    total: int
    valid: int
    invalid: int
    errors: list[str]


def _require(rec: dict[str, Any], key: str) -> None:
    if key not in rec:
        raise SchemaError(f"Missing required field: {key}")


def validate_candidate(rec: dict[str, Any]) -> None:
    required = [
        "candidate_id",
        "run_id",
        "round_id",
        "task_type",
        "backbone_id",
        "seed_id",
        "sequence",
        "mutations",
        "K",
        "prepared_atom_count",
        "eligibility",
        "source",
        "schema_version",
    ]
    for key in required:
        _require(rec, key)

    if rec["task_type"] not in TASK_TYPES:
        raise SchemaError(f"Invalid task_type: {rec['task_type']}")
    if rec["source"] not in SOURCE_TYPES:
        raise SchemaError(f"Invalid source: {rec['source']}")
    if not isinstance(rec["mutations"], list):
        raise SchemaError("mutations must be a list")
    if rec["K"] != len(rec["mutations"]):
        raise SchemaError("K must equal len(mutations)")
    if not isinstance(rec["prepared_atom_count"], int) or rec["prepared_atom_count"] < 0:
        raise SchemaError("prepared_atom_count must be non-negative int")
    if not isinstance(rec["eligibility"], dict):
        raise SchemaError("eligibility must be a dict")
    for k in ["bioemu", "uma_whole", "uma_local"]:
        if k not in rec["eligibility"]:
            raise SchemaError(f"eligibility missing key: {k}")
        if not isinstance(rec["eligibility"][k], bool):
            raise SchemaError(f"eligibility.{k} must be bool")

    if rec["task_type"] in {"ppi", "ligand"} and "decomposition" not in rec:
        raise SchemaError("decomposition required for ppi/ligand tasks")


def validate_oracle_score(rec: dict[str, Any]) -> None:
    required = ["candidate_id", "rho_B", "rho_U", "reward"]
    for key in required:
        _require(rec, key)

    if rec.get("spurs_mode") is not None and rec.get("spurs_mode") not in SPURS_MODES:
        raise SchemaError(f"Invalid spurs_mode: {rec.get('spurs_mode')}")

    for key in ("rho_B", "rho_U"):
        v = rec[key]
        if not isinstance(v, (float, int)) or not (0.0 <= float(v) <= 1.0):
            raise SchemaError(f"{key} must be float in [0,1]")

    reward = rec["reward"]
    if not isinstance(reward, (float, int)) or float(reward) <= 0:
        raise SchemaError("reward must be positive")


def validate_records(records: list[dict[str, Any]], kind: str) -> ValidationSummary:
    errors: list[str] = []
    valid = 0
    for i, rec in enumerate(records):
        try:
            if kind == "candidate":
                validate_candidate(rec)
            elif kind == "oracle":
                validate_oracle_score(rec)
            else:
                raise SchemaError(f"Unknown schema kind: {kind}")
            valid += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"idx={i}: {exc}")
    return ValidationSummary(total=len(records), valid=valid, invalid=len(records) - valid, errors=errors)


def ensure_unique_ids(records: list[dict[str, Any]], key: str) -> None:
    seen = set()
    dups = set()
    for rec in records:
        value = rec.get(key)
        if value in seen:
            dups.add(value)
        seen.add(value)
    if dups:
        sample = sorted(list(dups))[:5]
        raise SchemaError(f"Duplicate {key} values found, sample={sample}")
