#!/usr/bin/env python3
"""Build RF3 JSON inputs from ReactZyme train ligand templates."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("rf3.reactzyme")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _sanitize_token(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip())
    return cleaned.strip("._-") or "NA"


def _resolve_existing_path(raw: str, *, source_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (source_root / path).resolve()


def _extract_pocket_positions_from_features(features: list[dict]) -> list[int]:
    positions: list[int] = []
    for feature in features:
        ftype = str(feature.get("type", "")).strip().lower()
        if ftype not in {"binding site", "binding_site", "binding", "active site"}:
            continue

        location = feature.get("location") or {}
        position = location.get("position") or {}
        if isinstance(position, dict) and "value" in position:
            try:
                positions.append(int(position["value"]))
                continue
            except Exception:
                pass

        for start_key, end_key in (("begin", "end"), ("start", "end")):
            start_obj = location.get(start_key) or {}
            end_obj = location.get(end_key) or {}
            if not isinstance(start_obj, dict) or not isinstance(end_obj, dict):
                continue
            if "value" not in start_obj or "value" not in end_obj:
                continue
            try:
                start = int(start_obj["value"])
                end = int(end_obj["value"])
            except Exception:
                continue
            positions.extend(range(start, end + 1))
            break

    return sorted(set(positions))


def _load_pocket_cache(cache_dir: Path, logger: logging.Logger) -> dict[str, list[int]]:
    cache: dict[str, list[int]] = {}
    if not cache_dir.exists():
        logger.warning("Pocket cache directory does not exist: %s", cache_dir)
        return cache

    for json_path in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            cache[json_path.stem] = _extract_pocket_positions_from_features(
                data.get("features") or []
            )
        except Exception as exc:
            logger.warning("Failed to parse pocket cache %s: %s", json_path, exc)
    return cache


def _load_uniprot_sequences(tsv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            uniprot_id = str(row.get("Entry") or "").strip()
            sequence = str(row.get("Sequence") or "").strip()
            if uniprot_id and sequence:
                mapping[uniprot_id] = sequence
    return mapping


def _shard_round_robin(items: list[dict], n_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(max(1, n_shards))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return shards


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _job_name(pair_id: str, state: str) -> str:
    return _sanitize_token(f"{pair_id}__{state}")


def _count_sdf_atoms(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    total_atoms = 0
    for raw_block in text.split("$$$$"):
        block = raw_block.strip()
        if not block:
            continue
        lines = block.splitlines()

        counts_v3000 = next((line for line in lines if "V3000" in line and "COUNTS" in line), None)
        if counts_v3000 is not None:
            parts = counts_v3000.split()
            try:
                counts_idx = parts.index("COUNTS")
                total_atoms += int(parts[counts_idx + 1])
                continue
            except Exception as exc:
                raise ValueError(f"Could not parse V3000 atom count in {path}") from exc

        if len(lines) < 4:
            raise ValueError(f"SDF block too short to parse atom count in {path}")
        counts_line = lines[3]
        try:
            total_atoms += int(counts_line[:3].strip())
        except Exception as exc:
            raise ValueError(f"Could not parse V2000 atom count in {path}") from exc
    return total_atoms


def _build_rf3_example(
    *,
    pair_id: str,
    state: str,
    uniprot_id: str,
    rhea_id: str,
    row_id: str,
    protein_sequence: str,
    ligand_smiles: str,
    ligand_sdf: Path,
    pocket_positions: list[int],
    protein_chain_id: str,
    ligand_chain_id: str,
    pocket_distance_threshold: float,
    template_threshold: float,
    ligand_atom_count: int,
) -> dict:
    example_name = _job_name(pair_id, state)
    contacts = [[protein_chain_id, int(pos)] for pos in pocket_positions]

    return {
        "version": 1,
        "name": example_name,
        "components": [
            {
                "seq": protein_sequence,
                "chain_id": protein_chain_id,
            },
            {
                "path": str(ligand_sdf),
                "chain_id": ligand_chain_id,
            },
        ],
        "template_selection": [ligand_chain_id],
        "ground_truth_conformer_selection": [ligand_chain_id],
        "templates": [
            {
                "sdf": str(ligand_sdf),
                "chain_id": ligand_chain_id,
                "atom_map": "identical",
                "force": True,
                "threshold": float(template_threshold),
            }
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": ligand_chain_id,
                    "contacts": contacts,
                    "max_distance": float(pocket_distance_threshold),
                    "force": True,
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_chain_id,
                }
            }
        ],
        "metadata": {
            "source": "generate-constraints_0",
            "pair_id": pair_id,
            "state": state,
            "row_id": row_id,
            "rhea_id": rhea_id,
            "uniprot_id": uniprot_id,
            "protein_chain_id": protein_chain_id,
            "ligand_chain_id": ligand_chain_id,
            "sequence_length": len(protein_sequence),
            "ligand_smiles": ligand_smiles,
            "fragment_count": ligand_smiles.count(".") + 1 if ligand_smiles else 0,
            "ligand_atom_count": int(ligand_atom_count),
            "pocket_positions": pocket_positions,
            "pocket_count": len(pocket_positions),
            "ligand_sdf": str(ligand_sdf),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="generate-constraints_0",
        help="Root directory that contains the ReactZyme SDF outputs and metadata.",
    )
    parser.add_argument(
        "--manifest",
        default="generate-constraints_0/output_sdf_templates/train/manifest.csv",
        help="Manifest CSV emitted by the ETFlow template generation step.",
    )
    parser.add_argument(
        "--sequence-tsv",
        default="generate-constraints_0/data/reactzyme_data_split/cleaned_uniprot_rhea.tsv",
        help="UniProt-to-sequence TSV used to recover enzyme sequences.",
    )
    parser.add_argument(
        "--pocket-cache",
        default="generate-constraints_0/pocket_cache",
        help="Directory of UniProt feature JSON files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for generated RF3 JSON examples, shards, and manifests.",
    )
    parser.add_argument(
        "--states",
        choices=("both", "reactant", "product"),
        default="both",
        help="Which state JSONs to emit per accepted ReactZyme row.",
    )
    parser.add_argument(
        "--required-status",
        action="append",
        dest="required_statuses",
        default=None,
        help="Allowed manifest status value. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=600,
        help="Maximum allowed enzyme sequence length.",
    )
    parser.add_argument(
        "--max-ligand-atoms",
        type=int,
        default=256,
        help="Maximum allowed total atom count per ligand SDF template.",
    )
    parser.add_argument(
        "--max-pairs-per-sequence",
        type=int,
        default=2,
        help="Maximum number of accepted docking pairs per exact protein sequence.",
    )
    parser.add_argument(
        "--pocket-distance-threshold",
        type=float,
        default=8.0,
        help="Distance threshold used in the emitted pocket constraints.",
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.5,
        help="Threshold stored in the emitted ligand template metadata.",
    )
    parser.add_argument(
        "--protein-chain-id",
        default="A",
        help="Protein chain identifier recorded in the emitted RF3 JSON.",
    )
    parser.add_argument(
        "--ligand-chain-id",
        default="B",
        help="Ligand chain identifier recorded in the emitted RF3 JSON.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Number of round-robin shard JSON files to emit per state.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on manifest rows scanned before filtering.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit on accepted manifest rows before state expansion.",
    )
    parser.add_argument(
        "--allow-missing-pocket",
        action="store_true",
        help="Allow rows without pocket annotations.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    args = parser.parse_args()

    root = _repo_root()
    logger = _configure_logging(args.log_level)

    source_root = _resolve_existing_path(args.source_root, source_root=root)
    manifest_path = _resolve_existing_path(args.manifest, source_root=root)
    sequence_tsv = _resolve_existing_path(args.sequence_tsv, source_root=root)
    pocket_cache_dir = _resolve_existing_path(args.pocket_cache, source_root=root)
    output_root = (root / args.output_root).resolve()

    required_statuses = (
        set(args.required_statuses)
        if args.required_statuses
        else {"reactant:ok|product:ok"}
    )
    requested_states = (
        ("reactant", "product")
        if args.states == "both"
        else (args.states,)
    )

    logger.info("Loading UniProt sequences from %s", sequence_tsv)
    seq_by_uniprot = _load_uniprot_sequences(sequence_tsv)
    logger.info("Loaded %d UniProt sequences", len(seq_by_uniprot))

    logger.info("Loading pocket cache from %s", pocket_cache_dir)
    pocket_cache = _load_pocket_cache(pocket_cache_dir, logger)
    logger.info("Loaded pocket features for %d UniProt IDs", len(pocket_cache))

    t0 = time.perf_counter()
    examples_by_state: dict[str, list[dict]] = {state: [] for state in requested_states}
    manifest_rows: list[dict] = []
    accepted_pairs_by_sequence: dict[str, int] = defaultdict(int)
    summary: dict[str, int | float | list[str] | str] = {
        "total_rows_scanned": 0,
        "accepted_rows": 0,
        "emitted_examples": 0,
        "skipped_status": 0,
        "skipped_missing_sequence": 0,
        "skipped_sequence_too_long": 0,
        "skipped_missing_pocket": 0,
        "skipped_missing_smiles": 0,
        "skipped_missing_sdf": 0,
        "skipped_ligand_atom_limit": 0,
        "skipped_sequence_pair_cap": 0,
    }

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.max_rows is not None and int(summary["total_rows_scanned"]) >= args.max_rows:
                break
            summary["total_rows_scanned"] = int(summary["total_rows_scanned"]) + 1

            status = str(row.get("status") or "").strip()
            if status not in required_statuses:
                summary["skipped_status"] = int(summary["skipped_status"]) + 1
                continue

            uniprot_id = str(row.get("uniprot_id") or "").strip()
            sequence = seq_by_uniprot.get(uniprot_id, "")
            if not sequence:
                summary["skipped_missing_sequence"] = int(summary["skipped_missing_sequence"]) + 1
                continue
            if len(sequence) > args.max_seq_len:
                summary["skipped_sequence_too_long"] = int(
                    summary["skipped_sequence_too_long"]
                ) + 1
                continue

            pocket_positions = pocket_cache.get(uniprot_id, [])
            if not pocket_positions and not args.allow_missing_pocket:
                summary["skipped_missing_pocket"] = int(summary["skipped_missing_pocket"]) + 1
                continue

            state_payloads: list[tuple[str, str, Path, int]] = []
            missing_state = False
            for state in requested_states:
                smiles = str(row.get(f"{state}_smiles") or "").strip()
                sdf_raw = str(row.get(f"{state}_sdf") or "").strip()
                if not smiles:
                    summary["skipped_missing_smiles"] = int(summary["skipped_missing_smiles"]) + 1
                    missing_state = True
                    break
                if not sdf_raw:
                    summary["skipped_missing_sdf"] = int(summary["skipped_missing_sdf"]) + 1
                    missing_state = True
                    break
                sdf_path = _resolve_existing_path(sdf_raw, source_root=source_root)
                if not sdf_path.exists():
                    summary["skipped_missing_sdf"] = int(summary["skipped_missing_sdf"]) + 1
                    missing_state = True
                    break
                atom_count = _count_sdf_atoms(sdf_path)
                if atom_count > args.max_ligand_atoms:
                    summary["skipped_ligand_atom_limit"] = int(
                        summary["skipped_ligand_atom_limit"]
                    ) + 1
                    missing_state = True
                    break
                state_payloads.append((state, smiles, sdf_path, atom_count))
            if missing_state:
                continue

            pair_id = str(row.get("pair_id") or "").strip()
            row_id = str(row.get("row_id") or "").strip()
            rhea_id = str(row.get("rhea_id") or "").strip()

            if accepted_pairs_by_sequence[sequence] >= args.max_pairs_per_sequence:
                summary["skipped_sequence_pair_cap"] = int(
                    summary["skipped_sequence_pair_cap"]
                ) + 1
                continue

            accepted_rows = int(summary["accepted_rows"])
            if args.max_examples is not None and accepted_rows >= args.max_examples:
                break

            summary["accepted_rows"] = accepted_rows + 1
            accepted_pairs_by_sequence[sequence] += 1
            for state, smiles, sdf_path, atom_count in state_payloads:
                example = _build_rf3_example(
                    pair_id=pair_id,
                    state=state,
                    uniprot_id=uniprot_id,
                    rhea_id=rhea_id,
                    row_id=row_id,
                    protein_sequence=sequence,
                    ligand_smiles=smiles,
                    ligand_sdf=sdf_path,
                    pocket_positions=pocket_positions,
                    protein_chain_id=args.protein_chain_id,
                    ligand_chain_id=args.ligand_chain_id,
                    pocket_distance_threshold=args.pocket_distance_threshold,
                    template_threshold=args.template_threshold,
                    ligand_atom_count=atom_count,
                )
                examples_by_state[state].append(example)
                manifest_rows.append(
                    {
                        "pair_id": pair_id,
                        "example_name": example["name"],
                        "state": state,
                        "row_id": row_id,
                        "rhea_id": rhea_id,
                        "uniprot_id": uniprot_id,
                        "sequence_length": len(sequence),
                        "fragment_count": example["metadata"]["fragment_count"],
                        "ligand_atom_count": atom_count,
                        "pocket_count": len(pocket_positions),
                        "ligand_sdf": str(sdf_path),
                        "example_json": str(
                            (output_root / "examples" / state / f"{example['name']}.json").resolve()
                        ),
                    }
                )
                summary["emitted_examples"] = int(summary["emitted_examples"]) + 1

    output_root.mkdir(parents=True, exist_ok=True)
    for state, examples in examples_by_state.items():
        example_dir = output_root / "examples" / state
        for example in examples:
            _write_json(example_dir / f"{example['name']}.json", example)

        shard_dir = output_root / "shards" / state
        for shard_idx, shard in enumerate(_shard_round_robin(examples, args.shards)):
            _write_json(shard_dir / f"shard_{shard_idx:03d}.json", shard)

        _write_json(output_root / f"{state}.json", examples)

    _write_jsonl(output_root / "manifest.jsonl", manifest_rows)

    summary_payload = {
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "sequence_tsv": str(sequence_tsv),
        "pocket_cache": str(pocket_cache_dir),
        "output_root": str(output_root),
        "requested_states": list(requested_states),
        "required_statuses": sorted(required_statuses),
        "max_seq_len": int(args.max_seq_len),
        "max_ligand_atoms": int(args.max_ligand_atoms),
        "max_pairs_per_sequence": int(args.max_pairs_per_sequence),
        "pocket_distance_threshold": float(args.pocket_distance_threshold),
        "template_threshold": float(args.template_threshold),
        "protein_chain_id": args.protein_chain_id,
        "ligand_chain_id": args.ligand_chain_id,
        "allow_missing_pocket": bool(args.allow_missing_pocket),
        "counts": summary,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    _write_json(output_root / "summary.json", summary_payload)
    logger.info(
        "Wrote RF3 ReactZyme inputs to %s (accepted_rows=%d emitted_examples=%d elapsed=%.2fs)",
        output_root,
        summary_payload["counts"]["accepted_rows"],
        summary_payload["counts"]["emitted_examples"],
        summary_payload["elapsed_sec"],
    )
    print(output_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
