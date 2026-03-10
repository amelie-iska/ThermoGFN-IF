#!/usr/bin/env python3
"""Attach local MMSeqs2-generated MSAs to ReactZyme RF3 JSON inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, Sequence

from tqdm import tqdm


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("rf3.reactzyme.msa")
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


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_examples_for_state(input_root: Path, state: str) -> list[dict]:
    example_dir = input_root / "examples" / state
    if example_dir.exists():
        examples: list[dict] = []
        for json_path in sorted(example_dir.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                examples.extend(payload)
            else:
                examples.append(payload)
        if examples:
            return examples

    state_json = input_root / f"{state}.json"
    if state_json.exists():
        payload = json.loads(state_json.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]

    raise FileNotFoundError(f"Could not locate RF3 JSON inputs for state '{state}' under {input_root}")


def _list_payload_paths_for_state(input_root: Path, state: str) -> tuple[str, list[Path]]:
    example_dir = input_root / "examples" / state
    if example_dir.exists():
        paths = sorted(example_dir.glob("*.json"))
        if paths:
            return ("examples", paths)

    shard_dir = input_root / "shards" / state
    if shard_dir.exists():
        paths = sorted(shard_dir.glob("*.json"))
        if paths:
            return ("shards", paths)

    state_json = input_root / f"{state}.json"
    if state_json.exists():
        return ("state_json", [state_json])

    raise FileNotFoundError(f"Could not locate RF3 JSON inputs for state '{state}' under {input_root}")


def _load_examples_from_path(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def _example_pair_id(example: dict) -> str:
    metadata = example.get("metadata") or {}
    pair_id = str(metadata.get("pair_id") or "").strip()
    if pair_id:
        return pair_id
    name = str(example.get("name") or "").strip()
    if "__reactant" in name:
        return name.rsplit("__reactant", 1)[0]
    if "__product" in name:
        return name.rsplit("__product", 1)[0]
    return name


def _ordered_pair_ids(input_root: Path, requested_states: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    manifest_path = input_root / "manifest.jsonl"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pair_id = str(row.get("pair_id") or "").strip()
                if pair_id and pair_id not in seen:
                    seen.add(pair_id)
                    ordered.append(pair_id)

    for state in requested_states:
        _, paths = _list_payload_paths_for_state(input_root, state)
        for path in paths:
            for example in _load_examples_from_path(path):
                pair_id = _example_pair_id(example)
                if pair_id and pair_id not in seen:
                    seen.add(pair_id)
                    ordered.append(pair_id)
    return ordered


def _select_pair_ids(
    input_root: Path,
    requested_states: Sequence[str],
    max_docked_pairs: int | None,
    *,
    allowed_pair_ids: set[str] | None = None,
) -> set[str] | None:
    ordered = _ordered_pair_ids(input_root, requested_states)
    if allowed_pair_ids is not None:
        ordered = [pair_id for pair_id in ordered if pair_id in allowed_pair_ids]
    if max_docked_pairs is None or max_docked_pairs <= 0:
        return None if allowed_pair_ids is None else set(ordered)
    return set(ordered[:max_docked_pairs])


def _iter_examples_from_payloads(paths: Iterable[Path], selected_pair_ids: set[str] | None) -> Iterable[dict]:
    for path in paths:
        for example in _load_examples_from_path(path):
            if selected_pair_ids is not None and _example_pair_id(example) not in selected_pair_ids:
                continue
            yield example


def _validate_pair_ligands(
    payloads_by_state: dict[str, tuple[str, list[Path]]],
    *,
    selected_pair_ids: set[str] | None,
) -> tuple[dict[str, str], dict[str, int]]:
    validation_cache: dict[str, tuple[bool, str | None]] = {}
    invalid_pair_reason: dict[str, str] = {}
    counts = {
        "invalid_pair_total": 0,
        "invalid_pair_dummy_atom": 0,
        "invalid_pair_non_embeddable": 0,
    }

    requested_states = tuple(payloads_by_state.keys())
    active_pair_ids = None if selected_pair_ids is None else set(selected_pair_ids)
    pair_states_seen: dict[str, set[str]] = {}
    resolved_pair_ids: set[str] = set()
    progress = (
        tqdm(total=len(selected_pair_ids), desc="Validate Pairs", unit="pair")
        if selected_pair_ids is not None
        else None
    )

    def resolve_pair(pair_id: str) -> None:
        if pair_id in resolved_pair_ids:
            return
        resolved_pair_ids.add(pair_id)
        if active_pair_ids is not None:
            active_pair_ids.discard(pair_id)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(
                f"valid={len(resolved_pair_ids) - len(invalid_pair_reason)} invalid={len(invalid_pair_reason)}"
            )

    try:
        for state, (_layout, paths) in payloads_by_state.items():
            for path in paths:
                if active_pair_ids is not None and not active_pair_ids:
                    return invalid_pair_reason, counts
                for example in _load_examples_from_path(path):
                    pair_id = _example_pair_id(example)
                    if active_pair_ids is not None and pair_id not in active_pair_ids:
                        continue
                    if pair_id in resolved_pair_ids:
                        continue

                    ligand_smiles = str(
                        (example.get("metadata") or {}).get("ligand_smiles")
                        or (example.get("components") or [{}])[-1].get("smiles")
                        or ""
                    ).strip()
                    pair_states_seen.setdefault(pair_id, set()).add(state)

                    if not ligand_smiles:
                        invalid_pair_reason[pair_id] = "missing_smiles"
                        counts["invalid_pair_total"] += 1
                        counts["invalid_pair_non_embeddable"] += 1
                        resolve_pair(pair_id)
                        continue

                    ok, reason = _validate_smiles_for_rf3(
                        ligand_smiles,
                        validation_cache=validation_cache,
                    )
                    if not ok:
                        invalid_pair_reason[pair_id] = str(reason or "invalid_smiles")
                        counts["invalid_pair_total"] += 1
                        if reason == "dummy_atom":
                            counts["invalid_pair_dummy_atom"] += 1
                        else:
                            counts["invalid_pair_non_embeddable"] += 1
                        resolve_pair(pair_id)
                        continue

                    if len(pair_states_seen[pair_id]) >= len(requested_states):
                        resolve_pair(pair_id)
    finally:
        if progress is not None:
            progress.close()

    return invalid_pair_reason, counts


def _merge_invalid_pair_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def _msa_cache_path_for_sequence(msa_cache_dir: Path, sequence: str) -> Path:
    seq_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
    return msa_cache_dir / f"{seq_hash}.a3m"


def _select_valid_pair_ids(
    input_root: Path,
    requested_states: Sequence[str],
    max_docked_pairs: int | None,
    *,
    payloads_by_state: dict[str, tuple[str, list[Path]]],
) -> tuple[set[str] | None, dict[str, str], dict[str, int], int]:
    ordered_pair_ids = _ordered_pair_ids(input_root, requested_states)
    if max_docked_pairs is None:
        invalid_pair_reason, invalid_pair_counts = _validate_pair_ligands(
            payloads_by_state,
            selected_pair_ids=None,
        )
        selected_pair_ids = {
            pair_id for pair_id in ordered_pair_ids if pair_id not in invalid_pair_reason
        }
        return selected_pair_ids, invalid_pair_reason, invalid_pair_counts, len(ordered_pair_ids)

    selected_pairs: list[str] = []
    invalid_pair_reason: dict[str, str] = {}
    invalid_pair_counts = {
        "invalid_pair_total": 0,
        "invalid_pair_dummy_atom": 0,
        "invalid_pair_non_embeddable": 0,
    }
    cursor = 0
    total_pairs = len(ordered_pair_ids)
    scanned_pair_count = 0
    while len(selected_pairs) < max_docked_pairs and cursor < total_pairs:
        needed = max_docked_pairs - len(selected_pairs)
        batch_size = max(needed * 2, 512)
        candidate_ids = ordered_pair_ids[cursor : min(total_pairs, cursor + batch_size)]
        cursor += len(candidate_ids)
        scanned_pair_count += len(candidate_ids)
        candidate_set = set(candidate_ids)
        batch_invalid_reason, batch_invalid_counts = _validate_pair_ligands(
            payloads_by_state,
            selected_pair_ids=candidate_set,
        )
        _merge_invalid_pair_counts(invalid_pair_counts, batch_invalid_counts)
        invalid_pair_reason.update(batch_invalid_reason)
        for pair_id in candidate_ids:
            if pair_id in batch_invalid_reason:
                continue
            selected_pairs.append(pair_id)
            if len(selected_pairs) >= max_docked_pairs:
                break

    return set(selected_pairs), invalid_pair_reason, invalid_pair_counts, scanned_pair_count


def _copy_manifest_with_pair_filter(
    src_manifest: Path,
    dst_manifest: Path,
    *,
    selected_pair_ids: set[str] | None,
) -> None:
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)
    if selected_pair_ids is None:
        shutil.copy2(src_manifest, dst_manifest)
        return

    with src_manifest.open("r", encoding="utf-8") as src, dst_manifest.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            row = json.loads(line)
            if str(row.get("pair_id") or "").strip() in selected_pair_ids:
                dst.write(line)


def _protein_component(example: dict) -> dict:
    components = example.get("components") or []
    proteins = [component for component in components if "seq" in component]
    if len(proteins) != 1:
        raise ValueError(
            f"Expected exactly one protein component in example {example.get('name')}, found {len(proteins)}"
        )
    return proteins[0]


def _ligand_smiles_component(example: dict) -> dict | None:
    components = example.get("components") or []
    for component in components:
        if "smiles" in component:
            return component
    return None


def _infer_boltz_src_path(local_msa_root: Path) -> Path:
    candidates = [
        local_msa_root / "boltz-client" / "src",
        local_msa_root / "boltz" / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not infer Boltz source path under local MSA root. "
        "Pass --boltz-src-path explicitly."
    )


def _parse_cuda_devices(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _infer_local_mmseqs_assets(local_msa_root: Path) -> dict[str, Path]:
    mmseqs_bin = (local_msa_root / "ColabFold" / "MsaServer" / "bin" / "mmseqs").resolve()
    if not mmseqs_bin.exists():
        raise FileNotFoundError(f"Could not locate local mmseqs binary: {mmseqs_bin}")

    config_path = (local_msa_root / "config.uniref30.json").resolve()
    db_prefix: Path | None = None
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_uniref = (
            payload.get("paths", {})
            .get("colabfold", {})
            .get("uniref")
        )
        if raw_uniref:
            db_prefix = Path(str(raw_uniref)).expanduser().resolve()
            if db_prefix.name.endswith("_pad"):
                db_prefix = db_prefix.with_name(db_prefix.name[: -len("_pad")])

    if db_prefix is None:
        candidate = Path("/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302/uniref30_2302_db")
        if candidate.exists():
            db_prefix = candidate.resolve()

    if db_prefix is None or not db_prefix.exists():
        raise FileNotFoundError(
            "Could not infer the local UniRef30 MMSeqs2 database prefix. "
            f"Checked config under {config_path} and the standard nvme path."
        )

    db_seq = db_prefix.with_name(f"{db_prefix.name}_seq")
    db_aln = db_prefix.with_name(f"{db_prefix.name}_aln")
    missing = [str(path) for path in (db_prefix, db_seq, db_aln) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Local UniRef30 database is incomplete for direct MMSeqs2 search. "
            f"Missing: {missing}"
        )

    return {
        "mmseqs_bin": mmseqs_bin,
        "db_prefix": db_prefix,
        "db_seq": db_seq,
        "db_aln": db_aln,
    }


def _run_checked(
    cmd: Sequence[str | Path],
    *,
    env: dict[str, str],
    cwd: Path,
) -> None:
    text_cmd = [str(arg) for arg in cmd]
    result = subprocess.run(
        text_cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode == 0:
        return
    tail = "\n".join(result.stdout.splitlines()[-40:])
    raise RuntimeError(
        f"Command failed (exit={result.returncode}): {' '.join(text_cmd)}\n{tail}"
    )


def _run_local_mmseqs_chunk(
    chunk: Sequence[str],
    *,
    chunk_idx: int,
    msa_cache_dir: Path,
    mmseqs_bin: Path,
    db_prefix: Path,
    db_seq: Path,
    db_aln: Path,
    threads: int,
    use_filter: bool,
    max_seqs: int,
    num_iterations: int,
    cuda_device: str | None,
) -> dict[str, str]:
    if not chunk:
        return {}

    prefix = f"rf3_msa_chunk_{chunk_idx:04d}_"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=str(msa_cache_dir)) as tmpdir:
        base = Path(tmpdir)
        fasta_path = base / "job.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            for idx, sequence in enumerate(chunk):
                handle.write(f">{idx}\n{sequence}\n")

        env = os.environ.copy()
        env["MMSEQS_CALL_DEPTH"] = "1"
        if cuda_device:
            env["CUDA_VISIBLE_DEVICES"] = cuda_device

        qdb = base / "qdb"
        res = base / "res"
        tmp = base / "tmp"
        prof_res = base / "prof_res"
        prof_res_h = base / "prof_res_h"
        res_exp = base / "res_exp"
        res_exp_realign = base / "res_exp_realign"
        res_exp_realign_filter = base / "res_exp_realign_filter"
        uniref_a3m = base / "uniref.a3m"

        search_cmd: list[str | Path] = [
            mmseqs_bin,
            "createdb",
            fasta_path,
            qdb,
        ]
        _run_checked(search_cmd, env=env, cwd=base)

        search_cmd = [
            mmseqs_bin,
            "search",
            qdb,
            db_prefix,
            res,
            tmp,
            "--threads",
            str(threads),
            "--num-iterations",
            str(num_iterations),
            "--db-load-mode",
            "0",
            "-a",
            "-e",
            "0.1",
            "--max-seqs",
            str(max_seqs),
            "--gpu",
            "1",
            "--prefilter-mode",
            "1",
        ]
        _run_checked(search_cmd, env=env, cwd=base)

        profile_1 = tmp / "latest" / "profile_1"
        if not profile_1.exists():
            raise RuntimeError(f"MMSeqs2 search did not produce {profile_1}")

        _run_checked([mmseqs_bin, "mvdb", profile_1, prof_res], env=env, cwd=base)
        _run_checked([mmseqs_bin, "lndb", base / "qdb_h", prof_res_h], env=env, cwd=base)

        expand_cmd: list[str | Path] = [
            mmseqs_bin,
            "expandaln",
            qdb,
            db_seq,
            res,
            db_aln,
            res_exp,
            "--db-load-mode",
            "0",
            "--threads",
            str(threads),
            "--expansion-mode",
            "0",
            "-e",
            "inf",
            "--expand-filter-clusters",
            "1" if use_filter else "0",
            "--max-seq-id",
            "0.95",
        ]
        _run_checked(expand_cmd, env=env, cwd=base)

        align_cmd: list[str | Path] = [
            mmseqs_bin,
            "align",
            prof_res,
            db_seq,
            res_exp,
            res_exp_realign,
            "--db-load-mode",
            "0",
            "-e",
            "10",
            "--max-accept",
            "100000" if use_filter else "1000000",
            "--threads",
            str(threads),
            "--alt-ali",
            "10",
            "-a",
        ]
        _run_checked(align_cmd, env=env, cwd=base)

        filterresult_cmd: list[str | Path] = [
            mmseqs_bin,
            "filterresult",
            qdb,
            db_seq,
            res_exp_realign,
            res_exp_realign_filter,
            "--db-load-mode",
            "0",
            "--qid",
            "0",
            "--qsc",
            "0.8" if use_filter else "-20.0",
            "--diff",
            "0",
            "--threads",
            str(threads),
            "--max-seq-id",
            "1.0",
            "--filter-min-enable",
            "100",
        ]
        _run_checked(filterresult_cmd, env=env, cwd=base)

        result2msa_cmd: list[str | Path] = [
            mmseqs_bin,
            "result2msa",
            qdb,
            db_seq,
            res_exp_realign_filter,
            uniref_a3m,
            "--msa-format-mode",
            "6",
            "--db-load-mode",
            "0",
            "--threads",
            str(threads),
            "--filter-msa",
            "1" if use_filter else "0",
            "--filter-min-enable",
            "1000",
            "--diff",
            "3000",
            "--qid",
            "0.0,0.2,0.4,0.6,0.8,1.0",
            "--qsc",
            "0",
            "--max-seq-id",
            "0.95",
        ]
        _run_checked(result2msa_cmd, env=env, cwd=base)
        _run_checked(
            [mmseqs_bin, "unpackdb", uniref_a3m, base, "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"],
            env=env,
            cwd=base,
        )

        a3m_text_by_name: dict[str, str] = {}
        for idx in range(len(chunk)):
            a3m_path = base / f"{idx}.a3m"
            if not a3m_path.exists():
                raise RuntimeError(f"Expected unpacked A3M missing for chunk {chunk_idx}: {a3m_path}")
            a3m_text_by_name[a3m_path.name] = a3m_path.read_text(encoding="utf-8")
        return _map_local_a3m_texts_to_sequences(chunk, a3m_text_by_name)


def _count_a3m_sequences(a3m_text: str) -> int:
    return sum(1 for line in a3m_text.splitlines() if line.startswith(">"))


def _extract_first_a3m_query_sequence(a3m_text: str) -> str:
    sequence_lines: list[str] = []
    started = False
    for line in a3m_text.splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
            continue
        if started:
            sequence_lines.append(line.strip())
    return "".join(sequence_lines)


def _normalize_a3m_query_sequence(a3m_text: str) -> str:
    return "".join(ch for ch in _extract_first_a3m_query_sequence(a3m_text) if ch.isupper())


def _a3m_matches_sequence(msa_path: Path, sequence: str) -> bool:
    try:
        a3m_text = msa_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _normalize_a3m_query_sequence(a3m_text) == sequence


def _map_local_a3m_texts_to_sequences(
    chunk: Sequence[str],
    a3m_text_by_name: dict[str, str],
) -> dict[str, str]:
    expected_by_query = {sequence: sequence for sequence in chunk}
    mapping: dict[str, str] = {}
    unmatched: list[str] = []

    for name, a3m_text in sorted(a3m_text_by_name.items()):
        query_sequence = _normalize_a3m_query_sequence(a3m_text)
        source_sequence = expected_by_query.pop(query_sequence, None)
        if source_sequence is None or source_sequence in mapping:
            unmatched.append(f"{name}:{len(query_sequence)}")
            continue
        mapping[source_sequence] = a3m_text

    if len(mapping) != len(chunk):
        expected_lengths = sorted(len(sequence) for sequence in chunk)
        raise RuntimeError(
            "Could not map local MMSeqs A3M outputs back to input sequences. "
            f"matched={len(mapping)}/{len(chunk)} "
            f"expected_lengths={expected_lengths[:8]} "
            f"unmatched_files={unmatched[:8]}"
        )

    return mapping


def _trim_a3m_depth(a3m_text: str, max_depth: int | None) -> str:
    if max_depth is None or max_depth <= 0:
        return a3m_text

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in a3m_text.splitlines(keepends=True):
        if line.startswith(">"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    if not blocks:
        return a3m_text

    trimmed = "".join("".join(block) for block in blocks[:max_depth])
    if a3m_text.endswith("\n") and not trimmed.endswith("\n"):
        trimmed += "\n"
    return trimmed


class _QuietTqdm:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total", 0)
        self.n = 0

    def __enter__(self) -> "_QuietTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_description(self, *args, **kwargs) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.n += n


_RDKIT_EMBED_VALIDATOR = None
_RDKIT_EMBED_VALIDATOR_READY = False


def _validate_smiles_for_rf3(
    smiles: str,
    *,
    validation_cache: dict[str, tuple[bool, str | None]],
) -> tuple[bool, str | None]:
    cached = validation_cache.get(smiles)
    if cached is not None:
        return cached

    if "*" in smiles:
        validation_cache[smiles] = (False, "dummy_atom")
        return validation_cache[smiles]

    global _RDKIT_EMBED_VALIDATOR
    global _RDKIT_EMBED_VALIDATOR_READY
    if not _RDKIT_EMBED_VALIDATOR_READY:
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")

            def _validator(raw_smiles: str) -> tuple[bool, str | None]:
                mol = Chem.MolFromSmiles(raw_smiles)
                if mol is None:
                    return (False, "rdkit_parse")
                return (True, None)

            _RDKIT_EMBED_VALIDATOR = _validator
        except Exception:
            _RDKIT_EMBED_VALIDATOR = None
        _RDKIT_EMBED_VALIDATOR_READY = True

    if _RDKIT_EMBED_VALIDATOR is None:
        validation_cache[smiles] = (True, None)
        return validation_cache[smiles]

    validation_cache[smiles] = _RDKIT_EMBED_VALIDATOR(smiles)
    return validation_cache[smiles]


def _write_generated_msas(
    chunk_results: Sequence[dict[str, str]],
    *,
    msa_cache_dir: Path,
    msa_depth: int | None,
) -> dict[str, Path]:
    seq_to_path: dict[str, Path] = {}
    for mapping in chunk_results:
        for sequence, a3m in mapping.items():
            a3m = _trim_a3m_depth(a3m, msa_depth)
            seq_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
            msa_path = msa_cache_dir / f"{seq_hash}.a3m"
            msa_path.write_text(a3m, encoding="utf-8")
            seq_to_path[sequence] = msa_path.resolve()
    return seq_to_path


def _generate_msas_via_server(
    sequences: Sequence[str],
    *,
    boltz_src_path: Path,
    msa_cache_dir: Path,
    host_url: str,
    use_env: bool,
    use_filter: bool,
    pairing_strategy: str,
    reuse_cache: bool,
    msa_batch_size: int,
    msa_concurrency: int,
    msa_retries: int,
    msa_depth: int | None,
) -> dict[str, Path]:
    if str(boltz_src_path) not in sys.path:
        sys.path.insert(0, str(boltz_src_path))
    import boltz.data.msa.mmseqs2 as mmseqs2_mod  # type: ignore

    run_mmseqs2 = mmseqs2_mod.run_mmseqs2

    unique_sequences: list[str] = []
    for sequence in sequences:
        if sequence not in unique_sequences:
            unique_sequences.append(sequence)

    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    seq_to_path: dict[str, Path] = {}
    missing_sequences: list[str] = []
    for sequence in unique_sequences:
        msa_path = _msa_cache_path_for_sequence(msa_cache_dir, sequence)
        if (
            reuse_cache
            and msa_path.exists()
            and msa_path.stat().st_size > 0
            and _a3m_matches_sequence(msa_path, sequence)
        ):
            seq_to_path[sequence] = msa_path.resolve()
        else:
            if msa_path.exists() and not _a3m_matches_sequence(msa_path, sequence):
                msa_path.unlink(missing_ok=True)
            missing_sequences.append(sequence)

    if not missing_sequences:
        return seq_to_path

    if msa_batch_size <= 0:
        msa_batch_size = len(missing_sequences)
    if msa_concurrency <= 0:
        msa_concurrency = 1

    chunks = [
        missing_sequences[idx : idx + msa_batch_size]
        for idx in range(0, len(missing_sequences), msa_batch_size)
    ]
    total_unique = len(unique_sequences)
    total_missing = len(missing_sequences)
    total_cached = total_unique - total_missing
    total_chunks = len(chunks)

    def _run_chunk(chunk: Sequence[str], chunk_idx: int) -> dict[str, str]:
        chunk_hash = hashlib.sha1("||".join(chunk).encode("utf-8")).hexdigest()[:12]
        prefix = str((msa_cache_dir / f"mmseqs_{chunk_hash}_{chunk_idx:03d}").resolve())

        def _run(host: str):
            return run_mmseqs2(
                list(chunk),
                prefix=prefix,
                use_env=use_env,
                use_filter=use_filter,
                use_pairing=False,
                pairing_strategy=pairing_strategy,
                host_url=host,
                msa_server_username=None,
                msa_server_password=None,
                auth_headers=None,
            )

        attempts = max(1, int(msa_retries))
        last_error: Exception | None = None
        result = None
        orig_tqdm = getattr(mmseqs2_mod, "tqdm", None)
        orig_logger_disabled = getattr(mmseqs2_mod.logger, "disabled", False)
        if orig_tqdm is not None:
            mmseqs2_mod.tqdm = _QuietTqdm
        mmseqs2_mod.logger.disabled = True
        try:
            for attempt_idx in range(attempts):
                try:
                    result = _run(host_url)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    fallback_host = host_url.rstrip("/")
                    if not fallback_host.endswith("/api"):
                        fallback_host = f"{fallback_host}/api"
                        try:
                            result = _run(fallback_host)
                            last_error = None
                            break
                        except Exception as fallback_exc:
                            last_error = fallback_exc
                    if attempt_idx + 1 < attempts:
                        tqdm.write(
                            f"[msa] retrying chunk {chunk_idx + 1}/{total_chunks} "
                            f"(size={len(chunk)}) after error: {last_error}"
                        )
        finally:
            if orig_tqdm is not None:
                mmseqs2_mod.tqdm = orig_tqdm
            mmseqs2_mod.logger.disabled = orig_logger_disabled

        if last_error is not None or result is None:
            raise RuntimeError(
                f"MSA request failed for chunk {chunk_idx} after {attempts} attempt(s)"
            ) from last_error

        lines = result[0] if isinstance(result, tuple) else result
        if len(lines) != len(chunk):
            raise RuntimeError(
                f"MSA response size mismatch for chunk {chunk_idx}: got {len(lines)} alignments for {len(chunk)} sequences"
            )
        return {sequence: a3m for sequence, a3m in zip(chunk, lines)}

    chunk_results: list[dict[str, str]] = []
    chunk_errors: list[str] = []
    max_workers = min(len(chunks), msa_concurrency)
    chunk_bar = tqdm(total=total_chunks, desc="MSA Chunks", unit="chunk")
    seq_bar = tqdm(total=total_unique, initial=total_cached, desc="MSA Seqs", unit="seq")
    chunk_bar.set_postfix_str(f"done=0/{total_chunks}")
    seq_bar.set_postfix_str(f"cached={total_cached} missing={total_missing}")
    try:
        if max_workers <= 1:
            for idx, chunk in enumerate(chunks):
                chunk_bar.set_postfix_str(f"chunk={idx + 1}/{total_chunks} size={len(chunk)}")
                seq_bar.set_postfix_str(f"cached={total_cached} missing={total_missing} active=1")
                try:
                    chunk_results.append(_run_chunk(chunk, idx))
                    chunk_bar.update(1)
                    seq_bar.update(len(chunk))
                    seq_bar.set_postfix_str(
                        f"cached={total_cached} done={seq_bar.n - total_cached}/{total_missing}"
                    )
                except Exception as exc:
                    chunk_errors.append(f"chunk={idx} size={len(chunk)} error={exc}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_chunk, chunk, idx) for idx, chunk in enumerate(chunks)]
                future_to_idx = {future: idx for idx, future in enumerate(futures)}
                future_to_size = {
                    futures[idx]: len(chunk) for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = future_to_idx[future]
                    size = future_to_size[future]
                    try:
                        chunk_results.append(future.result())
                        chunk_bar.update(1)
                        seq_bar.update(size)
                        chunk_bar.set_postfix_str(
                            f"completed={chunk_bar.n}/{total_chunks} last={idx + 1} size={size}"
                        )
                        seq_bar.set_postfix_str(
                            f"cached={total_cached} done={seq_bar.n - total_cached}/{total_missing}"
                        )
                    except Exception as exc:
                        chunk_errors.append(f"chunk={idx} size={size} error={exc}")
    finally:
        chunk_bar.close()
        seq_bar.close()

    seq_to_path.update(
        _write_generated_msas(
            chunk_results,
            msa_cache_dir=msa_cache_dir,
            msa_depth=msa_depth,
        )
    )

    unresolved = [sequence for sequence in missing_sequences if sequence not in seq_to_path]
    if unresolved:
        details = "; ".join(chunk_errors) if chunk_errors else "unknown"
        raise RuntimeError(
            f"MSA generation failed for {len(unresolved)} sequence(s): {details}"
        )

    return seq_to_path


def _generate_msas_local_direct(
    sequences: Sequence[str],
    *,
    local_msa_root: Path,
    msa_cache_dir: Path,
    reuse_cache: bool,
    msa_batch_size: int,
    msa_concurrency: int,
    msa_retries: int,
    msa_depth: int | None,
    cuda_devices: list[str],
    use_filter: bool,
    max_seqs: int,
    num_iterations: int,
    msa_threads_per_job: int,
) -> dict[str, Path]:
    assets = _infer_local_mmseqs_assets(local_msa_root)

    unique_sequences: list[str] = []
    for sequence in sequences:
        if sequence not in unique_sequences:
            unique_sequences.append(sequence)

    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    seq_to_path: dict[str, Path] = {}
    missing_sequences: list[str] = []
    for sequence in unique_sequences:
        msa_path = _msa_cache_path_for_sequence(msa_cache_dir, sequence)
        if (
            reuse_cache
            and msa_path.exists()
            and msa_path.stat().st_size > 0
            and _a3m_matches_sequence(msa_path, sequence)
        ):
            seq_to_path[sequence] = msa_path.resolve()
        else:
            if msa_path.exists() and not _a3m_matches_sequence(msa_path, sequence):
                msa_path.unlink(missing_ok=True)
            missing_sequences.append(sequence)

    if not missing_sequences:
        return seq_to_path

    if msa_batch_size <= 0:
        msa_batch_size = len(missing_sequences)
    if msa_concurrency <= 0:
        msa_concurrency = 1
    if not cuda_devices:
        cuda_devices = [""]

    chunks = [
        missing_sequences[idx : idx + msa_batch_size]
        for idx in range(0, len(missing_sequences), msa_batch_size)
    ]
    total_unique = len(unique_sequences)
    total_missing = len(missing_sequences)
    total_cached = total_unique - total_missing
    total_chunks = len(chunks)
    worker_count = max(1, min(len(chunks), msa_concurrency))

    cpu_count = os.cpu_count() or 1
    threads_per_job = int(msa_threads_per_job)
    if threads_per_job <= 0:
        threads_per_job = max(8, min(16, cpu_count // worker_count))

    chunk_bar = tqdm(total=total_chunks, desc="MSA Chunks", unit="chunk")
    seq_bar = tqdm(total=total_unique, initial=total_cached, desc="MSA Seqs", unit="seq")
    progress_lock = threading.Lock()
    results: list[dict[str, str]] = []
    errors: list[str] = []
    active_workers: dict[str, tuple[str, int]] = {}
    work_queue: Queue[tuple[int, Sequence[str]]] = Queue()
    for idx, chunk in enumerate(chunks):
        work_queue.put((idx, chunk))

    attempts = max(1, int(msa_retries))

    def _active_summary() -> str:
        if not active_workers:
            return "idle"
        parts = []
        for worker, (label, size) in sorted(active_workers.items()):
            parts.append(f"{worker}:{label}({size})")
        return " | ".join(parts[:4])

    def _update_progress_postfix() -> None:
        chunk_bar.set_postfix_str(f"done={chunk_bar.n}/{total_chunks} active={_active_summary()}")
        seq_done = max(0, seq_bar.n - total_cached)
        active_seq_count = sum(size for _, size in active_workers.values())
        seq_bar.set_postfix_str(
            f"cached={total_cached} done={seq_done}/{total_missing} active={active_seq_count}"
        )

    def _worker(worker_idx: int, device: str) -> None:
        worker_key = f"w{worker_idx}@{device or 'default'}"
        while True:
            try:
                chunk_idx, chunk = work_queue.get_nowait()
            except Empty:
                return

            with progress_lock:
                active_workers[worker_key] = (f"chunk={chunk_idx + 1}/{total_chunks}", len(chunk))
                _update_progress_postfix()

            last_error: Exception | None = None
            chunk_result: dict[str, str] | None = None
            for attempt_idx in range(attempts):
                try:
                    chunk_result = _run_local_mmseqs_chunk(
                        chunk,
                        chunk_idx=chunk_idx,
                        msa_cache_dir=msa_cache_dir,
                        mmseqs_bin=assets["mmseqs_bin"],
                        db_prefix=assets["db_prefix"],
                        db_seq=assets["db_seq"],
                        db_aln=assets["db_aln"],
                        threads=threads_per_job,
                        use_filter=use_filter,
                        max_seqs=max_seqs,
                        num_iterations=num_iterations,
                        cuda_device=device or None,
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt_idx + 1 < attempts:
                        with progress_lock:
                            tqdm.write(
                                f"[msa-direct] retrying chunk {chunk_idx + 1}/{total_chunks} "
                                f"(gpu={device or 'default'} size={len(chunk)}) after error: {exc}"
                            )

            with progress_lock:
                active_workers.pop(worker_key, None)
                if chunk_result is None:
                    errors.append(
                        f"chunk={chunk_idx} gpu={device or 'default'} size={len(chunk)} error={last_error}"
                    )
                    tqdm.write(
                        f"[msa-direct] failed chunk {chunk_idx + 1}/{total_chunks} "
                        f"(gpu={device or 'default'} size={len(chunk)}): {last_error}"
                    )
                else:
                    results.append(chunk_result)
                    chunk_bar.update(1)
                    seq_bar.update(len(chunk))
                _update_progress_postfix()
            work_queue.task_done()

    threads: list[threading.Thread] = []
    try:
        _update_progress_postfix()
        for worker_idx in range(worker_count):
            device = cuda_devices[worker_idx % len(cuda_devices)]
            thread = threading.Thread(
                target=_worker,
                args=(worker_idx, device),
                name=f"msa-gpu-{device or worker_idx}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    finally:
        chunk_bar.close()
        seq_bar.close()

    seq_to_path.update(
        _write_generated_msas(
            results,
            msa_cache_dir=msa_cache_dir,
            msa_depth=msa_depth,
        )
    )
    unresolved = [sequence for sequence in missing_sequences if sequence not in seq_to_path]
    if unresolved:
        details = "; ".join(errors) if errors else "unknown"
        raise RuntimeError(
            f"Local direct MSA generation failed for {len(unresolved)} sequence(s): {details}"
        )
    return seq_to_path


def _generate_msas(
    sequences: Sequence[str],
    *,
    msa_backend: str,
    local_msa_root: Path,
    boltz_src_path: Path | None,
    msa_cache_dir: Path,
    host_url: str,
    use_env: bool,
    use_filter: bool,
    pairing_strategy: str,
    reuse_cache: bool,
    msa_batch_size: int,
    msa_concurrency: int,
    msa_retries: int,
    msa_depth: int | None,
    cuda_devices: list[str],
    max_seqs: int,
    num_iterations: int,
    msa_threads_per_job: int,
) -> dict[str, Path]:
    if msa_backend == "server":
        if boltz_src_path is None:
            raise ValueError("boltz_src_path is required when --msa-backend=server")
        return _generate_msas_via_server(
            sequences=sequences,
            boltz_src_path=boltz_src_path,
            msa_cache_dir=msa_cache_dir,
            host_url=host_url,
            use_env=use_env,
            use_filter=use_filter,
            pairing_strategy=pairing_strategy,
            reuse_cache=reuse_cache,
            msa_batch_size=msa_batch_size,
            msa_concurrency=msa_concurrency,
            msa_retries=msa_retries,
            msa_depth=msa_depth,
        )

    if msa_backend == "local_direct":
        return _generate_msas_local_direct(
            sequences=sequences,
            local_msa_root=local_msa_root,
            msa_cache_dir=msa_cache_dir,
            reuse_cache=reuse_cache,
            msa_batch_size=msa_batch_size,
            msa_concurrency=msa_concurrency,
            msa_retries=msa_retries,
            msa_depth=msa_depth,
            cuda_devices=cuda_devices,
            use_filter=use_filter,
            max_seqs=max_seqs,
            num_iterations=num_iterations,
            msa_threads_per_job=msa_threads_per_job,
        )

    raise ValueError(f"Unsupported MSA backend: {msa_backend}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _shard_round_robin(items: list[dict], n_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(max(1, n_shards))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="RF3 input directory emitted by build_reactzyme_rf3_inputs.py.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to write JSONs updated with msa_path entries.",
    )
    parser.add_argument(
        "--states",
        choices=("both", "reactant", "product"),
        default="both",
        help="Which state JSONs to process.",
    )
    parser.add_argument(
        "--local-msa-root",
        default="../enzyme-quiver/MMseqs2/local_msa",
        help="Shared local MMSeqs2 root. Used to infer the Boltz client path when needed.",
    )
    parser.add_argument(
        "--msa-backend",
        choices=("local_direct", "server"),
        default="local_direct",
        help="MSA generation backend. local_direct uses local MMSeqs2 GPU jobs directly; server uses the Boltz MMSeqs ticket API.",
    )
    parser.add_argument(
        "--boltz-src-path",
        default=None,
        help="Optional explicit path to Boltz client src for run_mmseqs2. Only used with --msa-backend=server.",
    )
    parser.add_argument(
        "--msa-cache-dir",
        default=None,
        help="Optional explicit MSA cache directory. Defaults to <output-root>/msas.",
    )
    parser.add_argument(
        "--msa-server-url",
        default="http://127.0.0.1:8080/api",
        help="Local MMSeqs2 server URL.",
    )
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cached .a3m files.")
    parser.add_argument("--use-env-db", action="store_true", help="Request the environmental DB as well.")
    parser.add_argument("--use-filter", dest="use_filter", action="store_true", help="Enable MMSeqs filtering (default).")
    parser.add_argument("--no-use-filter", dest="use_filter", action="store_false", help="Disable MMSeqs filtering.")
    parser.add_argument("--pairing-strategy", default="greedy", help="Pairing strategy passed to run_mmseqs2.")
    parser.add_argument("--msa-batch-size", type=int, default=64, help="Sequences per MMSeqs2 batch request.")
    parser.add_argument("--msa-concurrency", type=int, default=8, help="Number of concurrent MMSeqs2 batch requests.")
    parser.add_argument("--msa-retries", type=int, default=2, help="Retries per MMSeqs2 batch request.")
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        help="Comma-separated CUDA devices exposed to local_direct MSA workers (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--mmseqs-max-seqs",
        type=int,
        default=4096,
        help="Maximum MMSeqs hits retained during local_direct search (default: 4096).",
    )
    parser.add_argument(
        "--mmseqs-num-iterations",
        type=int,
        default=3,
        help="MMSeqs search iterations for local_direct backend (default: 3).",
    )
    parser.add_argument(
        "--msa-threads-per-job",
        type=int,
        default=0,
        help="Threads per local_direct MMSeqs chunk. Use 0 to auto-size from CPU count and concurrency.",
    )
    parser.add_argument(
        "--msa-depth",
        type=int,
        default=2048,
        help="Maximum number of sequences retained in each written A3M, including the query. Use 0 to disable trimming.",
    )
    parser.add_argument("--shards", type=int, default=1, help="Number of round-robin shard JSON files to emit per state.")
    parser.add_argument(
        "--max-docked-pairs",
        "--max-examples",
        dest="max_docked_pairs",
        type=int,
        default=2000,
        help=(
            "Maximum number of docking pairs to prepare for inference. "
            "Use 0 to disable the cap."
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.set_defaults(use_filter=True)
    args = parser.parse_args()

    root = _repo_root()
    logger = _configure_logging(args.log_level)

    input_root = _resolve_path(args.input_root, base_dir=root)
    output_root = _resolve_path(args.output_root, base_dir=root)
    local_msa_root = _resolve_path(args.local_msa_root, base_dir=root)
    boltz_src_path = None
    if args.msa_backend == "server":
        boltz_src_path = (
            _resolve_path(args.boltz_src_path, base_dir=root)
            if args.boltz_src_path
            else _infer_boltz_src_path(local_msa_root)
        )
    msa_cache_dir = (
        _resolve_path(args.msa_cache_dir, base_dir=root)
        if args.msa_cache_dir
        else (output_root / "msas").resolve()
    )
    requested_states = (
        ("reactant", "product") if args.states == "both" else (args.states,)
    )
    msa_depth = None if int(args.msa_depth) <= 0 else int(args.msa_depth)
    cuda_devices = _parse_cuda_devices(args.cuda_devices)

    logger.info("Loading RF3 examples from %s", input_root)
    payloads_by_state = {
        state: _list_payload_paths_for_state(input_root, state) for state in requested_states
    }
    max_docked_pairs = None if int(args.max_docked_pairs) <= 0 else int(args.max_docked_pairs)
    selected_pair_ids, invalid_pair_reason, invalid_pair_counts, scanned_pair_count = _select_valid_pair_ids(
        input_root,
        requested_states,
        max_docked_pairs,
        payloads_by_state=payloads_by_state,
    )

    sequences: list[str] = []
    existing_msa_count = 0
    state_example_counts: dict[str, int] = {}
    for state, (_, paths) in payloads_by_state.items():
        state_count = 0
        for example in _iter_examples_from_payloads(paths, selected_pair_ids):
            protein = _protein_component(example)
            sequence = str(protein.get("seq") or "").strip()
            if not sequence:
                raise ValueError(f"Protein component is missing seq for example {example.get('name')}")
            sequences.append(sequence)
            state_count += 1
            msa_path = protein.get("msa_path")
            if msa_path and Path(str(msa_path)).exists():
                existing_msa_count += 1
        state_example_counts[state] = state_count

    unique_sequences = list(dict.fromkeys(sequences))
    cached_unique_sequences = (
        sum(
            1
            for sequence in unique_sequences
            if _msa_cache_path_for_sequence(msa_cache_dir, sequence).exists()
        )
        if args.reuse_cache
        else 0
    )
    missing_unique_sequences = len(unique_sequences) - cached_unique_sequences
    selected_pair_count = None if selected_pair_ids is None else len(selected_pair_ids)

    logger.info(
        "Preparing MSAs for %d examples (selected_pairs=%s scanned_candidate_pairs=%d unique_sequences=%d cached=%d missing=%d already_have_msa_path=%d backend=%s skipped_invalid_pairs=%d)",
        sum(state_example_counts.values()),
        "unlimited" if selected_pair_count is None else selected_pair_count,
        scanned_pair_count,
        len(unique_sequences),
        cached_unique_sequences,
        missing_unique_sequences,
        existing_msa_count,
        args.msa_backend,
        len(invalid_pair_reason),
    )

    t0 = time.perf_counter()
    seq_to_msa = _generate_msas(
        sequences=sequences,
        msa_backend=args.msa_backend,
        local_msa_root=local_msa_root,
        boltz_src_path=boltz_src_path,
        msa_cache_dir=msa_cache_dir,
        host_url=args.msa_server_url,
        use_env=bool(args.use_env_db),
        use_filter=bool(args.use_filter),
        pairing_strategy=args.pairing_strategy,
        reuse_cache=bool(args.reuse_cache),
        msa_batch_size=int(args.msa_batch_size),
        msa_concurrency=int(args.msa_concurrency),
        msa_retries=int(args.msa_retries),
        msa_depth=msa_depth,
        cuda_devices=cuda_devices,
        max_seqs=int(args.mmseqs_max_seqs),
        num_iterations=int(args.mmseqs_num_iterations),
        msa_threads_per_job=int(args.msa_threads_per_job),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if (input_root / "manifest.jsonl").exists():
        _copy_manifest_with_pair_filter(
            input_root / "manifest.jsonl",
            output_root / "manifest.jsonl",
            selected_pair_ids=selected_pair_ids,
        )

    written_examples = 0
    written_payloads = 0
    for state, (layout, paths) in payloads_by_state.items():
        if layout == "examples":
            updated: list[dict] = []
            for path in paths:
                for example in _iter_examples_from_payloads([path], selected_pair_ids):
                    protein = _protein_component(example)
                    sequence = str(protein["seq"]).strip()
                    msa_path = seq_to_msa.get(sequence)
                    if msa_path is None:
                        raise RuntimeError(f"Missing generated MSA for example {example.get('name')}")
                    protein["msa_path"] = str(msa_path)
                    updated.append(example)
                    _write_json(output_root / "examples" / state / f"{example['name']}.json", example)
                    written_examples += 1
            _write_json(output_root / f"{state}.json", updated)
            for shard_idx, shard in enumerate(_shard_round_robin(updated, args.shards)):
                _write_json(output_root / "shards" / state / f"shard_{shard_idx:03d}.json", shard)
                written_payloads += 1
            continue

        if layout == "state_json":
            updated: list[dict] = []
            for path in paths:
                for example in _iter_examples_from_payloads([path], selected_pair_ids):
                    protein = _protein_component(example)
                    sequence = str(protein["seq"]).strip()
                    msa_path = seq_to_msa.get(sequence)
                    if msa_path is None:
                        raise RuntimeError(f"Missing generated MSA for example {example.get('name')}")
                    protein["msa_path"] = str(msa_path)
                    updated.append(example)
                    written_examples += 1
            _write_json(output_root / f"{state}.json", updated)
            for shard_idx, shard in enumerate(_shard_round_robin(updated, args.shards)):
                _write_json(output_root / "shards" / state / f"shard_{shard_idx:03d}.json", shard)
                written_payloads += 1
            continue

        if layout == "shards":
            for path in paths:
                updated = []
                for example in _iter_examples_from_payloads([path], selected_pair_ids):
                    protein = _protein_component(example)
                    sequence = str(protein["seq"]).strip()
                    msa_path = seq_to_msa.get(sequence)
                    if msa_path is None:
                        raise RuntimeError(f"Missing generated MSA for example {example.get('name')}")
                    protein["msa_path"] = str(msa_path)
                    updated.append(example)
                    written_examples += 1
                if updated:
                    _write_json(output_root / "shards" / state / path.name, updated)
                    written_payloads += 1
            continue

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "requested_states": list(requested_states),
        "msa_backend": args.msa_backend,
        "local_msa_root": str(local_msa_root),
        "boltz_src_path": None if boltz_src_path is None else str(boltz_src_path),
        "msa_cache_dir": str(msa_cache_dir),
        "msa_server_url": args.msa_server_url,
        "cuda_devices": cuda_devices,
        "msa_depth": None if msa_depth is None else int(msa_depth),
        "reuse_cache": bool(args.reuse_cache),
        "use_env_db": bool(args.use_env_db),
        "use_filter": bool(args.use_filter),
        "pairing_strategy": args.pairing_strategy,
        "msa_batch_size": int(args.msa_batch_size),
        "msa_concurrency": int(args.msa_concurrency),
        "msa_retries": int(args.msa_retries),
        "mmseqs_max_seqs": int(args.mmseqs_max_seqs),
        "mmseqs_num_iterations": int(args.mmseqs_num_iterations),
        "msa_threads_per_job": int(args.msa_threads_per_job),
        "max_docked_pairs": None if max_docked_pairs is None else int(max_docked_pairs),
        "counts": {
            "states": state_example_counts,
            "written_examples": written_examples,
            "written_payloads": written_payloads,
            "unique_sequences": len(set(sequences)),
            "cached_unique_sequences": cached_unique_sequences,
            "missing_unique_sequences": missing_unique_sequences,
            "scanned_candidate_pairs": scanned_pair_count,
            "invalid_pairs_total": len(invalid_pair_reason),
            "invalid_pair_dummy_atom": int(invalid_pair_counts["invalid_pair_dummy_atom"]),
            "invalid_pair_non_embeddable": int(
                invalid_pair_counts["invalid_pair_non_embeddable"]
            ),
            "selected_pairs": selected_pair_count,
        },
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    _write_json(output_root / "summary.json", summary)
    logger.info(
        "Wrote RF3 inputs with msa_path to %s (written_examples=%d elapsed=%.2fs)",
        output_root,
        written_examples,
        summary["elapsed_sec"],
    )
    print(output_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
