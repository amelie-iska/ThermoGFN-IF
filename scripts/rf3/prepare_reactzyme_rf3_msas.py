#!/usr/bin/env python3
"""Attach local MMSeqs2-generated MSAs to ReactZyme RF3 JSON inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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


def _select_pair_ids(input_root: Path, requested_states: Sequence[str], max_docked_pairs: int | None) -> set[str] | None:
    if max_docked_pairs is None or max_docked_pairs <= 0:
        return None

    manifest_path = input_root / "manifest.jsonl"
    selected: list[str] = []
    seen: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pair_id = str(row.get("pair_id") or "").strip()
                if not pair_id or pair_id in seen:
                    continue
                seen.add(pair_id)
                selected.append(pair_id)
                if len(selected) >= max_docked_pairs:
                    return set(selected)

    for state in requested_states:
        _, paths = _list_payload_paths_for_state(input_root, state)
        for path in paths:
            for example in _load_examples_from_path(path):
                pair_id = _example_pair_id(example)
                if not pair_id or pair_id in seen:
                    continue
                seen.add(pair_id)
                selected.append(pair_id)
                if len(selected) >= max_docked_pairs:
                    return set(selected)
    return set(selected)


def _iter_examples_from_payloads(paths: Iterable[Path], selected_pair_ids: set[str] | None) -> Iterable[dict]:
    for path in paths:
        for example in _load_examples_from_path(path):
            if selected_pair_ids is not None and _example_pair_id(example) not in selected_pair_ids:
                continue
            yield example


def _protein_component(example: dict) -> dict:
    components = example.get("components") or []
    proteins = [component for component in components if "seq" in component]
    if len(proteins) != 1:
        raise ValueError(
            f"Expected exactly one protein component in example {example.get('name')}, found {len(proteins)}"
        )
    return proteins[0]


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


def _count_a3m_sequences(a3m_text: str) -> int:
    return sum(1 for line in a3m_text.splitlines() if line.startswith(">"))


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


def _generate_msas(
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
        seq_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
        msa_path = msa_cache_dir / f"{seq_hash}.a3m"
        if reuse_cache and msa_path.exists() and msa_path.stat().st_size > 0:
            seq_to_path[sequence] = msa_path.resolve()
        else:
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
    total_missing = len(missing_sequences)
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
    seq_bar = tqdm(total=total_missing, desc="MSA Seqs", unit="seq")
    try:
        if max_workers <= 1:
            for idx, chunk in enumerate(chunks):
                chunk_bar.set_postfix_str(f"chunk={idx + 1}/{total_chunks} size={len(chunk)}")
                try:
                    chunk_results.append(_run_chunk(chunk, idx))
                    chunk_bar.update(1)
                    seq_bar.update(len(chunk))
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
        "--boltz-src-path",
        default=None,
        help="Optional explicit path to Boltz client src for run_mmseqs2.",
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
    parser.add_argument("--use-filter", action="store_true", help="Enable MMSeqs filtering.")
    parser.add_argument("--pairing-strategy", default="greedy", help="Pairing strategy passed to run_mmseqs2.")
    parser.add_argument("--msa-batch-size", type=int, default=64, help="Sequences per MMSeqs2 batch request.")
    parser.add_argument("--msa-concurrency", type=int, default=4, help="Number of concurrent MMSeqs2 batch requests.")
    parser.add_argument("--msa-retries", type=int, default=2, help="Retries per MMSeqs2 batch request.")
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
    args = parser.parse_args()

    root = _repo_root()
    logger = _configure_logging(args.log_level)

    input_root = _resolve_path(args.input_root, base_dir=root)
    output_root = _resolve_path(args.output_root, base_dir=root)
    local_msa_root = _resolve_path(args.local_msa_root, base_dir=root)
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

    logger.info("Loading RF3 examples from %s", input_root)
    payloads_by_state = {
        state: _list_payload_paths_for_state(input_root, state) for state in requested_states
    }
    max_docked_pairs = None if int(args.max_docked_pairs) <= 0 else int(args.max_docked_pairs)
    selected_pair_ids = _select_pair_ids(input_root, requested_states, max_docked_pairs)

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

    logger.info(
        "Preparing MSAs for %d examples (%d unique sequences, %d already have msa_path, max_docked_pairs=%s)",
        sum(state_example_counts.values()),
        len(set(sequences)),
        existing_msa_count,
        "unlimited" if max_docked_pairs is None else max_docked_pairs,
    )

    t0 = time.perf_counter()
    seq_to_msa = _generate_msas(
        sequences=sequences,
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
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if (input_root / "manifest.jsonl").exists():
        if selected_pair_ids is None:
            shutil.copy2(input_root / "manifest.jsonl", output_root / "manifest.jsonl")
        else:
            with (input_root / "manifest.jsonl").open("r", encoding="utf-8") as src, (
                output_root / "manifest.jsonl"
            ).open("w", encoding="utf-8") as dst:
                for line in src:
                    row = json.loads(line)
                    if str(row.get("pair_id") or "").strip() in selected_pair_ids:
                        dst.write(line)

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
        "local_msa_root": str(local_msa_root),
        "boltz_src_path": str(boltz_src_path),
        "msa_cache_dir": str(msa_cache_dir),
        "msa_server_url": args.msa_server_url,
        "msa_depth": None if msa_depth is None else int(msa_depth),
        "reuse_cache": bool(args.reuse_cache),
        "use_env_db": bool(args.use_env_db),
        "use_filter": bool(args.use_filter),
        "pairing_strategy": args.pairing_strategy,
        "msa_batch_size": int(args.msa_batch_size),
        "msa_concurrency": int(args.msa_concurrency),
        "msa_retries": int(args.msa_retries),
        "max_docked_pairs": None if max_docked_pairs is None else int(max_docked_pairs),
        "counts": {
            "states": state_example_counts,
            "written_examples": written_examples,
            "written_payloads": written_payloads,
            "unique_sequences": len(set(sequences)),
            "selected_pairs": None if selected_pair_ids is None else len(selected_pair_ids),
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
