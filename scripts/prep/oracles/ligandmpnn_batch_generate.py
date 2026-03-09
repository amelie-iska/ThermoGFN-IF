#!/usr/bin/env python3
"""Batch LigandMPNN sequence generation with one model load.

This worker prepares PDB inputs (converting CIF/mmCIF when required), runs
LigandMPNN once with --pdb_path_multi, and emits one generated sequence per item.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from prody import parseMMCIF, writePDB

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = list(iterable) if iterable is not None else None
            self._total = int(kwargs.get("total") or (len(self._iterable) if self._iterable is not None else 0))
            self._desc = str(kwargs.get("desc") or "progress")
            self._count = 0
            self._step = max(1, self._total // 20)
            print(f"[{self._desc}] start total={self._total}", flush=True)

        def __iter__(self):
            if self._iterable is None:
                return
            for item in self._iterable:
                self._count += 1
                if self._count == 1 or self._count == self._total or (self._count % self._step) == 0:
                    print(f"[{self._desc}] {self._count}/{self._total}", flush=True)
                yield item

        def update(self, n: int = 1) -> None:
            self._count += int(max(0, n))
            if self._count == 1 or self._count >= self._total or (self._count % self._step) == 0:
                shown = min(self._count, self._total)
                print(f"[{self._desc}] {shown}/{self._total}", flush=True)

        def set_postfix_str(self, _value: str) -> None:
            return None

        def refresh(self) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return _TqdmFallback(iterable, **kwargs)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _prepare_pdb(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".pdb":
        shutil.copy2(source, dest)
        return
    if suffix in {".cif", ".mmcif"}:
        structure = parseMMCIF(str(source))
        atom = structure.select("not water and not hydrogen")
        if atom is None:
            raise RuntimeError(f"failed to parse atoms from {source}")
        for chain in atom.getHierView():
            chain_id = chain.getChid().split(".")[-1]
            chain.setChids(chain_id)
        writePDB(str(dest), atom)
        return
    raise RuntimeError(f"unsupported structure extension: {source}")


def _parse_generated_sequence(fasta_path: Path) -> str:
    headers: list[str] = []
    seqs: list[str] = []
    cur_header: str | None = None
    cur_seq_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    headers.append(cur_header)
                    seqs.append("".join(cur_seq_parts))
                cur_header = line
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
    if cur_header is not None:
        headers.append(cur_header)
        seqs.append("".join(cur_seq_parts))

    if not headers:
        raise RuntimeError(f"no fasta entries in {fasta_path}")

    # Generated samples contain "id=" in the header. Native line does not.
    for h, s in zip(headers, seqs):
        if "id=" in h and s:
            return s
    # Fallback: if only one sequence exists, use it.
    for s in seqs:
        if s:
            return s
    raise RuntimeError(f"no non-empty sequence in {fasta_path}")


def _count_generated_fastas(seq_dir: Path) -> int:
    if not seq_dir.exists():
        return 0
    return sum(1 for p in seq_dir.glob("*.fa") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--ligandmpnn-root", required=True)
    ap.add_argument("--model-type", default="ligand_mpnn")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--number-of-batches", type=int, default=1)
    ap.add_argument("--checkpoint-ligand-mpnn", default="")
    ap.add_argument("--ligand-mpnn-use-atom-context", type=int, default=1)
    ap.add_argument("--parse-atoms-with-zero-occupancy", type=int, default=0)
    ap.add_argument("--run-heartbeat-sec", type=float, default=5.0)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    ligand_root = Path(args.ligandmpnn_root).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if not ligand_root.exists():
        raise FileNotFoundError(f"LigandMPNN root not found: {ligand_root}")
    if not (ligand_root / "run.py").exists():
        raise FileNotFoundError(f"LigandMPNN run.py not found in {ligand_root}")

    items = _load_jsonl(input_path)
    print(f"[ligandmpnn-batch] loaded inputs={len(items)}", flush=True)

    run_dir = output_path.parent / f"ligandmpnn_tmp_{int(time.time())}"
    prepared_dir = run_dir / "prepared_pdbs"
    outputs_dir = run_dir / "outputs"
    mapping_json = run_dir / "pdb_path_multi.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    idx_to_meta: dict[int, dict] = {}
    pdb_map: dict[str, str] = {}
    prep_bar = tqdm(
        items,
        total=len(items),
        desc="ligandmpnn:prepare",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=args.no_progress,
    )
    for item in prep_bar:
        idx = int(item["idx"])
        src = Path(item["cif_path"]).resolve()
        if not src.exists():
            raise FileNotFoundError(f"missing structure file idx={idx}: {src}")
        stem = str(item.get("stem") or src.stem)
        safe = _safe_name(stem)
        dest = prepared_dir / f"item_{idx:07d}_{safe}.pdb"
        _prepare_pdb(src, dest)
        idx_to_meta[idx] = {
            "idx": idx,
            "source_path": str(src),
            "prepared_pdb": str(dest),
            "name": dest.stem,
        }
        pdb_map[str(dest)] = ""
        prep_bar.set_postfix_str(f"idx={idx} file={dest.name}")

    with mapping_json.open("w", encoding="utf-8") as fh:
        json.dump(pdb_map, fh, indent=2, sort_keys=True)

    cmd = [
        "python",
        "run.py",
        "--model_type",
        args.model_type,
        "--pdb_path_multi",
        str(mapping_json),
        "--out_folder",
        str(outputs_dir),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--batch_size",
        str(args.batch_size),
        "--number_of_batches",
        str(args.number_of_batches),
        "--verbose",
        "0",
        "--ligand_mpnn_use_atom_context",
        str(args.ligand_mpnn_use_atom_context),
        "--parse_atoms_with_zero_occupancy",
        str(args.parse_atoms_with_zero_occupancy),
    ]
    if args.checkpoint_ligand_mpnn:
        cmd.extend(["--checkpoint_ligand_mpnn", str(Path(args.checkpoint_ligand_mpnn).resolve())])

    print(f"[ligandmpnn-batch] launching run.py for {len(items)} structures", flush=True)
    run_start = time.perf_counter()
    heartbeat_sec = max(0.5, float(args.run_heartbeat_sec))
    seq_dir = outputs_dir / "seqs"
    run_bar = tqdm(
        total=len(items),
        desc="ligandmpnn:run",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=args.no_progress,
    )
    run_count = 0
    next_heartbeat = run_start + heartbeat_sec
    proc = subprocess.Popen(cmd, cwd=str(ligand_root))  # noqa: S603
    while True:
        rc = proc.poll()
        current_count = _count_generated_fastas(seq_dir)
        if current_count > run_count:
            run_bar.update(current_count - run_count)
            run_count = current_count
        elapsed = time.perf_counter() - run_start
        run_bar.set_postfix_str(f"done={run_count}/{len(items)} elapsed={elapsed:.0f}s")
        run_bar.refresh()
        now = time.perf_counter()
        if args.no_progress and now >= next_heartbeat:
            print(
                f"[ligandmpnn:run] done={run_count}/{len(items)} elapsed={elapsed:.0f}s",
                flush=True,
            )
            next_heartbeat = now + heartbeat_sec
        if rc is not None:
            break
        time.sleep(heartbeat_sec)

    final_count = _count_generated_fastas(seq_dir)
    if final_count > run_count:
        run_bar.update(final_count - run_count)
        run_count = final_count
    total_run_elapsed = time.perf_counter() - run_start
    run_bar.set_postfix_str(f"done={run_count}/{len(items)} elapsed={total_run_elapsed:.0f}s")
    run_bar.close()
    print(
        f"[ligandmpnn-batch] run.py finished rc={rc} generated={run_count}/{len(items)} elapsed={total_run_elapsed:.2f}s",
        flush=True,
    )
    if rc != 0:
        raise RuntimeError(f"LigandMPNN run.py failed rc={rc}")

    out_rows: list[dict] = []
    parse_bar = tqdm(
        sorted(idx_to_meta.items()),
        total=len(idx_to_meta),
        desc="ligandmpnn:parse",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=args.no_progress,
    )
    for idx, meta in parse_bar:
        fasta_path = outputs_dir / "seqs" / f"{meta['name']}.fa"
        if not fasta_path.exists():
            raise FileNotFoundError(f"missing fasta output idx={idx}: {fasta_path}")
        seq = _parse_generated_sequence(fasta_path)
        out_rows.append(
            {
                "idx": idx,
                "source_path": meta["source_path"],
                "prepared_pdb": meta["prepared_pdb"],
                "sequence": seq,
                "ok": True,
            }
        )
        parse_bar.set_postfix_str(f"idx={idx} len={len(seq)}")

    out_rows.sort(key=lambda x: int(x["idx"]))
    _write_jsonl(output_path, out_rows)
    elapsed = time.perf_counter() - t0
    print(f"[ligandmpnn-batch] wrote outputs={len(out_rows)} path={output_path}", flush=True)
    print(f"[ligandmpnn-batch] total={elapsed:.2f}s avg={elapsed/max(1,len(out_rows)):.2f}s/item", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
