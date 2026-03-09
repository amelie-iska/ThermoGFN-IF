#!/usr/bin/env python3
"""Profile BioEmu VRAM on 5 increasing-length systems and project max capacity.

Strict behavior:
- Uses torch CUDA telemetry recorded by scripts/prep/oracles/bioemu_sample_and_features.py
- Fails if CUDA telemetry is unavailable or missing for any profiled sample.
- No non-torch VRAM fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


@dataclass
class ProfileRow:
    candidate_id: str
    length: int
    effective_batch: int
    peak_alloc_bytes: int
    peak_reserved_bytes: int
    total_vram_bytes: int
    peak_frac_alloc: float
    peak_frac_reserved: float


def _hash_sequence(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _parse_targets(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 5:
        raise ValueError(f"--length-targets must provide exactly 5 integers, got: {s}")
    return vals


def _select_five(
    rows: list[dict],
    length_targets: list[int],
    embeds_cache_dir: Path,
    require_cached_embeds: bool,
) -> list[dict]:
    candidates = []
    for r in rows:
        seq = r.get("sequence", "")
        if not seq:
            continue
        if not r.get("eligibility", {}).get("bioemu", False):
            continue
        if not r.get("cif_path"):
            continue
        if require_cached_embeds:
            h = _hash_sequence(seq)
            s = embeds_cache_dir / f"{h}_single.npy"
            p = embeds_cache_dir / f"{h}_pair.npy"
            if not (s.exists() and p.exists()):
                continue
        candidates.append(r)

    if not candidates:
        raise RuntimeError("No eligible candidates found after filtering")

    selected: list[dict] = []
    used_ids: set[str] = set()
    for t in length_targets:
        best = None
        best_key = None
        for r in candidates:
            cid = str(r.get("candidate_id", ""))
            if not cid or cid in used_ids:
                continue
            L = len(r["sequence"])
            key = (abs(L - t), L)
            if best_key is None or key < best_key:
                best = r
                best_key = key
        if best is None:
            raise RuntimeError(f"Could not select profile candidate near length target {t}")
        used_ids.add(str(best["candidate_id"]))
        selected.append(best)
    return selected


def _effective_batch(batch_size_100: int, length: int) -> int:
    return max(1, int(batch_size_100 * (100.0 / float(length)) ** 2))


def _profile_and_project(
    root: Path,
    selected_path: Path,
    scored_path: Path,
    batch_size_100: int,
    num_samples: int,
    model_name: str,
    target_vram_frac: float,
    log_level: str,
    filter_samples: bool,
    env: dict[str, str],
) -> tuple[list[ProfileRow], dict, list[dict]]:
    scorer = root / "scripts" / "prep" / "oracles" / "bioemu_sample_and_features.py"
    selected = _load_jsonl(selected_path)
    if len(selected) != 5:
        raise RuntimeError(f"Expected 5 selected rows, got {len(selected)}")

    per_candidate_records: list[dict] = []
    failures: list[dict] = []
    for i, row in enumerate(selected):
        one_in = scored_path.parent / f"_candidate_{i:02d}.jsonl"
        one_out = scored_path.parent / f"_candidate_{i:02d}_scored.jsonl"
        _write_jsonl(one_in, [row])
        cmd = [
            sys.executable,
            str(scorer),
            "--candidate-path",
            str(one_in),
            "--output-path",
            str(one_out),
            "--model-name",
            model_name,
            "--num-samples",
            str(num_samples),
            "--batch-size-100",
            str(batch_size_100),
            "--no-auto-batch-from-vram",
            "--require-torch-cuda-vram",
            "--filter-samples" if filter_samples else "--no-filter-samples",
            "--log-level",
            log_level,
        ]
        try:
            subprocess.run(cmd, check=True, env=env)
            scored_one = _load_jsonl(one_out)
            if len(scored_one) != 1:
                raise RuntimeError(f"Expected 1 row for candidate {i}, got {len(scored_one)}")
            per_candidate_records.append(scored_one[0])
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "length": len(row.get("sequence", "")),
                    "error": str(exc),
                }
            )
        finally:
            if one_in.exists():
                one_in.unlink()

    _write_jsonl(scored_path, per_candidate_records)
    scored = per_candidate_records
    if len(scored) == 0:
        raise RuntimeError(f"No successful profile rows. failures={failures}")

    prof: list[ProfileRow] = []
    for r in scored:
        if r.get("bioemu_status") != "ok":
            raise RuntimeError(f"Profile row failed: candidate_id={r.get('candidate_id')} status={r.get('bioemu_status')}")
        if r.get("bioemu_vram_source") != "torch.cuda":
            raise RuntimeError(
                f"VRAM source is not torch.cuda for candidate_id={r.get('candidate_id')}: {r.get('bioemu_vram_source')}"
            )
        peak_alloc = r.get("bioemu_peak_vram_bytes")
        peak_reserved = r.get("bioemu_peak_reserved_vram_bytes")
        total = r.get("bioemu_total_vram_bytes")
        if peak_alloc is None or peak_reserved is None or total is None:
            raise RuntimeError(f"Missing strict VRAM fields for candidate_id={r.get('candidate_id')}")
        L = len(r.get("sequence", ""))
        eff_b = _effective_batch(batch_size_100=batch_size_100, length=L)
        prof.append(
            ProfileRow(
                candidate_id=str(r.get("candidate_id")),
                length=L,
                effective_batch=eff_b,
                peak_alloc_bytes=int(peak_alloc),
                peak_reserved_bytes=int(peak_reserved),
                total_vram_bytes=int(total),
                peak_frac_alloc=float(peak_alloc) / float(total),
                peak_frac_reserved=float(peak_reserved) / float(total),
            )
        )

    total_vram = prof[0].total_vram_bytes
    if any(p.total_vram_bytes != total_vram for p in prof):
        raise RuntimeError("Inconsistent total VRAM across profile rows")

    # Conservative density model based on reserved memory:
    # peak_reserved ~= alpha * (L^2 * effective_batch) + beta ; use beta=0 conservative.
    dens = [
        p.peak_reserved_bytes / max(1.0, float(p.length * p.length * p.effective_batch))
        for p in prof
    ]
    alpha = max(dens)  # conservative upper envelope
    budget = float(total_vram) * float(target_vram_frac)

    # Project max length for effective_batch=1.
    max_length_batch1 = int(max(1.0, math.floor(math.sqrt(budget / alpha))))

    # Project max batch_size_100 at L=100.
    max_batch_size_100 = int(max(1.0, math.floor(budget / (alpha * 100.0 * 100.0))))

    projection = {
        "target_vram_frac": target_vram_frac,
        "total_vram_bytes": int(total_vram),
        "alpha_reserved_bytes_per_L2B": alpha,
        "projected_max_length_at_effective_batch_1": max_length_batch1,
        "projected_max_batch_size_100_at_length_100": max_batch_size_100,
    }
    return prof, projection, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="runs/thermogfn_ligandmpnn/bootstrap/D_0_train.jsonl",
        help="Input candidate/spec records with sequence/cif_path/eligibility",
    )
    parser.add_argument(
        "--output-root",
        default="runs/bioemu_vram_profile",
    )
    parser.add_argument(
        "--length-targets",
        default="100,180,260,340,400",
        help="Exactly 5 comma-separated residue targets",
    )
    parser.add_argument("--batch-size-100", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--model-name", default="bioemu-v1.1")
    parser.add_argument("--target-vram-frac", type=float, default=0.90)
    parser.add_argument("--filter-samples", dest="filter_samples", action="store_true")
    parser.add_argument("--no-filter-samples", dest="filter_samples", action="store_false")
    parser.add_argument("--require-cached-embeds", action="store_true")
    parser.add_argument("--ensure-colabfold-runtime", dest="ensure_colabfold_runtime", action="store_true")
    parser.add_argument("--no-ensure-colabfold-runtime", dest="ensure_colabfold_runtime", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    parser.set_defaults(ensure_colabfold_runtime=True, filter_samples=False)
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    out_root = (root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Normalize cache/runtime environment for direct python execution.
    env = os.environ.copy()
    hf_home = Path(env.setdefault("HF_HOME", str((root / ".cache" / "huggingface").resolve())))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str((hf_home / "hub").resolve()))
    env.setdefault("TRANSFORMERS_CACHE", str((hf_home / "transformers").resolve()))
    env.setdefault("TORCH_HOME", str((hf_home / "torch").resolve()))
    env.setdefault("XDG_CACHE_HOME", str((root / ".cache" / "xdg").resolve()))
    env.setdefault("UV_CACHE_DIR", str((root / ".cache" / "uv").resolve()))
    env.setdefault("PIP_CACHE_DIR", str((root / ".cache" / "pip").resolve()))
    env.setdefault("BIOEMU_COLABFOLD_DIR", str((root / ".cache" / "bioemu" / "colabfold").resolve()))

    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(env["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["PIP_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["BIOEMU_COLABFOLD_DIR"]).mkdir(parents=True, exist_ok=True)

    if args.ensure_colabfold_runtime:
        setup_script = root / "scripts" / "env" / "setup_bioemu_colabfold_runtime.sh"
        if not setup_script.exists():
            raise FileNotFoundError(f"Missing runtime setup script: {setup_script}")
        subprocess.run(
            ["bash", str(setup_script), "--bioemu-env", "bioemu"],
            check=True,
            env=env,
        )

    input_path = (root / args.input_jsonl).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input-jsonl not found: {input_path}")
    targets = _parse_targets(args.length_targets)

    embeds_cache = root / ".cache" / "bioemu" / "embeds_cache"
    rows = _load_jsonl(input_path)
    selected = _select_five(
        rows=rows,
        length_targets=targets,
        embeds_cache_dir=embeds_cache,
        require_cached_embeds=args.require_cached_embeds,
    )

    selected_path = out_root / "selected_5.jsonl"
    scored_path = out_root / "scored_5.jsonl"
    report_json = out_root / "projection_report.json"
    report_md = out_root / "projection_report.md"
    _write_jsonl(selected_path, selected)

    prof, projection, failures = _profile_and_project(
        root=root,
        selected_path=selected_path,
        scored_path=scored_path,
        batch_size_100=args.batch_size_100,
        num_samples=args.num_samples,
        model_name=args.model_name,
        target_vram_frac=args.target_vram_frac,
        log_level=args.log_level,
        filter_samples=args.filter_samples,
        env=env,
    )

    prof_sorted = sorted(prof, key=lambda x: x.length)
    payload = {
        "input_jsonl": str(input_path),
        "selected_path": str(selected_path),
        "scored_path": str(scored_path),
        "batch_size_100": args.batch_size_100,
        "num_samples": args.num_samples,
        "model_name": args.model_name,
        "rows": [
            {
                "candidate_id": p.candidate_id,
                "length": p.length,
                "effective_batch": p.effective_batch,
                "peak_alloc_bytes": p.peak_alloc_bytes,
                "peak_reserved_bytes": p.peak_reserved_bytes,
                "total_vram_bytes": p.total_vram_bytes,
                "peak_frac_alloc": p.peak_frac_alloc,
                "peak_frac_reserved": p.peak_frac_reserved,
            }
            for p in prof_sorted
        ],
        "failures": failures,
        "projection": projection,
        "elapsed_sec": time.perf_counter() - t0,
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# BioEmu VRAM Profile",
        "",
        f"- input: `{input_path}`",
        f"- batch_size_100: `{args.batch_size_100}`",
        f"- num_samples: `{args.num_samples}`",
        f"- target_vram_frac: `{args.target_vram_frac}`",
        f"- filter_samples: `{args.filter_samples}`",
        "",
        "## Measured",
        "",
        "| length | effective_batch | peak_alloc_GiB | peak_reserved_GiB | peak_alloc_frac | peak_reserved_frac | candidate_id |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in prof_sorted:
        lines.append(
            f"| {p.length} | {p.effective_batch} | {p.peak_alloc_bytes / (1024**3):.3f} | "
            f"{p.peak_reserved_bytes / (1024**3):.3f} | {p.peak_frac_alloc:.3f} | {p.peak_frac_reserved:.3f} | {p.candidate_id} |"
        )
    lines += [
        "",
        "## Projection",
        "",
        f"- projected max length at effective_batch=1 (reserved-memory model): **{projection['projected_max_length_at_effective_batch_1']} residues**",
        f"- projected max batch_size_100 at length 100: **{projection['projected_max_batch_size_100_at_length_100']}**",
        f"- total_vram_GiB: **{projection['total_vram_bytes'] / (1024**3):.3f}**",
        "",
        "_Model:_ conservative upper-envelope on `peak_reserved_bytes / (L^2 * effective_batch)` from measured rows.",
    ]
    if failures:
        lines += [
            "",
            "## Failures",
            "",
        ]
        for f in failures:
            lines.append(
                f"- candidate_id={f.get('candidate_id')} length={f.get('length')} error=`{f.get('error')}`"
            )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report_json)
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
