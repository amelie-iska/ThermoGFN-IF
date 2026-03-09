#!/usr/bin/env python3
"""Score candidates with SPURS multi-mutant branch (production)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _chain_candidates(rec: dict, default_chain: str | None) -> list[str | None]:
    candidates = [
        rec.get("chain_id"),
        rec.get("chain"),
        default_chain,
        "A",
        None,
    ]
    out: list[str | None] = []
    seen: set[str] = set()
    for c in candidates:
        if c is None:
            key = "__none__"
        else:
            c = str(c).strip()
            if not c:
                continue
            key = c
        if key in seen:
            continue
        seen.add(key)
        out.append(None if key == "__none__" else key)
    return out


def _cif_to_pdb(src: Path, cache_dir: Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO

    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:12]
    dst = cache_dir / f"{src.stem}_{digest}.pdb"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(src.stem, str(src))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst))
    return dst


def _prepare_structure_for_spurs(cif_or_pdb: Path, cache_dir: Path) -> Path:
    suffix = cif_or_pdb.suffix.lower()
    if suffix == ".pdb":
        return cif_or_pdb
    if suffix in {".cif", ".mmcif"}:
        return _cif_to_pdb(cif_or_pdb, cache_dir)
    if suffix == ".gz" and cif_or_pdb.name.lower().endswith(".cif.gz"):
        digest = hashlib.sha1(str(cif_or_pdb).encode("utf-8")).hexdigest()[:12]
        tmp_cif = cache_dir / f"{cif_or_pdb.stem}_{digest}.cif"
        if not tmp_cif.exists() or tmp_cif.stat().st_mtime < cif_or_pdb.stat().st_mtime:
            with gzip.open(cif_or_pdb, "rb") as src, open(tmp_cif, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return _cif_to_pdb(tmp_cif, cache_dir)
    return cif_or_pdb


def _parse_with_fallback(parse_pdb, structure_path: Path, pdb_name: str, chains: list[str | None], cfg):
    errs = []
    for ch in chains:
        try:
            return parse_pdb(str(structure_path), pdb_name, ch, cfg), ch
        except Exception as exc:  # noqa: BLE001
            errs.append("chain=%r err=%s" % (ch, exc))
    raise RuntimeError("; ".join(errs))


def _score_production(
    candidates: list[dict],
    chain: str,
    repo_id: str,
    logger,
    *,
    no_progress: bool = False,
) -> list[dict]:
    import torch
    from spurs.inference import get_SPURS_multi_from_hub, parse_pdb, parse_pdb_for_mutation
    from train.thermogfn.progress import iter_progress

    model, cfg = get_SPURS_multi_from_hub(repo_id=repo_id)
    model.eval()
    cache_dir = _repo_root() / ".cache" / "spurs_structures"
    cache_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[int]] = {}
    for i, rec in enumerate(iter_progress(candidates, total=len(candidates), desc="spurs:group", no_progress=no_progress)):
        cif = rec.get("cif_path")
        if not cif or rec.get("K", 0) < 2:
            continue
        grouped.setdefault(str(cif), []).append(i)

    grouped_items = list(grouped.items())
    for cif, idxs in iter_progress(grouped_items, total=len(grouped_items), desc="spurs:multi", no_progress=no_progress):
        try:
            example = candidates[idxs[0]]
            structure_path = _prepare_structure_for_spurs(Path(cif).resolve(), cache_dir)
            chains = _chain_candidates(example, chain)
            pdb, used_chain = _parse_with_fallback(
                parse_pdb=parse_pdb,
                structure_path=structure_path,
                pdb_name=example.get("backbone_id", "target"),
                chains=chains,
                cfg=cfg,
            )
            mut_lists = [candidates[i].get("mutations", []) for i in idxs]
            mut_ids, append_tensors = parse_pdb_for_mutation(mut_lists)
            pdb["mut_ids"] = mut_ids
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pdb["append_tensors"] = append_tensors.to(device)
            with torch.no_grad():
                ddg = model(pdb)
            for j, i in enumerate(idxs):
                candidates[i]["spurs_multi_mean"] = float(ddg[j].item())
                candidates[i]["spurs_multi_std"] = 0.12
                candidates[i]["spurs_multi_chain"] = used_chain if used_chain is not None else ""
                candidates[i]["spurs_multi_input_path"] = str(structure_path)
                candidates[i]["spurs_multi_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("SPURS multi scoring failed group_cif=%s n=%d err=%s" % (cif, len(idxs), exc)) from exc

    for rec in iter_progress(candidates, total=len(candidates), desc="spurs:fill", no_progress=no_progress):
        if "spurs_multi_mean" not in rec:
            # For K<2 or non-grouped rows keep single-model score as baseline.
            rec["spurs_multi_mean"] = float(rec.get("spurs_mean", 0.0))
            rec["spurs_multi_std"] = float(rec.get("spurs_std", 0.1))
            rec["spurs_multi_status"] = rec.get("spurs_multi_status", "derived_single")
    logger.info("SPURS multi scoring completed with no parse/model errors")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repo-id", default="cyclization9/SPURS")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("oracle.spurs_multi", level=args.log_level)
    rows = read_records(root / args.candidate_path)
    logger.info("Scoring candidates=%d with SPURS multi branch", len(rows))
    rows = _score_production(rows, chain=args.chain, repo_id=args.repo_id, logger=logger, no_progress=args.no_progress)

    write_records(root / args.output_path, rows)
    logger.info("SPURS multi scoring complete: wrote=%d elapsed=%.2fs", len(rows), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
