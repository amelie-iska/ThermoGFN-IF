#!/usr/bin/env python3
"""Score candidates with SPURS single/multi-mutant oracle branch (production)."""

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
    from spurs.inference import get_SPURS_from_hub, parse_pdb
    from train.thermogfn.progress import iter_progress

    load_t0 = time.perf_counter()
    logger.info("SPURS model load start: repo_id=%s", repo_id)
    model, cfg = get_SPURS_from_hub(repo_id=repo_id)
    model.eval()
    logger.info("SPURS model load complete: elapsed=%.2fs", time.perf_counter() - load_t0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    cache_dir = _repo_root() / ".cache" / "spurs_structures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("SPURS scoring loop start: candidates=%d default_chain=%s cache_dir=%s", len(candidates), chain, cache_dir)

    cache: dict[str, tuple] = {}
    out = []
    for i, rec in enumerate(iter_progress(candidates, total=len(candidates), desc="spurs:score", no_progress=no_progress), start=1):
        cif = rec.get("cif_path")
        if not cif:
            raise ValueError(f"candidate {rec.get('candidate_id')} missing cif_path")

        try:
            structure_path = _prepare_structure_for_spurs(Path(str(cif)).resolve(), cache_dir)
            chain_pref = rec.get("chain_id") or rec.get("chain") or chain or "A"
            key = "%s|%s" % (str(structure_path), str(chain_pref))
            if key not in cache:
                chains = _chain_candidates(rec, chain)
                pdb, used_chain = _parse_with_fallback(
                    parse_pdb=parse_pdb,
                    structure_path=structure_path,
                    pdb_name=rec.get("backbone_id", "target"),
                    chains=chains,
                    cfg=cfg,
                )
                with torch.no_grad():
                    ddg = model(pdb, return_logist=True)
                cache[key] = (ddg, used_chain, str(structure_path))
            ddg, used_chain, used_path = cache[key]
            rec["spurs_chain"] = used_chain if used_chain is not None else ""
            rec["spurs_input_path"] = used_path

            muts = rec.get("mutations", [])
            total = 0.0
            used = 0
            for mut in muts:
                if len(mut) < 3:
                    continue
                new = mut[-1]
                try:
                    pos = int(mut[1:-1]) - 1
                except Exception:
                    continue
                if pos < 0 or pos >= ddg.shape[0] or new not in alphabet:
                    continue
                total += float(ddg[pos, alphabet.index(new)].item())
                used += 1
            rec["spurs_mean"] = total if used > 0 else 0.0
            rec["spurs_std"] = 0.1 + 0.03 * max(0, rec.get("K", 0) - 1)
            if rec.get("K", 0) <= 1:
                rec["spurs_mode"] = "single"
            elif rec.get("K", 0) == 2:
                rec["spurs_mode"] = "double"
            else:
                rec["spurs_mode"] = "higher"
            rec["spurs_status"] = "ok"
            out.append(rec)
            if i == 1 or (i % 2000) == 0 or i == len(candidates):
                logger.info("SPURS progress: %d/%d", i, len(candidates))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SPURS scoring failed candidate_id=%s cif=%s err=%s"
                % (
                    rec.get("candidate_id"),
                    cif,
                    exc,
                )
            ) from exc
    logger.info("SPURS scoring completed with no parse/model errors cache_entries=%d", len(cache))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repo-id", default="cyclization9/SPURS")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging

    logger = configure_logging("oracle.spurs_single", level=args.log_level)
    candidates = read_records(root / args.candidate_path)
    logger.info("Scoring candidates=%d with SPURS single/double branch", len(candidates))

    scored = _score_production(candidates, chain=args.chain, repo_id=args.repo_id, logger=logger, no_progress=args.no_progress)

    write_records(root / args.output_path, scored)
    logger.info("SPURS scoring complete: wrote=%d elapsed=%.2fs", len(scored), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
