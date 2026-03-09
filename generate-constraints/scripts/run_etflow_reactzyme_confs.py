#!/usr/bin/env python
"""Generate ETFlow conformer SDFs for ReactZyme reactant/product pairs."""

import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from etflow import BaseFlow
from etflow.commons.featurization import MoleculeFeaturizer
from etflow.commons.sample import batched_sampling

LOGGER = logging.getLogger("etflow_confs")
RDLogger.DisableLog("rdApp.*")


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "NA"


def make_pair_id(split: str, row_id: str, rhea_id: str, counter: Dict[str, int]) -> str:
    base = f"{split}__{row_id}"
    if rhea_id:
        base = f"{base}__{sanitize_token(rhea_id)}"
    base = sanitize_token(base)
    idx = counter.get(base, 0)
    counter[base] = idx + 1
    if idx > 0:
        return f"{base}__{idx}"
    return base


def iter_csv_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def rdkit_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol, addCoords=False)
        return mol
    except Exception:
        return None


def fragment_count(mol: Chem.Mol) -> int:
    try:
        return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))
    except Exception:
        return 1


def has_dummy_smiles(smiles: str) -> bool:
    return "*" in smiles


def count_clashes(
    mol: Chem.Mol,
    coords: np.ndarray,
    threshold: float,
    heavy_only: bool = True,
) -> int:
    if coords.size == 0:
        return 0
    if heavy_only:
        atom_indices = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1
        ]
    else:
        atom_indices = list(range(mol.GetNumAtoms()))

    if len(atom_indices) < 2:
        return 0

    idx_map = {idx: i for i, idx in enumerate(atom_indices)}
    coords_sel = coords[atom_indices]
    diff = coords_sel[:, None, :] - coords_sel[None, :, :]
    d2 = (diff**2).sum(-1)
    thr2 = threshold**2
    np.fill_diagonal(d2, np.inf)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i in idx_map and j in idx_map:
            hi = idx_map[i]
            hj = idx_map[j]
            d2[hi, hj] = np.inf
            d2[hj, hi] = np.inf

    clash_mask = d2 < thr2
    return int(clash_mask.sum() // 2)


def score_conformer(mol: Chem.Mol, coords: np.ndarray, method: str) -> Optional[float]:
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    try:
        if method == "mmff":
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                return float(ff.CalcEnergy())
        if method in {"mmff", "uff"}:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            return float(ff.CalcEnergy())
    except Exception:
        return None
    return None


def pick_best_conformer(
    mol: Chem.Mol,
    positions: np.ndarray,
    score_method: str,
) -> Tuple[int, Optional[float]]:
    if positions.shape[0] == 1 or score_method == "none":
        return 0, None
    best_idx = 0
    best_energy = None
    for idx in range(positions.shape[0]):
        energy = score_conformer(mol, positions[idx], score_method)
        if energy is None:
            continue
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_idx = idx
    return best_idx, best_energy


def write_sdf(
    mol: Chem.Mol,
    coords: np.ndarray,
    out_path: Path,
    name: str,
    pair_id: str,
    role: str,
    smiles: str,
    energy: Optional[float] = None,
) -> None:
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", name)
    mol.SetProp("pair_id", pair_id)
    mol.SetProp("role", role)
    mol.SetProp("smiles", smiles)
    if energy is not None:
        mol.SetProp("energy", f"{energy:.6f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def setup_logging(level: str, log_file: Optional[Path] = None) -> None:
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(LOGGER.handlers):
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ETFlow conformers for ReactZyme CSV splits.")
    parser.add_argument("--csv-dir", type=Path, default=Path("data/ReactZyme-CSVs"))
    parser.add_argument("--out-dir", type=Path, default=Path("output_sdf_templates"))
    parser.add_argument("--cache-dir", type=Path, default=Path("output_sdf_templates/etflow_cache"))
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--num-confs", type=int, default=5)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--n-timesteps", type=int, default=200)
    parser.add_argument("--sampler-type", type=str, default="stochastic")
    parser.add_argument("--s-churn", type=float, default=2.0)
    parser.add_argument("--t-min", type=float, default=0.2)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--score", type=str, choices=["mmff", "uff", "none"], default="mmff")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-reactions", type=int, default=None,
                        help="Stop after processing this many reaction rows total across splits")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-dir", type=Path, default=Path("output_sdf_templates/logs"))
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Optional single log file for all splits (overrides --log-dir)")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--quality",
        type=str,
        choices=["fast", "balanced", "high"],
        default="high",
        help="Preset for sampling quality; overrides sampler settings unless manually reset.",
    )
    parser.add_argument("--clash-distance", type=float, default=1.2,
                        help="Distance threshold (angstrom) for nonbonded clash detection")
    parser.add_argument("--clash-all-atoms", action="store_true",
                        help="Include hydrogens when checking clashes (default: heavy atoms only)")
    parser.add_argument("--clash-dir", type=Path, default=Path("output_sdf_templates/clashing"),
                        help="Where to write SDFs for ligands with all conformers clashing")
    args = parser.parse_args()

    if args.quality == "fast":
        args.n_timesteps = 50
        args.sampler_type = "ode"
        args.s_churn = 1.0
        args.t_min = 1.0
        args.t_max = 1.0
    elif args.quality == "balanced":
        args.n_timesteps = 100
        args.sampler_type = "stochastic"
        args.s_churn = 1.5
        args.t_min = 0.2
        args.t_max = 0.8
    elif args.quality == "high":
        args.n_timesteps = 200
        args.sampler_type = "stochastic"
        args.s_churn = 2.0
        args.t_min = 0.2
        args.t_max = 0.8

    if args.num_confs < 1:
        print("[error] --num-confs must be >= 1", file=sys.stderr)
        sys.exit(1)

    setup_logging(args.log_level, args.log_file)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("[error] No splits provided.", file=sys.stderr)
        sys.exit(1)

    if args.clean and args.out_dir.exists():
        for split in splits:
            split_dir = args.out_dir / split
            if split_dir.exists():
                for path in split_dir.glob("*"):
                    if path.is_file():
                        path.unlink()

    device_str = args.device
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    LOGGER.info("Loading ETFlow model: model=drugs-o3 device=%s cache=%s", device_str, args.cache_dir)
    LOGGER.info(
        "Sampling config: quality=%s n_timesteps=%s sampler=%s s_churn=%s t_min=%s t_max=%s std=%s",
        args.quality,
        args.n_timesteps,
        args.sampler_type,
        args.s_churn,
        args.t_min,
        args.t_max,
        args.std,
    )
    model = BaseFlow.from_default(model="drugs-o3", device=device_str, cache=str(args.cache_dir))
    model = model.to(device)
    model.eval()
    featurizer = MoleculeFeaturizer()

    non_clash_count = 0

    for split in splits:
        if args.max_reactions is not None and non_clash_count >= args.max_reactions:
            LOGGER.info("Max non-clashing reactions limit reached before split=%s; stopping.", split)
            break
        if args.log_file is None:
            split_log = args.log_dir / f"run_etflow_{split}.log"
            setup_logging(args.log_level, split_log)
        csv_path = args.csv_dir / f"reactzyme_{split}.csv"
        if not csv_path.exists():
            LOGGER.warning("Missing CSV for split %s: %s", split, csv_path)
            continue

        out_split_dir = args.out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_split_dir / "manifest.csv"
        errors_path = out_split_dir / "errors.csv"
        total_rows = count_csv_rows(csv_path) if args.max_rows is None else args.max_rows
        LOGGER.info("Processing split=%s rows=%s output=%s", split, total_rows, out_split_dir)
        split_t0 = time.perf_counter()

        with manifest_path.open("w", newline="", encoding="utf-8") as mf, errors_path.open(
            "w", newline="", encoding="utf-8"
        ) as ef:
            manifest_writer = csv.writer(mf)
            error_writer = csv.writer(ef)
            manifest_writer.writerow(
                [
                    "pair_id",
                    "row_id",
                    "rhea_id",
                    "uniprot_id",
                    "reactant_smiles",
                    "product_smiles",
                    "reactant_sdf",
                    "product_sdf",
                    "status",
                ]
            )
            error_writer.writerow(["pair_id", "role", "smiles", "reason"])

            counter: Dict[str, int] = {}
            iterator = iter_csv_rows(csv_path)
            if tqdm is not None:
                iterator = tqdm(iterator, desc=f"ETFlow {split}", total=total_rows)
            count = 0
            stats = {
                "rows": 0,
                "reactant_ok": 0,
                "product_ok": 0,
                "reactant_exists": 0,
                "product_exists": 0,
                "parse_failed": 0,
                "atom_mismatch": 0,
                "missing_smiles": 0,
                "errors": 0,
                "multi_fragment": 0,
                "gaussian_fallback": 0,
                "dummy_pairs": 0,
                "clash_filtered": 0,
                "clash_all": 0,
            }
            for row in iterator:
                if args.max_rows is not None and count >= args.max_rows:
                    break
                if args.max_reactions is not None and non_clash_count >= args.max_reactions:
                    LOGGER.info("Max non-clashing reactions limit reached (%s); stopping.", args.max_reactions)
                    break
                count += 1
                stats["rows"] += 1
                row_id = (row.get("row_id") or "").strip() or "row"
                rhea_id = (row.get("rhea_id") or "").strip()
                uniprot_id = (row.get("uniprot_id") or "").strip()
                pair_id = make_pair_id(split, row_id, rhea_id, counter)

                reactant_smiles = (row.get("reactant_smiles") or "").strip()
                product_smiles = (row.get("product_smiles") or "").strip()

                if not reactant_smiles or not product_smiles:
                    error_writer.writerow([pair_id, "both", "", "missing_smiles"])
                    stats["missing_smiles"] += 1
                    manifest_writer.writerow(
                        [
                            pair_id,
                            row_id,
                            rhea_id,
                            uniprot_id,
                            reactant_smiles,
                            product_smiles,
                            "",
                            "",
                            "missing_smiles",
                        ]
                    )
                    if count % args.log_every == 0:
                        LOGGER.info("Progress split=%s rows=%s", split, count)
                    continue

                if has_dummy_smiles(reactant_smiles) or has_dummy_smiles(product_smiles):
                    stats["dummy_pairs"] += 1
                    error_writer.writerow([pair_id, "reactant", reactant_smiles, "dummy_atom"])
                    error_writer.writerow([pair_id, "product", product_smiles, "dummy_atom"])
                    manifest_writer.writerow(
                        [
                            pair_id,
                            row_id,
                            rhea_id,
                            uniprot_id,
                            reactant_smiles,
                            product_smiles,
                            "",
                            "",
                            "dummy_atom",
                        ]
                    )
                    if count % args.log_every == 0:
                        LOGGER.info(
                            "Progress split=%s rows=%s dummy_pairs=%s",
                            split,
                            count,
                            stats["dummy_pairs"],
                        )
                    continue

                reactant_sdf = out_split_dir / f"{pair_id}__reactant.sdf"
                product_sdf = out_split_dir / f"{pair_id}__product.sdf"
                reactant_out = str(reactant_sdf)
                product_out = str(product_sdf)

                status = []
                role_results: Dict[str, Dict[str, object]] = {}

                for role, smiles, sdf_path in [
                    ("reactant", reactant_smiles, reactant_sdf),
                    ("product", product_smiles, product_sdf),
                ]:
                    if sdf_path.exists() and not args.overwrite:
                        status.append(f"{role}:exists")
                        role_results[role] = {"status": "exists", "path": str(sdf_path)}
                        if role == "reactant":
                            stats["reactant_exists"] += 1
                        else:
                            stats["product_exists"] += 1
                        continue

                    mol = rdkit_mol_from_smiles(smiles)
                    if mol is None:
                        error_writer.writerow([pair_id, role, smiles, "rdkit_parse_failed"])
                        status.append(f"{role}:parse_failed")
                        role_results[role] = {"status": "parse_failed", "path": ""}
                        stats["parse_failed"] += 1
                        continue

                    frag_count = fragment_count(mol)
                    if frag_count > 1:
                        stats["multi_fragment"] += 1
                        LOGGER.debug("Multi-fragment SMILES pair_id=%s role=%s fragments=%s",
                                     pair_id, role, frag_count)

                    prior_before = getattr(model, "prior_type", None)
                    if frag_count > 1:
                        # Harmonic prior can produce NaNs on disconnected graphs.
                        if prior_before != "gaussian":
                            model.prior_type = "gaussian"
                            stats["gaussian_fallback"] += 1
                            LOGGER.debug("Switching to gaussian prior pair_id=%s role=%s", pair_id, role)

                    try:
                        data = featurizer.get_data_from_smiles(smiles)
                        t0 = time.perf_counter()
                        positions = batched_sampling(
                            model,
                            data,
                            max_batch_size=args.max_batch_size,
                            num_samples=args.num_confs,
                            n_timesteps=args.n_timesteps,
                            seed=None,
                            device=device_str,
                            s_churn=args.s_churn,
                            t_min=args.t_min,
                            t_max=args.t_max,
                            std=args.std,
                            sampler_type=args.sampler_type,
                        )
                        dt = time.perf_counter() - t0
                        LOGGER.debug("ETFlow sample pair_id=%s role=%s confs=%s time=%.3fs",
                                     pair_id, role, positions.shape[0], dt)
                    except Exception as exc:
                        error_writer.writerow([pair_id, role, smiles, f"etflow_error:{exc}"])
                        status.append(f"{role}:etflow_error")
                        role_results[role] = {"status": "etflow_error", "path": ""}
                        stats["errors"] += 1
                        LOGGER.exception("ETFlow failed pair_id=%s role=%s", pair_id, role)
                        continue
                    finally:
                        if prior_before is not None:
                            model.prior_type = prior_before

                    if positions.shape[1] != mol.GetNumAtoms():
                        error_writer.writerow([pair_id, role, smiles, "atom_count_mismatch"])
                        status.append(f"{role}:atom_mismatch")
                        role_results[role] = {"status": "atom_mismatch", "path": ""}
                        stats["atom_mismatch"] += 1
                        continue

                    heavy_only = not args.clash_all_atoms
                    clash_counts = [
                        count_clashes(
                            mol,
                            positions[i],
                            threshold=args.clash_distance,
                            heavy_only=heavy_only,
                        )
                        for i in range(positions.shape[0])
                    ]
                    non_clash_indices = [i for i, c in enumerate(clash_counts) if c == 0]

                    if non_clash_indices:
                        if len(non_clash_indices) < positions.shape[0]:
                            stats["clash_filtered"] += 1
                            error_writer.writerow(
                                [
                                    pair_id,
                                    role,
                                    smiles,
                                    f"clash_filtered:{positions.shape[0]-len(non_clash_indices)}/{positions.shape[0]}",
                                ]
                            )
                            LOGGER.debug(
                                "Clash filtered pair_id=%s role=%s removed=%s of %s",
                                pair_id,
                                role,
                                positions.shape[0] - len(non_clash_indices),
                                positions.shape[0],
                            )
                        positions_for_score = positions[non_clash_indices]
                        score_t0 = time.perf_counter()
                        best_local_idx, best_energy = pick_best_conformer(
                            mol, positions_for_score, args.score
                        )
                        score_dt = time.perf_counter() - score_t0
                        best_idx = non_clash_indices[best_local_idx]
                        LOGGER.debug(
                            "Score pair_id=%s role=%s method=%s best_idx=%s energy=%s time=%.3fs",
                            pair_id,
                            role,
                            args.score,
                            best_idx,
                            best_energy,
                            score_dt,
                        )
                        clash_status = "ok"
                    else:
                        stats["clash_all"] += 1
                        error_writer.writerow([pair_id, role, smiles, "clash_all"])
                        LOGGER.warning(
                            "All conformers clashing pair_id=%s role=%s",
                            pair_id,
                            role,
                        )
                        score_t0 = time.perf_counter()
                        best_idx, best_energy = pick_best_conformer(mol, positions, args.score)
                        score_dt = time.perf_counter() - score_t0
                        LOGGER.debug(
                            "Score (clash_all) pair_id=%s role=%s method=%s best_idx=%s energy=%s time=%.3fs",
                            pair_id,
                            role,
                            args.score,
                            best_idx,
                            best_energy,
                            score_dt,
                        )
                        clash_status = "clash_all"

                    role_results[role] = {
                        "status": clash_status,
                        "mol": mol,
                        "positions": positions,
                        "best_idx": best_idx,
                        "best_energy": best_energy,
                        "path": "",
                        "smiles": smiles,
                    }
                    status.append(f"{role}:{clash_status}")
                    if role == "reactant":
                        stats["reactant_ok"] += 1
                    else:
                        stats["product_ok"] += 1

                pair_has_clash_all = any(
                    res.get("status") == "clash_all" for res in role_results.values()
                )
                pair_output_dir = (args.clash_dir / split) if pair_has_clash_all else out_split_dir
                if pair_has_clash_all:
                    pair_output_dir.mkdir(parents=True, exist_ok=True)
                    LOGGER.warning("Pair %s has clash_all; writing both ligands to %s", pair_id, pair_output_dir)

                for role in ["reactant", "product"]:
                    res = role_results.get(role)
                    if not res:
                        continue
                    res_status = res.get("status")
                    if res_status in {"ok", "clash_all"}:
                        out_path = pair_output_dir / f"{pair_id}__{role}.sdf"
                        write_sdf(
                            res["mol"],
                            res["positions"][res["best_idx"]],
                            out_path,
                            name=f"{pair_id}__{role}",
                            pair_id=pair_id,
                            role=role,
                            smiles=res.get("smiles", ""),
                            energy=res.get("best_energy"),
                        )
                        res["path"] = str(out_path)
                    elif res_status == "exists":
                        candidate = pair_output_dir / f"{pair_id}__{role}.sdf"
                        if candidate.exists():
                            res["path"] = str(candidate)
                    else:
                        res["path"] = ""

                    if role == "reactant":
                        reactant_out = res.get("path", "") or reactant_out
                    else:
                        product_out = res.get("path", "") or product_out

                pair_non_clashing = (
                    "reactant" in role_results
                    and "product" in role_results
                    and role_results["reactant"].get("status") in {"ok", "exists"}
                    and role_results["product"].get("status") in {"ok", "exists"}
                )
                if pair_non_clashing:
                    non_clash_count += 1

                manifest_writer.writerow(
                    [
                        pair_id,
                        row_id,
                        rhea_id,
                        uniprot_id,
                        reactant_smiles,
                        product_smiles,
                        reactant_out,
                        product_out,
                        "|".join(status),
                    ]
                )
                if tqdm is not None:
                    iterator.set_postfix(
                        r_ok=stats["reactant_ok"],
                        p_ok=stats["product_ok"],
                        miss=stats["missing_smiles"],
                        dummy=stats["dummy_pairs"],
                        clash_all=stats["clash_all"],
                        good=non_clash_count,
                        err=stats["errors"],
                    )
                    if count % args.log_every == 0:
                        LOGGER.info(
                            "Progress split=%s rows=%s good=%s r_ok=%s p_ok=%s exists=%s/%s dummy_pairs=%s clash_all=%s errors=%s",
                            split,
                            count,
                            non_clash_count,
                            stats["reactant_ok"],
                            stats["product_ok"],
                            stats["reactant_exists"],
                            stats["product_exists"],
                            stats["dummy_pairs"],
                            stats["clash_all"],
                            stats["errors"],
                        )

            LOGGER.info(
                "Completed split=%s rows=%s r_ok=%s p_ok=%s exists=%s/%s parse_fail=%s atom_mismatch=%s missing_smiles=%s dummy_pairs=%s errors=%s multi_frag=%s gaussian_fallback=%s clash_filtered=%s clash_all=%s",
                split,
                stats["rows"],
                stats["reactant_ok"],
                stats["product_ok"],
                stats["reactant_exists"],
                stats["product_exists"],
                stats["parse_failed"],
                stats["atom_mismatch"],
                stats["missing_smiles"],
                stats["dummy_pairs"],
                stats["errors"],
                stats["multi_fragment"],
                stats["gaussian_fallback"],
                stats["clash_filtered"],
                stats["clash_all"],
            )
        split_dt = time.perf_counter() - split_t0
        LOGGER.info("Split=%s runtime=%.2fs", split, split_dt)


if __name__ == "__main__":
    main()
