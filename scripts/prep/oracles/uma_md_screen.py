#!/usr/bin/env python3
"""UMA scoring wrapper with atom-budget routing and packed batched prediction.

This scorer batches multiple candidates per UMA predict call using an atom-count
budget. Packing count is based on atom count after hydrogenation + first-shell
waters when available (or estimated on demand).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
from typing import Any

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


_PREDICTOR_CACHE: dict[tuple[str, int, str], object] = {}
_CALCULATOR_CACHE: dict[tuple[str, int, str], object] = {}
_ASE_ATOMS_CACHE: dict[str, Any] = {}
_ASE_DATA_CACHE: dict[str, Any] = {}
_PREPARED_ATOM_COUNT_CACHE: dict[tuple[str, float, float], int] = {}
_SPHERE_POINTS_CACHE: dict[int, np.ndarray] = {}


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")



def _read_mmcif_atoms(path: Path):
    from ase import Atoms

    headers: list[str] = []
    in_atom_loop = False
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "loop_":
                headers = []
                in_atom_loop = False
                continue
            if line.startswith("_"):
                headers.append(line)
                if all(h.startswith("_atom_site.") for h in headers):
                    in_atom_loop = True
                continue
            if in_atom_loop and headers:
                if line.startswith("#") or line.startswith("loop_") or line.startswith("data_") or line.startswith("_"):
                    break
                fields = shlex.split(line)
                if len(fields) < len(headers):
                    continue
                h_to_i = {h: i for i, h in enumerate(headers)}
                i_type = h_to_i.get("_atom_site.type_symbol")
                i_x = h_to_i.get("_atom_site.Cartn_x")
                i_y = h_to_i.get("_atom_site.Cartn_y")
                i_z = h_to_i.get("_atom_site.Cartn_z")
                if i_type is None or i_x is None or i_y is None or i_z is None:
                    continue
                try:
                    sym = fields[i_type]
                    x = float(fields[i_x])
                    y = float(fields[i_y])
                    z = float(fields[i_z])
                except Exception:
                    continue
                symbols.append(sym)
                positions.append((x, y, z))
    if not symbols:
        raise ValueError(f"no atom records parsed from mmCIF: {path}")
    return Atoms(symbols=symbols, positions=positions)



def _load_ase_atoms(path: Path):
    key = str(path.resolve())
    if key in _ASE_ATOMS_CACHE:
        return _ASE_ATOMS_CACHE[key].copy()

    from ase.io import read

    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"} or (suffix == ".gz" and path.name.lower().endswith(".cif.gz")):
        atoms = _read_mmcif_atoms(path)
    elif suffix == ".pdb":
        atoms = read(str(path), format="proteindatabank")
    else:
        atoms = read(str(path))

    if "charge" not in atoms.info:
        atoms.info["charge"] = 0
    if "spin" not in atoms.info:
        atoms.info["spin"] = 0
    _ASE_ATOMS_CACHE[key] = atoms.copy()
    return atoms



def _get_predictor_and_calc(model_name: str, workers: int, device: str):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    cache_key = (model_name, workers, device)
    predictor = _PREDICTOR_CACHE.get(cache_key)
    calc = _CALCULATOR_CACHE.get(cache_key)
    if predictor is None or calc is None:
        kwargs = {"device": device}
        if workers > 1:
            kwargs["workers"] = workers
        predictor = pretrained_mlip.get_predict_unit(model_name, **kwargs)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        _PREDICTOR_CACHE[cache_key] = predictor
        _CALCULATOR_CACHE[cache_key] = calc
    return predictor, calc



def _estimate_prepared_atom_count_openmm(
    structure_path: Path,
    *,
    shell_ang: float,
    ph: float,
) -> int:
    """Estimate atom count after hydrogenation + first-shell waters.

    This uses explicit hydrogenation with OpenMM and a geometric first-shell
    estimate from solvent-accessible surface area (SASA) rather than full
    explicit-solvent box construction.
    """

    cache_key = (str(structure_path.resolve()), float(shell_ang), float(ph))
    if cache_key in _PREPARED_ATOM_COUNT_CACHE:
        return int(_PREPARED_ATOM_COUNT_CACHE[cache_key])

    from openmm import unit
    from openmm.app import Modeller, PDBFile, PDBxFile
    from scipy.spatial import cKDTree

    path = structure_path
    suffix = path.suffix.lower()

    if suffix == ".gz":
        # OpenMM readers do not handle gz directly.
        with _open_text(path) as fh, tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=True) as tmp:
            tmp.write(fh.read())
            tmp.flush()
            pdbx = PDBxFile(tmp.name)
            topology = pdbx.topology
            positions = pdbx.positions
    elif suffix in {".cif", ".mmcif"}:
        pdbx = PDBxFile(str(path))
        topology = pdbx.topology
        positions = pdbx.positions
    elif suffix == ".pdb":
        pdb = PDBFile(str(path))
        topology = pdb.topology
        positions = pdb.positions
    else:
        # Route unknown to temporary PDB via ASE.
        atoms = _load_ase_atoms(path)
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=True) as tmp:
            from ase.io import write

            write(tmp.name, atoms, format="proteindatabank")
            tmp.flush()
            pdb = PDBFile(tmp.name)
            topology = pdb.topology
            positions = pdb.positions

    modeller = Modeller(topology, positions)
    modeller.addHydrogens(pH=float(ph))

    top = modeller.topology
    pos_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom), dtype=np.float32)
    if pos_ang.ndim != 2 or pos_ang.shape[1] != 3:
        raise RuntimeError("unexpected OpenMM positions shape while estimating prepared atom count")

    atoms = list(top.atoms())
    symbols = []
    for a in atoms:
        elem = getattr(a, "element", None)
        symbols.append((getattr(elem, "symbol", "") or "").upper())

    # Solvent accessible surface estimate with probe-expanded radii.
    # Radii (Angstrom), conservative defaults for uncommon elements.
    vdwr = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "F": 1.47,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        "MG": 1.73,
        "ZN": 1.39,
        "FE": 1.56,
        "CA": 1.94,
        "NA": 2.27,
        "K": 2.75,
    }
    probe = 1.4
    radii = np.asarray([vdwr.get(sym, 1.70) for sym in symbols], dtype=np.float32)
    expanded = radii + probe

    n_points = 48
    dirs = _SPHERE_POINTS_CACHE.get(n_points)
    if dirs is None:
        # Fibonacci sphere.
        idx = np.arange(n_points, dtype=np.float32) + 0.5
        phi = np.arccos(1.0 - 2.0 * idx / float(n_points))
        theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
        dirs = np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=1,
        ).astype(np.float32)
        _SPHERE_POINTS_CACHE[n_points] = dirs

    tree = cKDTree(pos_ang)
    max_expanded = float(np.max(expanded))
    sasa = 0.0
    for i in range(pos_ang.shape[0]):
        ri = float(expanded[i])
        center = pos_ang[i]
        pts = center[None, :] + ri * dirs
        nbr_idx = tree.query_ball_point(center, r=ri + max_expanded + 1e-6)
        nbr_idx = [j for j in nbr_idx if j != i]
        if not nbr_idx:
            frac = 1.0
        else:
            nbr_pos = pos_ang[nbr_idx]
            nbr_r = expanded[nbr_idx]
            d = np.linalg.norm(pts[:, None, :] - nbr_pos[None, :, :], axis=2)
            covered = (d < (nbr_r[None, :] - 1e-6)).any(axis=1)
            frac = float((~covered).mean())
        sasa += frac * (4.0 * math.pi * ri * ri)

    # First-shell waters from shell volume around accessible surface.
    # rho_water ~= 0.0334 molecules / A^3, 3 atoms per water.
    rho_water = 0.0334
    shell_thickness = max(0.0, float(shell_ang))
    n_waters = sasa * shell_thickness * rho_water
    water_atoms = int(round(3.0 * n_waters))

    n_h = int(pos_ang.shape[0])
    n_total = max(n_h, n_h + water_atoms)
    _PREPARED_ATOM_COUNT_CACHE[cache_key] = n_total
    return n_total



def _resolve_packing_atom_count(
    rec: dict,
    *,
    estimate_prepared_atoms: bool,
    shell_ang: float,
    ph: float,
    strict_estimation: bool,
) -> tuple[int, str]:
    # Explicit field if present (preferred).
    for key in (
        "prepared_atom_count_hydrogenated_shell",
        "prepared_atom_count_hydrogenated_first_shell",
        "atom_count_hydrogenated_shell",
        "atom_count_hydrogenated_first_shell",
    ):
        val = rec.get(key)
        if isinstance(val, int) and val > 0:
            return int(val), key

    structure_path = Path(str(rec.get("cif_path", ""))).resolve()
    if estimate_prepared_atoms:
        try:
            n = _estimate_prepared_atom_count_openmm(
                structure_path,
                shell_ang=float(shell_ang),
                ph=float(ph),
            )
            return int(n), "estimated_openmm_hydrogenated_first_shell"
        except Exception:
            if strict_estimation:
                raise

    # Last resort uses provided prepared_atom_count.
    base = int(rec.get("prepared_atom_count", 0))
    return max(1, base), "prepared_atom_count"



def _pack_candidates_by_atoms(
    active: list[dict],
    *,
    atom_budget: int,
    max_candidates_per_step: int,
) -> tuple[list[dict], int]:
    """Greedy atom-budget packing.

    Input `active` should be sorted descending by packing atom count for good fill.
    """
    budget = max(1, int(atom_budget))
    selected: list[dict] = []
    used = 0

    for item in active:
        if len(selected) >= max_candidates_per_step:
            break
        n = int(item["_uma_pack_atoms"])
        if used + n <= budget:
            selected.append(item)
            used += n

    if not selected:
        # Ensure progress with at least one candidate.
        item = active[0]
        selected = [item]
        used = int(item["_uma_pack_atoms"])

    return selected, int(used)



def _update_atom_budget(
    current: int,
    control_vram_bytes: int | None,
    total_vram_bytes: int | None,
    *,
    target_frac: float,
    max_growth_factor: float,
    max_shrink_factor: float,
    min_value: int,
    max_value: int,
) -> int:
    if control_vram_bytes is None or total_vram_bytes is None or total_vram_bytes <= 0:
        return current
    frac = float(control_vram_bytes) / float(total_vram_bytes)
    if frac <= 0.0:
        return current
    desired = float(current) * (float(target_frac) / frac)
    desired = min(desired, float(current) * max(1.0, float(max_growth_factor)))
    desired = max(desired, float(current) * max(0.0, float(max_shrink_factor)))
    return int(max(min_value, min(max_value, int(round(desired)))))



def _estimate_atom_budget_from_alpha(
    *,
    alpha_bytes_per_atom: float | None,
    total_vram_bytes: int | None,
    target_frac: float,
    min_value: int,
    max_value: int,
) -> int | None:
    if alpha_bytes_per_atom is None or alpha_bytes_per_atom <= 0.0:
        return None
    if total_vram_bytes is None or total_vram_bytes <= 0:
        return None
    budget_bytes = float(total_vram_bytes) * float(target_frac)
    atoms = int(round(budget_bytes / float(alpha_bytes_per_atom)))
    return int(max(min_value, min(max_value, atoms)))



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", default="uma-s-1p1")
    parser.add_argument("--temps", default="300,330,360,390,420")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--atom-budget", type=int, default=10000)
    parser.add_argument("--atom-budget-min", type=int, default=1000)
    parser.add_argument("--atom-budget-max", type=int, default=60000)
    parser.add_argument("--max-candidates-per-step", type=int, default=8)
    parser.add_argument("--target-vram-frac", type=float, default=0.90)
    parser.add_argument("--vram-control-metric", choices=["allocated", "reserved"], default="reserved")
    parser.add_argument("--atom-budget-max-growth-factor", type=float, default=2.0)
    parser.add_argument("--atom-budget-max-shrink-factor", type=float, default=0.5)
    parser.add_argument("--auto-atom-budget-from-vram", dest="auto_atom_budget_from_vram", action="store_true")
    parser.add_argument("--no-auto-atom-budget-from-vram", dest="auto_atom_budget_from_vram", action="store_false")
    parser.add_argument("--estimate-prepared-atoms", dest="estimate_prepared_atoms", action="store_true")
    parser.add_argument("--no-estimate-prepared-atoms", dest="estimate_prepared_atoms", action="store_false")
    parser.add_argument("--strict-prepared-atom-estimation", action="store_true")
    parser.add_argument("--hydration-shell-ang", type=float, default=3.5)
    parser.add_argument("--hydration-ph", type=float, default=7.0)
    parser.add_argument("--require-torch-cuda-vram", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(auto_atom_budget_from_vram=True, estimate_prepared_atoms=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging
    import torch
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    logger = configure_logging("oracle.uma", level=args.log_level)
    rows = read_records(root / args.candidate_path)
    logger.info(
        "UMA scoring start: candidates=%d model=%s workers=%d atom_budget=%d auto_budget=%s target_vram=%.2f max_cands_step=%d estimate_prepared_atoms=%s shell_ang=%.2f",
        len(rows),
        args.model_name,
        args.workers,
        args.atom_budget,
        args.auto_atom_budget_from_vram,
        args.target_vram_frac,
        args.max_candidates_per_step,
        args.estimate_prepared_atoms,
        args.hydration_shell_ang,
    )

    cuda_ok = bool(torch.cuda.is_available())
    if args.require_torch_cuda_vram and not cuda_ok:
        raise RuntimeError("Torch CUDA is unavailable; strict VRAM telemetry required")

    device = "cuda" if cuda_ok else "cpu"
    predictor, calc = _get_predictor_and_calc(args.model_name, int(args.workers), device)

    out = [dict(r) for r in rows]
    eligible: list[dict] = []
    for i, rec in enumerate(out):
        allow = rec.get("eligibility", {}).get("uma_whole", False) or rec.get("eligibility", {}).get("uma_local", False)
        if not allow:
            rec["uma_features"] = None
            rec["uma_calibrated"] = None
            rec["uma_std"] = None
            rec["uma_status"] = "ineligible"
            continue

        path_val = rec.get("cif_path")
        if not path_val:
            raise RuntimeError(f"candidate missing cif_path: candidate_id={rec.get('candidate_id')}")

        structure_path = Path(str(path_val)).resolve()
        pack_atoms, source = _resolve_packing_atom_count(
            rec,
            estimate_prepared_atoms=bool(args.estimate_prepared_atoms),
            shell_ang=float(args.hydration_shell_ang),
            ph=float(args.hydration_ph),
            strict_estimation=bool(args.strict_prepared_atom_estimation),
        )

        rec["uma_atom_count_for_packing"] = int(pack_atoms)
        rec["uma_atom_count_source"] = source
        rec["_row_idx"] = i
        rec["_structure_path"] = str(structure_path)
        rec["_uma_pack_atoms"] = int(pack_atoms)
        eligible.append(rec)

    if not eligible:
        write_records(root / args.output_path, out)
        logger.info(
            "UMA scoring complete: wrote=%d eligible=0 skipped=%d elapsed=%.2fs",
            len(out),
            len(out),
            time.perf_counter() - t0,
        )
        print(root / args.output_path)
        return 0

    # Pack larger systems first for conservative VRAM control.
    eligible.sort(key=lambda r: int(r.get("_uma_pack_atoms", 0)), reverse=True)

    next_atom_budget = int(max(args.atom_budget_min, min(args.atom_budget_max, args.atom_budget)))
    alpha_bytes_per_atom: float | None = None

    completed = 0
    step_id = 0
    pbar = None
    if tqdm is not None and not args.no_progress:
        pbar = tqdm(total=len(eligible), desc="uma:score", dynamic_ncols=True, leave=False)

    while completed < len(eligible):
        active = [r for r in eligible if r.get("uma_status") != "ok"]
        if not active:
            break

        step_id += 1
        selected, used_atoms = _pack_candidates_by_atoms(
            active,
            atom_budget=int(next_atom_budget),
            max_candidates_per_step=max(1, int(args.max_candidates_per_step)),
        )

        # Build AtomicData list.
        data_list = []
        atoms_list = []
        for rec in selected:
            spath = str(rec["_structure_path"])
            if spath in _ASE_DATA_CACHE:
                d = copy.deepcopy(_ASE_DATA_CACHE[spath])
                atoms = _ASE_ATOMS_CACHE[spath].copy()
            else:
                atoms = _load_ase_atoms(Path(spath))
                d = calc.a2g(atoms)
                _ASE_ATOMS_CACHE[spath] = atoms.copy()
                _ASE_DATA_CACHE[spath] = copy.deepcopy(d)
            data_list.append(d)
            atoms_list.append(atoms)

        batch_data = atomicdata_list_to_batch(data_list)

        telemetry: dict[str, Any] = {"vram_source": None}
        total_vram_bytes: int | None = None
        if cuda_ok:
            dev_idx = torch.cuda.current_device()
            total_vram_bytes = int(torch.cuda.get_device_properties(dev_idx).total_memory)
            torch.cuda.synchronize(dev_idx)
            torch.cuda.empty_cache()
            telemetry = {
                "vram_source": "torch.cuda",
                "cuda_device_index": int(dev_idx),
                "cuda_device_name": str(torch.cuda.get_device_name(dev_idx)),
                "start_allocated_bytes": int(torch.cuda.memory_allocated(dev_idx)),
                "start_reserved_bytes": int(torch.cuda.memory_reserved(dev_idx)),
            }
            torch.cuda.reset_peak_memory_stats(dev_idx)

        pred = predictor.predict(batch_data)
        energies = pred["energy"].detach().cpu().numpy().reshape(-1)

        if cuda_ok:
            dev_idx = torch.cuda.current_device()
            torch.cuda.synchronize(dev_idx)
            telemetry.update(
                {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev_idx)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev_idx)),
                    "end_allocated_bytes": int(torch.cuda.memory_allocated(dev_idx)),
                    "end_reserved_bytes": int(torch.cuda.memory_reserved(dev_idx)),
                }
            )

        peak_alloc = telemetry.get("peak_allocated_bytes")
        peak_reserved = telemetry.get("peak_reserved_bytes")
        control_vram = peak_reserved if args.vram_control_metric == "reserved" else peak_alloc
        if control_vram is not None and used_atoms > 0:
            local_alpha = float(control_vram) / float(used_atoms)
            alpha_bytes_per_atom = local_alpha if alpha_bytes_per_atom is None else max(alpha_bytes_per_atom, local_alpha)

        for j, rec in enumerate(selected):
            i = int(rec["_row_idx"])
            nat = max(int(len(atoms_list[j])), 1)
            energy = float(energies[j])
            dg = -energy / float(nat)
            pack_atoms = int(rec.get("_uma_pack_atoms", nat))
            std = 0.18 if pack_atoms <= 8000 else 0.30

            out[i]["uma_features"] = {
                "model_name": args.model_name,
                "energy_total": energy,
                "energy_per_atom": energy / float(nat),
                "dg_300": dg,
                "t_half": 330.0 + max(-20.0, min(40.0, dg * 10.0)),
                "natoms_eval": nat,
                "natoms_packing": pack_atoms,
                "structure_path": str(rec["_structure_path"]),
            }
            out[i]["uma_calibrated"] = dg
            out[i]["uma_std"] = std
            out[i]["uma_status"] = "ok"
            out[i]["uma_step_id"] = int(step_id)
            out[i]["uma_candidates_in_step"] = int(len(selected))
            out[i]["uma_atom_budget_used"] = int(used_atoms)
            out[i]["uma_atom_budget_target"] = int(next_atom_budget)
            out[i]["uma_peak_vram_bytes"] = int(peak_alloc) if peak_alloc is not None else None
            out[i]["uma_peak_reserved_vram_bytes"] = int(peak_reserved) if peak_reserved is not None else None
            out[i]["uma_total_vram_bytes"] = int(total_vram_bytes) if total_vram_bytes is not None else None
            out[i]["uma_vram_source"] = telemetry.get("vram_source")
            out[i]["uma_vram_control_metric"] = args.vram_control_metric
            out[i]["uma_control_vram_bytes"] = int(control_vram) if control_vram is not None else None
            out[i]["uma_control_vram_frac"] = (
                float(control_vram) / float(total_vram_bytes)
                if control_vram is not None and total_vram_bytes is not None and total_vram_bytes > 0
                else None
            )

            completed += 1
            if pbar is not None:
                pbar.update(1)

        if args.auto_atom_budget_from_vram:
            next_atom_budget = _update_atom_budget(
                int(next_atom_budget),
                int(control_vram) if control_vram is not None else None,
                int(total_vram_bytes) if total_vram_bytes is not None else None,
                target_frac=float(args.target_vram_frac),
                max_growth_factor=float(args.atom_budget_max_growth_factor),
                max_shrink_factor=float(args.atom_budget_max_shrink_factor),
                min_value=int(args.atom_budget_min),
                max_value=int(args.atom_budget_max),
            )
            est = _estimate_atom_budget_from_alpha(
                alpha_bytes_per_atom=alpha_bytes_per_atom,
                total_vram_bytes=int(total_vram_bytes) if total_vram_bytes is not None else None,
                target_frac=float(args.target_vram_frac),
                min_value=int(args.atom_budget_min),
                max_value=int(args.atom_budget_max),
            )
            if est is not None:
                next_atom_budget = int(est)

        if step_id == 1 or (step_id % 8) == 0:
            logger.info(
                "UMA packed step=%d completed=%d/%d candidates_in_step=%d atom_budget_used=%d atom_budget_target=%d control_frac=%s peak_alloc_gib=%s peak_reserved_gib=%s alpha_bytes_per_atom=%s next_atom_budget=%d",
                step_id,
                completed,
                len(eligible),
                len(selected),
                used_atoms,
                next_atom_budget,
                (
                    f"{(float(control_vram) / float(total_vram_bytes)):.3f}"
                    if control_vram is not None and total_vram_bytes is not None and total_vram_bytes > 0
                    else "n/a"
                ),
                (f"{float(peak_alloc) / float(1024**3):.3f}" if peak_alloc is not None else "n/a"),
                (f"{float(peak_reserved) / float(1024**3):.3f}" if peak_reserved is not None else "n/a"),
                (f"{alpha_bytes_per_atom:.2f}" if alpha_bytes_per_atom is not None else "n/a"),
                next_atom_budget,
            )

    if pbar is not None:
        pbar.close()

    for rec in out:
        rec.pop("_row_idx", None)
        rec.pop("_structure_path", None)
        rec.pop("_uma_pack_atoms", None)

    eligible_count = sum(1 for r in out if r.get("uma_status") == "ok")
    write_records(root / args.output_path, out)
    logger.info(
        "UMA scoring complete: wrote=%d eligible=%d skipped=%d elapsed=%.2fs",
        len(out),
        eligible_count,
        len(out) - eligible_count,
        time.perf_counter() - t0,
    )
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
