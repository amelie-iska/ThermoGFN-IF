"""Self-contained UMA catalytic runtime for whole-enzyme broad/sMD screening."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
import os
from pathlib import Path
import shlex
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .uma_cat_reward import KB_KCAL_MOL_K, gating_free_energy_kcal_mol, log10_rate_proxy


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value in {"", ".", "?"}:
        return None
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value in {"", ".", "?", "nan"}:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip()
    if not symbol:
        return "C"
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


@dataclass
class StructureData:
    path: str
    positions: np.ndarray
    symbols: list[str]
    atom_names: list[str]
    residue_names: list[str]
    chain_ids: list[str]
    residue_ids: list[int | None]
    group_pdb: list[str]

    def copy_positions(self) -> np.ndarray:
        return np.array(self.positions, dtype=np.float64, copy=True)

    def count_atoms(self) -> int:
        return int(self.positions.shape[0])

    def to_ase_atoms(self, *, charge: int = 0, spin: int = 0):
        from ase import Atoms

        atoms = Atoms(symbols=self.symbols, positions=self.positions.copy())
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(spin)
        return atoms


_STRUCTURE_CACHE: dict[str, StructureData] = {}
_CALCULATOR_CACHE: dict[tuple[str, str, int], Any] = {}


def load_structure(path: str | Path) -> StructureData:
    p = Path(path).resolve()
    key = str(p)
    if key in _STRUCTURE_CACHE:
        cached = _STRUCTURE_CACHE[key]
        return StructureData(
            path=cached.path,
            positions=cached.positions.copy(),
            symbols=list(cached.symbols),
            atom_names=list(cached.atom_names),
            residue_names=list(cached.residue_names),
            chain_ids=list(cached.chain_ids),
            residue_ids=list(cached.residue_ids),
            group_pdb=list(cached.group_pdb),
        )
    suffix = p.suffix.lower()
    if suffix in {".cif", ".mmcif"} or (suffix == ".gz" and p.name.lower().endswith(".cif.gz")):
        structure = _read_mmcif_structure(p)
    elif suffix == ".pdb":
        structure = _read_pdb_structure(p)
    else:
        raise ValueError(f"unsupported structure extension: {p}")
    _STRUCTURE_CACHE[key] = structure
    return load_structure(p)


def count_atoms_in_structure(path: str | Path) -> int:
    return int(load_structure(path).count_atoms())


def _read_mmcif_structure(path: Path) -> StructureData:
    headers: list[str] = []
    in_atom_loop = False
    rows: list[list[str]] = []

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "loop_":
                headers = []
                in_atom_loop = False
                continue
            if stripped.startswith("_"):
                headers.append(stripped)
                if headers and all(h.startswith("_atom_site.") for h in headers):
                    in_atom_loop = True
                continue
            if in_atom_loop and headers:
                if stripped.startswith("#") or stripped.startswith("loop_") or stripped.startswith("data_") or stripped.startswith("_"):
                    break
                fields = shlex.split(line, posix=True)
                if len(fields) < len(headers):
                    continue
                rows.append(fields[: len(headers)])

    if not rows:
        raise ValueError(f"no atom_site rows parsed from {path}")

    h_to_i = {h: i for i, h in enumerate(headers)}

    def get(row: list[str], *candidates: str) -> str | None:
        for name in candidates:
            idx = h_to_i.get(name)
            if idx is not None and idx < len(row):
                return row[idx]
        return None

    positions: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    chain_ids: list[str] = []
    residue_ids: list[int | None] = []
    group_pdb: list[str] = []

    for row in rows:
        x = _parse_float(get(row, "_atom_site.Cartn_x"))
        y = _parse_float(get(row, "_atom_site.Cartn_y"))
        z = _parse_float(get(row, "_atom_site.Cartn_z"))
        if x is None or y is None or z is None:
            continue
        positions.append((x, y, z))
        symbols.append(_normalize_symbol(get(row, "_atom_site.type_symbol") or "C"))
        atom_names.append((get(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id") or "").strip())
        residue_names.append((get(row, "_atom_site.auth_comp_id", "_atom_site.label_comp_id") or "").strip())
        chain_ids.append((get(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id") or "").strip())
        residue_ids.append(_parse_int(get(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")))
        group_pdb.append((get(row, "_atom_site.group_PDB") or "").strip().upper())

    if not positions:
        raise ValueError(f"no atomic coordinates parsed from {path}")

    return StructureData(
        path=str(path),
        positions=np.asarray(positions, dtype=np.float64),
        symbols=symbols,
        atom_names=atom_names,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_ids=residue_ids,
        group_pdb=group_pdb,
    )


def _read_pdb_structure(path: Path) -> StructureData:
    positions: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    chain_ids: list[str] = []
    residue_ids: list[int | None] = []
    group_pdb: list[str] = []
    with _open_text(path) as fh:
        for raw in fh:
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            group_pdb.append(raw[0:6].strip().upper())
            atom_names.append(raw[12:16].strip())
            residue_names.append(raw[17:20].strip())
            chain_ids.append(raw[21:22].strip())
            residue_ids.append(_parse_int(raw[22:26]))
            x = _parse_float(raw[30:38])
            y = _parse_float(raw[38:46])
            z = _parse_float(raw[46:54])
            if x is None or y is None or z is None:
                continue
            positions.append((x, y, z))
            symbol = raw[76:78].strip() or raw[12:13].strip() or "C"
            symbols.append(_normalize_symbol(symbol))
    if not positions:
        raise ValueError(f"no atom records parsed from {path}")
    return StructureData(
        path=str(path),
        positions=np.asarray(positions, dtype=np.float64),
        symbols=symbols,
        atom_names=atom_names,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_ids=residue_ids,
        group_pdb=group_pdb,
    )


def _is_heavy(symbol: str) -> bool:
    return symbol.upper() != "H"


def _is_water(res_name: str) -> bool:
    return res_name.upper() in {"HOH", "WAT", "H2O"}


def protein_heavy_indices(
    structure: StructureData,
    *,
    chain_id: str | None = None,
    exclude_positions: set[int] | None = None,
    stride: int = 1,
) -> np.ndarray:
    out: list[int] = []
    skip = exclude_positions or set()
    step = max(1, int(stride))
    for idx, (grp, sym, ch, resid) in enumerate(
        zip(structure.group_pdb, structure.symbols, structure.chain_ids, structure.residue_ids, strict=False)
    ):
        if grp != "ATOM":
            continue
        if chain_id and ch != chain_id:
            continue
        if resid is not None and resid in skip:
            continue
        if not _is_heavy(sym):
            continue
        out.append(idx)
    if step > 1 and len(out) > step:
        out = out[::step]
    return np.asarray(out, dtype=np.int64)


def pocket_heavy_indices(
    structure: StructureData,
    *,
    chain_id: str,
    pocket_positions: list[int],
) -> np.ndarray:
    pocket = set(int(x) for x in pocket_positions)
    out = [
        idx
        for idx, (grp, sym, ch, resid) in enumerate(
            zip(structure.group_pdb, structure.symbols, structure.chain_ids, structure.residue_ids, strict=False)
        )
        if grp == "ATOM" and ch == chain_id and resid in pocket and _is_heavy(sym)
    ]
    return np.asarray(out, dtype=np.int64)


def ligand_heavy_indices(structure: StructureData, *, chain_id: str | None = None) -> np.ndarray:
    out: list[int] = []
    for idx, (grp, sym, ch, res_name) in enumerate(
        zip(structure.group_pdb, structure.symbols, structure.chain_ids, structure.residue_names, strict=False)
    ):
        if grp != "HETATM":
            continue
        if chain_id and ch != chain_id:
            continue
        if _is_water(res_name) or not _is_heavy(sym):
            continue
        out.append(idx)
    if not out and chain_id:
        # Some RF3 outputs may write the ligand as ATOM records under a dedicated chain.
        for idx, (sym, ch, res_name) in enumerate(
            zip(structure.symbols, structure.chain_ids, structure.residue_names, strict=False)
        ):
            if ch == chain_id and not _is_water(res_name) and _is_heavy(sym):
                out.append(idx)
    return np.asarray(out, dtype=np.int64)


def match_identity_indices(
    left: StructureData,
    right: StructureData,
    *,
    chain_id: str,
    residue_positions: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    pocket = set(int(x) for x in residue_positions)
    left_map: dict[tuple[str, int | None, str], int] = {}
    right_map: dict[tuple[str, int | None, str], int] = {}
    for idx, (grp, sym, ch, resid, atom_name) in enumerate(
        zip(left.group_pdb, left.symbols, left.chain_ids, left.residue_ids, left.atom_names, strict=False)
    ):
        if grp == "ATOM" and ch == chain_id and resid in pocket and _is_heavy(sym):
            left_map[(ch, resid, atom_name)] = idx
    for idx, (grp, sym, ch, resid, atom_name) in enumerate(
        zip(right.group_pdb, right.symbols, right.chain_ids, right.residue_ids, right.atom_names, strict=False)
    ):
        if grp == "ATOM" and ch == chain_id and resid in pocket and _is_heavy(sym):
            right_map[(ch, resid, atom_name)] = idx
    shared = [key for key in left_map if key in right_map]
    shared.sort(key=lambda k: (k[1] if k[1] is not None else -1, k[2]))
    if not shared:
        raise ValueError("no shared pocket atom identities between endpoint structures")
    return (
        np.asarray([left_map[k] for k in shared], dtype=np.int64),
        np.asarray([right_map[k] for k in shared], dtype=np.int64),
    )


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("kabsch_align expects arrays of shape [N,3] with matching N")
    ref_cent = reference.mean(axis=0)
    mob_cent = mobile.mean(axis=0)
    ref0 = reference - ref_cent
    mob0 = mobile - mob_cent
    cov = mob0.T @ ref0
    u, _s, vt = np.linalg.svd(cov)
    d = np.linalg.det(vt.T @ u.T)
    corr = np.eye(3)
    corr[2, 2] = -1.0 if d < 0 else 1.0
    rot = vt.T @ corr @ u.T
    trans = ref_cent - mob_cent @ rot.T
    return rot, trans


def apply_transform(positions: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return positions @ rotation.T + translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def map_ligand_atoms_between_endpoints(
    reactant: StructureData,
    product: StructureData,
    *,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
) -> dict[str, Any]:
    react_pocket_idx, prod_pocket_idx = match_identity_indices(
        reactant,
        product,
        chain_id=protein_chain_id,
        residue_positions=pocket_positions,
    )
    rot, trans = kabsch_align(
        reactant.positions[react_pocket_idx],
        product.positions[prod_pocket_idx],
    )
    product_aligned = apply_transform(product.positions, rot, trans)

    react_lig_idx = ligand_heavy_indices(reactant, chain_id=ligand_chain_id)
    prod_lig_idx = ligand_heavy_indices(product, chain_id=ligand_chain_id)
    if len(react_lig_idx) == 0 or len(prod_lig_idx) == 0:
        raise ValueError("ligand atoms missing in reactant or product structure")

    react_sym = np.asarray([reactant.symbols[i] for i in react_lig_idx], dtype=object)
    prod_sym = np.asarray([product.symbols[i] for i in prod_lig_idx], dtype=object)

    pair_left: list[int] = []
    pair_right: list[int] = []
    per_element: dict[str, int] = {}
    for element in sorted(set(react_sym.tolist()) & set(prod_sym.tolist())):
        left_local = np.where(react_sym == element)[0]
        right_local = np.where(prod_sym == element)[0]
        if left_local.size == 0 or right_local.size == 0:
            continue
        cost = pairwise_distances(
            reactant.positions[react_lig_idx[left_local]],
            product_aligned[prod_lig_idx[right_local]],
        )
        rows, cols = linear_sum_assignment(cost)
        n_take = min(len(rows), len(cols), left_local.size, right_local.size)
        rows = rows[:n_take]
        cols = cols[:n_take]
        pair_left.extend(react_lig_idx[left_local[rows]].tolist())
        pair_right.extend(prod_lig_idx[right_local[cols]].tolist())
        per_element[element] = int(n_take)

    if not pair_left:
        raise ValueError("no ligand atom correspondence found between reactant and product")

    return {
        "reactant_indices": np.asarray(pair_left, dtype=np.int64),
        "product_indices": np.asarray(pair_right, dtype=np.int64),
        "product_aligned_positions": product_aligned,
        "shared_atom_count": int(len(pair_left)),
        "element_counts": per_element,
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_productive_metrics(
    positions: np.ndarray,
    *,
    ref_positions: np.ndarray,
    anchor_indices: np.ndarray,
    pocket_indices: np.ndarray,
    ligand_indices: np.ndarray,
    contact_cutoff_a: float,
    ligand_rmsd_max_a: float,
    pocket_rmsd_max_a: float,
    min_contacts: int,
    min_distance_safe_a: float,
    soft_weights: dict[str, float] | None = None,
    soft_scales: dict[str, float] | None = None,
) -> dict[str, float]:
    align_indices = anchor_indices if len(anchor_indices) > 0 else pocket_indices
    rot, trans = kabsch_align(ref_positions[align_indices], positions[align_indices])
    aligned = apply_transform(positions, rot, trans)

    pocket_ref = ref_positions[pocket_indices]
    pocket_cur = aligned[pocket_indices]
    lig_ref = ref_positions[ligand_indices]
    lig_cur = aligned[ligand_indices]

    pocket_r = rmsd(pocket_cur, pocket_ref)
    ligand_r = rmsd(lig_cur, lig_ref)
    dist = pairwise_distances(pocket_cur, lig_cur)
    min_dist = float(np.min(dist)) if dist.size else float("inf")
    contacts = int(np.sum(dist <= float(contact_cutoff_a)))

    is_productive = int(
        ligand_r <= float(ligand_rmsd_max_a)
        and pocket_r <= float(pocket_rmsd_max_a)
        and contacts >= int(min_contacts)
        and min_dist >= float(min_distance_safe_a)
    )

    w = {"ligand_rmsd": 0.40, "contacts": 0.25, "distance": 0.10, "pocket_rmsd": 0.25}
    if soft_weights:
        w.update({k: float(v) for k, v in soft_weights.items()})
    s = {"ligand_rmsd": 1.0, "contacts": 4.0, "distance": 0.5, "pocket_rmsd": 1.0}
    if soft_scales:
        s.update({k: float(v) for k, v in soft_scales.items()})

    soft_score = (
        w["ligand_rmsd"] * math.exp(-(ligand_r**2) / (2.0 * max(1e-6, s["ligand_rmsd"]) ** 2))
        + w["contacts"] * _sigmoid((contacts - float(min_contacts)) / max(1e-6, s["contacts"]))
        + w["distance"] * math.exp(-(max(0.0, float(min_distance_safe_a) - min_dist) ** 2) / (2.0 * max(1e-6, s["distance"]) ** 2))
        + w["pocket_rmsd"] * math.exp(-(pocket_r**2) / (2.0 * max(1e-6, s["pocket_rmsd"]) ** 2))
    )

    return {
        "ligand_rmsd_aligned_a": float(ligand_r),
        "pocket_rmsd_a": float(pocket_r),
        "contact_count": float(contacts),
        "min_pocket_ligand_distance_a": float(min_dist),
        "is_productive": float(is_productive),
        "soft_score": float(soft_score),
    }


def estimate_effective_sample_size(series: list[float]) -> float:
    n = len(series)
    if n <= 1:
        return float(n)
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 1e-12:
        return float(n)
    rho_sum = 0.0
    max_lag = min(n - 1, 200)
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(x[:-lag], x[lag:]) / ((n - lag) * var))
        if not np.isfinite(rho) or rho <= 0.0:
            break
        rho_sum += rho
    tau = max(1.0, 1.0 + 2.0 * rho_sum)
    return float(n / tau)


def _count_visits(binary_series: list[int]) -> tuple[int, float, float]:
    visits = 0
    dwell_lengths: list[int] = []
    cur = 0
    first_hit = None
    prev = 0
    for idx, value in enumerate(binary_series):
        if value and not prev:
            visits += 1
            if first_hit is None:
                first_hit = idx
            cur = 1
        elif value:
            cur += 1
        elif prev:
            dwell_lengths.append(cur)
            cur = 0
        prev = value
    if cur:
        dwell_lengths.append(cur)
    mean_dwell = float(np.mean(dwell_lengths)) if dwell_lengths else 0.0
    first = float(first_hit if first_hit is not None else -1)
    return visits, mean_dwell, first


def _get_uma_calculator(model_name: str, device: str, workers: int = 1):
    key = (str(model_name), str(device), int(workers))
    if key in _CALCULATOR_CACHE:
        return _CALCULATOR_CACHE[key]
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    raw_device = str(device)
    fairchem_device = raw_device
    if raw_device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = raw_device.split(":", 1)[1]
        fairchem_device = "cuda"
    kwargs = {"device": fairchem_device}
    if int(workers) > 1:
        kwargs["workers"] = int(workers)
    predictor = pretrained_mlip.get_predict_unit(str(model_name), **kwargs)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    _CALCULATOR_CACHE[key] = calc
    return calc


def run_broad_uma_screen(
    *,
    reactant_complex_path: str | Path,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_per_fs: float,
    steps: int,
    replicas: int,
    save_every: int,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    contact_cutoff_a: float = 4.5,
    ligand_rmsd_max_a: float = 2.5,
    pocket_rmsd_max_a: float = 2.0,
    min_contacts: int = 4,
    min_distance_safe_a: float = 1.5,
    anchor_stride: int = 8,
    soft_weights: dict[str, float] | None = None,
    soft_scales: dict[str, float] | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    structure = load_structure(reactant_complex_path)
    pocket_idx = pocket_heavy_indices(structure, chain_id=protein_chain_id, pocket_positions=pocket_positions)
    ligand_idx = ligand_heavy_indices(structure, chain_id=ligand_chain_id)
    anchor_idx = protein_heavy_indices(
        structure,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=anchor_stride,
    )
    if len(pocket_idx) == 0:
        raise ValueError("no pocket atoms selected for broad UMA screen")
    if len(ligand_idx) == 0:
        raise ValueError("no ligand atoms selected for broad UMA screen")
    if len(anchor_idx) == 0:
        anchor_idx = pocket_idx

    ref_positions = structure.copy_positions()
    calc = _get_uma_calculator(model_name, device, workers=calculator_workers)
    broad_rows: list[dict[str, Any]] = []
    replicate_summaries: list[dict[str, Any]] = []
    productive_all: list[int] = []
    soft_all: list[float] = []

    for replica_idx in range(int(replicas)):
        atoms = structure.to_ase_atoms()
        atoms.calc = calc
        rng = np.random.RandomState(int(seed) + replica_idx)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature_k), rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = Langevin(
            atoms,
            float(timestep_fs) * units.fs,
            temperature_K=float(temperature_k),
            friction=float(friction_per_fs),
        )

        rep_rows: list[dict[str, Any]] = []

        def observe(step_idx: int) -> None:
            metrics = compute_productive_metrics(
                atoms.positions.copy(),
                ref_positions=ref_positions,
                anchor_indices=anchor_idx,
                pocket_indices=pocket_idx,
                ligand_indices=ligand_idx,
                contact_cutoff_a=contact_cutoff_a,
                ligand_rmsd_max_a=ligand_rmsd_max_a,
                pocket_rmsd_max_a=pocket_rmsd_max_a,
                min_contacts=min_contacts,
                min_distance_safe_a=min_distance_safe_a,
                soft_weights=soft_weights,
                soft_scales=soft_scales,
            )
            metrics["replica"] = int(replica_idx)
            metrics["step"] = int(step_idx)
            rep_rows.append(metrics)

        observe(0)
        for step_idx in range(1, int(steps) + 1):
            dyn.run(1)
            if step_idx % int(max(1, save_every)) == 0 or step_idx == int(steps):
                observe(step_idx)

        binary = [int(row["is_productive"]) for row in rep_rows]
        visits, dwell, first_hit = _count_visits(binary)
        p_rep = float(np.mean(binary)) if binary else 0.0
        replicate_summaries.append(
            {
                "replica": int(replica_idx),
                "p_gnac": float(p_rep),
                "p_soft": float(np.mean([row["soft_score"] for row in rep_rows])) if rep_rows else 0.0,
                "n_visits": int(visits),
                "mean_dwell_frames": float(dwell),
                "first_hit_frame": float(first_hit),
            }
        )
        broad_rows.extend(rep_rows)
        productive_all.extend(binary)
        soft_all.extend([float(row["soft_score"]) for row in rep_rows])

    p_gnac = float(np.mean(productive_all)) if productive_all else 0.0
    delta_g_gate = gating_free_energy_kcal_mol(p_gnac, temperature_k=float(temperature_k))
    p_soft = float(np.mean(soft_all)) if soft_all else 0.0
    n_eff = estimate_effective_sample_size(productive_all)
    p_std = math.sqrt(max(1e-12, p_gnac * (1.0 - p_gnac)) / max(1.0, n_eff))
    p_lcb = max(0.0, p_gnac - 1.96 * p_std)
    dg_gate_std = _gating_delta_g_std_kcal_mol(p_gnac, p_std, float(temperature_k))
    instability_penalty = float(sum(1.0 for row in broad_rows if row["min_pocket_ligand_distance_a"] < 1.2))
    visits, dwell, first_hit = _count_visits(productive_all)

    return {
        "status": "ok",
        "replicate_summaries": replicate_summaries,
        "broad_rows": broad_rows,
        "p_gnac": float(p_gnac),
        "p_soft": float(p_soft),
        "delta_g_gate_kcal_mol": float(delta_g_gate),
        "productive_visit_count": int(visits),
        "productive_dwell_frames": float(dwell),
        "first_hit_frame": float(first_hit),
        "replicate_var_p_gnac": float(np.var([row["p_gnac"] for row in replicate_summaries])) if replicate_summaries else 0.0,
        "n_eff": float(n_eff),
        "p_gnac_std": float(p_std),
        "p_gnac_lcb": float(p_lcb),
        "delta_g_gate_std_kcal_mol": float(dg_gate_std),
        "instability_penalty": float(instability_penalty),
    }


@dataclass
class SteeringProtocol:
    steered_indices: np.ndarray
    start_targets: np.ndarray
    end_targets: np.ndarray
    anchor_indices: np.ndarray
    anchor_targets: np.ndarray
    k_steer_eva2: float
    k_anchor_eva2: float
    total_steps: int
    force_clip_eva: float | None = None
    step_idx: int = 0
    cumulative_work: float = 0.0
    work_profile: list[dict[str, float]] = field(default_factory=list)

    def _lambda_for_step(self, step: int) -> float:
        if self.total_steps <= 1:
            return 1.0
        return float(step) / float(self.total_steps - 1)

    def current_targets(self, step: int | None = None) -> np.ndarray:
        use_step = self.step_idx if step is None else int(step)
        lam = self._lambda_for_step(use_step)
        return (1.0 - lam) * self.start_targets + lam * self.end_targets

    def bias_energy_and_forces(self, positions: np.ndarray, step: int | None = None) -> tuple[float, np.ndarray]:
        use_step = self.step_idx if step is None else int(step)
        targets = self.current_targets(use_step)
        forces = np.zeros_like(positions, dtype=np.float64)
        energy = 0.0

        disp = positions[self.steered_indices] - targets
        energy += 0.5 * float(self.k_steer_eva2) * float(np.sum(disp * disp))
        steer_force = -float(self.k_steer_eva2) * disp
        if self.force_clip_eva is not None and self.force_clip_eva > 0:
            norms = np.linalg.norm(steer_force, axis=1, keepdims=True)
            mask = norms > float(self.force_clip_eva)
            steer_force[mask[:, 0]] *= float(self.force_clip_eva) / np.maximum(norms[mask[:, 0]], 1e-12)
        forces[self.steered_indices] += steer_force

        if len(self.anchor_indices) > 0 and self.k_anchor_eva2 > 0:
            adisp = positions[self.anchor_indices] - self.anchor_targets
            energy += 0.5 * float(self.k_anchor_eva2) * float(np.sum(adisp * adisp))
            forces[self.anchor_indices] += -float(self.k_anchor_eva2) * adisp

        return float(energy), forces

    def advance_protocol(self, positions: np.ndarray) -> float:
        current = self.bias_energy_and_forces(positions, self.step_idx)[0]
        next_step = min(self.step_idx + 1, self.total_steps - 1)
        nxt = self.bias_energy_and_forces(positions, next_step)[0]
        delta = float(nxt - current)
        self.cumulative_work += delta
        self.step_idx = next_step
        self.work_profile.append(
            {
                "step": float(self.step_idx),
                "lambda": float(self._lambda_for_step(self.step_idx)),
                "work_increment_kcal_mol": float(delta),
                "cumulative_work_kcal_mol": float(self.cumulative_work),
            }
        )
        return float(delta)


class SteeredUMACalculator:
    implemented_properties = ("energy", "forces")

    def __init__(self, base_calc: Any, protocol: SteeringProtocol):
        self.base_calc = base_calc
        self.protocol = protocol
        self.results: dict[str, Any] = {}
        self.last_physical_energy = 0.0
        self.last_bias_energy = 0.0

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.base_calc.calculate(atoms, properties=["energy", "forces"], system_changes=system_changes)
        phys_e = float(self.base_calc.results["energy"])
        phys_f = np.asarray(self.base_calc.results["forces"], dtype=np.float64)
        bias_e, bias_f = self.protocol.bias_energy_and_forces(np.asarray(atoms.positions, dtype=np.float64))
        self.last_physical_energy = float(phys_e)
        self.last_bias_energy = float(bias_e)
        self.results = {
            "energy": float(phys_e + bias_e),
            "forces": phys_f + bias_f,
            "free_energy": float(phys_e + bias_e),
        }

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if not self.results:
            self.calculate(atoms, properties=["energy", "forces"], system_changes=None)
        return self.results["energy"]

    def get_forces(self, atoms=None):
        if not self.results:
            self.calculate(atoms, properties=["energy", "forces"], system_changes=None)
        return self.results["forces"]


class FixedTargetBiasCalculator:
    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        base_calc: Any,
        *,
        steered_indices: np.ndarray,
        steer_targets: np.ndarray,
        anchor_indices: np.ndarray,
        anchor_targets: np.ndarray,
        k_window_eva2: float,
        k_anchor_eva2: float,
    ):
        self.base_calc = base_calc
        self.steered_indices = np.asarray(steered_indices, dtype=np.int64)
        self.steer_targets = np.asarray(steer_targets, dtype=np.float64)
        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
        self.anchor_targets = np.asarray(anchor_targets, dtype=np.float64)
        self.k_window_eva2 = float(k_window_eva2)
        self.k_anchor_eva2 = float(k_anchor_eva2)
        self.results: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.base_calc.calculate(atoms, properties=["energy", "forces"], system_changes=system_changes)
        phys_e = float(self.base_calc.results["energy"])
        phys_f = np.asarray(self.base_calc.results["forces"], dtype=np.float64)
        positions = np.asarray(atoms.positions, dtype=np.float64)

        forces = np.zeros_like(positions, dtype=np.float64)
        energy = 0.0

        sdisp = positions[self.steered_indices] - self.steer_targets
        energy += 0.5 * self.k_window_eva2 * float(np.sum(sdisp * sdisp))
        forces[self.steered_indices] += -self.k_window_eva2 * sdisp

        if len(self.anchor_indices) > 0 and self.k_anchor_eva2 > 0:
            adisp = positions[self.anchor_indices] - self.anchor_targets
            energy += 0.5 * self.k_anchor_eva2 * float(np.sum(adisp * adisp))
            forces[self.anchor_indices] += -self.k_anchor_eva2 * adisp

        self.results = {
            "energy": float(phys_e + energy),
            "forces": phys_f + forces,
            "free_energy": float(phys_e + energy),
        }

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if not self.results:
            self.calculate(atoms, properties=["energy", "forces"], system_changes=None)
        return self.results["energy"]

    def get_forces(self, atoms=None):
        if not self.results:
            self.calculate(atoms, properties=["energy", "forces"], system_changes=None)
        return self.results["forces"]


def _project_progress_lambda(
    positions: np.ndarray,
    *,
    steered_indices: np.ndarray,
    start_targets: np.ndarray,
    end_targets: np.ndarray,
) -> float:
    current = np.asarray(positions[steered_indices], dtype=np.float64).reshape(-1)
    start = np.asarray(start_targets, dtype=np.float64).reshape(-1)
    end = np.asarray(end_targets, dtype=np.float64).reshape(-1)
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        return 0.0
    lam = float(np.dot(current - start, delta) / denom)
    return float(min(1.0, max(0.0, lam)))


def _interpolate_path_targets(path_lambdas: np.ndarray, path_targets: np.ndarray, lam: float) -> np.ndarray:
    lam = float(min(1.0, max(0.0, lam)))
    if lam <= float(path_lambdas[0]):
        return np.asarray(path_targets[0], dtype=np.float64)
    if lam >= float(path_lambdas[-1]):
        return np.asarray(path_targets[-1], dtype=np.float64)
    hi = int(np.searchsorted(path_lambdas, lam, side="right"))
    lo = max(0, hi - 1)
    lam_lo = float(path_lambdas[lo])
    lam_hi = float(path_lambdas[hi])
    if lam_hi <= lam_lo + 1e-12:
        return np.asarray(path_targets[lo], dtype=np.float64)
    frac = (lam - lam_lo) / (lam_hi - lam_lo)
    return (1.0 - frac) * np.asarray(path_targets[lo], dtype=np.float64) + frac * np.asarray(path_targets[hi], dtype=np.float64)


def _window_bias_energy(
    positions: np.ndarray,
    *,
    steered_indices: np.ndarray,
    target_positions: np.ndarray,
    k_window_eva2: float,
) -> float:
    disp = np.asarray(positions[steered_indices], dtype=np.float64) - np.asarray(target_positions, dtype=np.float64)
    return 0.5 * float(k_window_eva2) * float(np.sum(disp * disp))


def _solve_mbar_free_energies(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    k_states, n_samples = u_kn.shape
    f_k = np.zeros(k_states, dtype=np.float64)
    n_k = np.asarray(n_k, dtype=np.float64)
    for _ in range(max_iter):
        log_terms = f_k[:, None] - u_kn + np.log(n_k[:, None] + 1e-300)
        max_log = np.max(log_terms, axis=0)
        denom = np.exp(log_terms - max_log[None, :]).sum(axis=0)
        log_denom = max_log + np.log(denom + 1e-300)
        new_f = np.empty_like(f_k)
        for state_idx in range(k_states):
            vals = -u_kn[state_idx] - log_denom
            m = float(np.max(vals))
            new_f[state_idx] = -(m + math.log(float(np.exp(vals - m).sum()) + 1e-300))
        new_f -= new_f[0]
        if float(np.max(np.abs(new_f - f_k))) < tol:
            return new_f
        f_k = new_f
    return f_k


def _jarzynski_profile(work_traces: list[np.ndarray], temperature_k: float) -> tuple[np.ndarray, np.ndarray]:
    beta = 1.0 / (0.00198720425864083 * float(temperature_k))
    stacked = np.stack(work_traces, axis=0)
    shifted = stacked - stacked[:, :1]
    free = -np.log(np.mean(np.exp(-beta * shifted), axis=0)) / beta
    lambdas = np.linspace(0.0, 1.0, stacked.shape[1], dtype=np.float64)
    return lambdas, free


def _gating_delta_g_std_kcal_mol(p_gnac: float, p_std: float, temperature_k: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p_gnac)))
    rt = KB_KCAL_MOL_K * float(temperature_k)
    deriv = rt / max(1e-12, p * (1.0 - p))
    return float(abs(deriv) * max(0.0, float(p_std)))


def _bootstrap_jarzynski_uncertainty(
    work_traces: list[np.ndarray],
    *,
    temperature_k: float,
    seed: int,
    n_boot: int = 128,
) -> tuple[float, float]:
    if len(work_traces) <= 1:
        return 0.0, 0.0
    rng = np.random.RandomState(int(seed))
    stacked = np.stack(work_traces, axis=0)
    n_traces = stacked.shape[0]
    barriers: list[float] = []
    delta_gs: list[float] = []
    for _ in range(max(8, int(n_boot))):
        idx = rng.randint(0, n_traces, size=n_traces)
        _lambdas, free = _jarzynski_profile([stacked[i] for i in idx], temperature_k=float(temperature_k))
        barriers.append(float(np.max(free) - free[0]))
        delta_gs.append(float(free[-1] - free[0]))
    return float(np.std(barriers, ddof=1)), float(np.std(delta_gs, ddof=1))


def _reconstruct_path_pmf(
    *,
    samples_lambda: list[float],
    bias_rows: list[np.ndarray],
    window_counts: np.ndarray,
    total_states: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not bias_rows:
        raise RuntimeError("no PMF samples collected")

    u_kn = np.stack(bias_rows, axis=1)
    f_k = _solve_mbar_free_energies(u_kn, window_counts)
    log_terms = f_k[:, None] - u_kn + np.log(window_counts[:, None] + 1e-300)
    max_log = np.max(log_terms, axis=0)
    denom = np.exp(log_terms - max_log[None, :]).sum(axis=0)
    weights = 1.0 / np.exp(max_log + np.log(denom + 1e-300))
    weights = weights / np.sum(weights)

    lambda_arr = np.asarray(samples_lambda, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, total_states + 1, dtype=np.float64)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(total_states, dtype=np.float64)
    for lam, weight in zip(lambda_arr, weights, strict=False):
        idx = min(total_states - 1, max(0, int(np.searchsorted(bins, lam, side="right") - 1)))
        hist[idx] += float(weight)
    hist = np.clip(hist, 1e-300, None)
    pmf = -(1.0 / beta) * np.log(hist)
    reactant_mask = bin_centers <= 0.15
    reactant_ref = float(np.min(pmf[reactant_mask])) if np.any(reactant_mask) else float(pmf[0])
    pmf -= reactant_ref
    return bin_centers, pmf


def run_steered_uma_dynamics(
    *,
    reactant_complex_path: str | Path,
    product_complex_path: str | Path,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_per_fs: float,
    images: int,
    steps_per_image: int,
    replicas: int,
    k_steer_eva2: float,
    k_anchor_eva2: float,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    anchor_stride: int = 12,
    force_clip_eva: float | None = 25.0,
    seed: int = 13,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    reactant = load_structure(reactant_complex_path)
    product = load_structure(product_complex_path)
    mapping = map_ligand_atoms_between_endpoints(
        reactant,
        product,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pocket_positions=pocket_positions,
    )

    anchor_idx = protein_heavy_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=anchor_stride,
    )
    if len(anchor_idx) == 0:
        anchor_idx = pocket_heavy_indices(reactant, chain_id=protein_chain_id, pocket_positions=pocket_positions)
    if len(anchor_idx) == 0:
        raise ValueError("no anchor atoms found for steered UMA dynamics")

    total_steps = max(2, int(images) * int(steps_per_image))
    calc = _get_uma_calculator(model_name, device, workers=calculator_workers)

    work_traces: list[np.ndarray] = []
    replica_summaries: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    path_snapshots: list[list[np.ndarray]] = []
    near_ts_rows: list[dict[str, Any]] = []

    for replica_idx in range(int(replicas)):
        atoms = reactant.to_ase_atoms()
        protocol = SteeringProtocol(
            steered_indices=np.asarray(mapping["reactant_indices"], dtype=np.int64),
            start_targets=reactant.positions[np.asarray(mapping["reactant_indices"], dtype=np.int64)].copy(),
            end_targets=mapping["product_aligned_positions"][np.asarray(mapping["product_indices"], dtype=np.int64)].copy(),
            anchor_indices=anchor_idx,
            anchor_targets=reactant.positions[anchor_idx].copy(),
            k_steer_eva2=float(k_steer_eva2),
            k_anchor_eva2=float(k_anchor_eva2),
            total_steps=int(total_steps),
            force_clip_eva=force_clip_eva,
        )
        steer_calc = SteeredUMACalculator(calc, protocol)
        atoms.calc = steer_calc

        rng = np.random.RandomState(int(seed) + 10_000 + replica_idx)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature_k), rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)

        dyn = Langevin(
            atoms,
            float(timestep_fs) * units.fs,
            temperature_K=float(temperature_k),
            friction=float(friction_per_fs),
        )

        protocol.work_profile.append(
            {
                "step": 0.0,
                "lambda": 0.0,
                "work_increment_kcal_mol": 0.0,
                "cumulative_work_kcal_mol": 0.0,
            }
        )
        rep_snapshots = [atoms.positions[np.asarray(mapping["reactant_indices"], dtype=np.int64)].copy()]
        for _ in range(int(total_steps) - 1):
            dyn.run(1)
            protocol.advance_protocol(np.asarray(atoms.positions, dtype=np.float64))
            if protocol.step_idx % max(1, int(steps_per_image)) == 0 or protocol.step_idx == int(total_steps) - 1:
                rep_snapshots.append(atoms.positions[np.asarray(mapping["reactant_indices"], dtype=np.int64)].copy())

        work_trace = np.asarray([row["cumulative_work_kcal_mol"] for row in protocol.work_profile], dtype=np.float64)
        work_traces.append(work_trace)
        path_snapshots.append(rep_snapshots)

        react_lig = atoms.positions[np.asarray(mapping["reactant_indices"], dtype=np.int64)].copy()
        prod_lig = mapping["product_aligned_positions"][np.asarray(mapping["product_indices"], dtype=np.int64)].copy()
        final_rmsd = rmsd(react_lig, prod_lig)
        summary = {
            "replica": int(replica_idx),
            "final_cumulative_work_kcal_mol": float(work_trace[-1]),
            "final_product_rmsd_a": float(final_rmsd),
            "shared_atom_count": int(mapping["shared_atom_count"]),
        }
        replica_summaries.append(summary)
        endpoint_rows.extend(
            {
                "replica": int(replica_idx),
                **row,
            }
            for row in protocol.work_profile
        )

        step_grid = list(range(0, int(total_steps), max(1, int(steps_per_image))))
        if not step_grid or step_grid[-1] != int(total_steps) - 1:
            step_grid.append(int(total_steps) - 1)
        last_cum = 0.0
        for snap_idx, step_idx in enumerate(step_grid[: len(rep_snapshots)]):
            lam = protocol._lambda_for_step(step_idx)
            cum = float(work_trace[min(step_idx, len(work_trace) - 1)])
            local_work = float(cum - last_cum)
            last_cum = cum
            snap_positions = rep_snapshots[snap_idx]
            start_rmsd = rmsd(snap_positions, protocol.start_targets)
            end_rmsd = rmsd(snap_positions, protocol.end_targets)
            symmetry = max(0.0, 1.0 - abs(2.0 * float(lam) - 1.0))
            score = float((1.0 + max(0.0, local_work)) * symmetry)
            near_ts_rows.append(
                {
                    "replica": int(replica_idx),
                    "image_index": int(snap_idx),
                    "step": int(step_idx),
                    "lambda": float(lam),
                    "local_work_kcal_mol": float(local_work),
                    "cumulative_work_kcal_mol": float(cum),
                    "start_ligand_rmsd_a": float(start_rmsd),
                    "end_ligand_rmsd_a": float(end_rmsd),
                    "near_ts_score": float(score),
                }
            )

    lambdas, free_profile = _jarzynski_profile(work_traces, temperature_k=float(temperature_k))
    barrier = float(np.max(free_profile) - free_profile[0])
    delta_g = float(free_profile[-1] - free_profile[0])
    mean_work = float(np.mean([trace[-1] for trace in work_traces]))
    std_work = float(np.std([trace[-1] for trace in work_traces])) if len(work_traces) > 1 else 0.0
    barrier_std, delta_g_std = _bootstrap_jarzynski_uncertainty(
        work_traces,
        temperature_k=float(temperature_k),
        seed=int(seed) + 50_000,
    )
    path_targets = np.mean(np.asarray(path_snapshots, dtype=np.float64), axis=0)
    path_lambdas = np.linspace(0.0, 1.0, path_targets.shape[0], dtype=np.float64)
    near_ts_rows = sorted(near_ts_rows, key=lambda row: row["near_ts_score"], reverse=True)

    return {
        "status": "ok",
        "mapping": {
            "shared_atom_count": int(mapping["shared_atom_count"]),
            "element_counts": mapping["element_counts"],
        },
        "replica_summaries": replica_summaries,
        "endpoint_rows": endpoint_rows,
        "near_ts_candidates": near_ts_rows[: min(16, len(near_ts_rows))],
        "path_lambdas": path_lambdas.tolist(),
        "path_targets": path_targets.tolist(),
        "lambdas": lambdas.tolist(),
        "jarzynski_free_profile_kcal_mol": free_profile.tolist(),
        "delta_g_smd_barrier_kcal_mol": float(barrier),
        "delta_g_smd_barrier_std_kcal_mol": float(barrier_std),
        "delta_g_react_to_prod_kcal_mol": float(delta_g),
        "delta_g_react_to_prod_std_kcal_mol": float(delta_g_std),
        "mean_final_work_kcal_mol": float(mean_work),
        "std_final_work_kcal_mol": float(std_work),
    }


def run_path_umbrella_pmf(
    *,
    reactant_complex_path: str | Path,
    product_complex_path: str | Path,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_per_fs: float,
    windows: int,
    steps_per_window: int,
    save_every: int,
    replicas: int,
    k_window_eva2: float,
    k_anchor_eva2: float,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    anchor_stride: int = 12,
    path_lambdas: list[float] | np.ndarray | None = None,
    path_targets: list[Any] | np.ndarray | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    reactant = load_structure(reactant_complex_path)
    product = load_structure(product_complex_path)
    mapping = map_ligand_atoms_between_endpoints(
        reactant,
        product,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pocket_positions=pocket_positions,
    )
    anchor_idx = protein_heavy_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=anchor_stride,
    )
    if len(anchor_idx) == 0:
        anchor_idx = pocket_heavy_indices(reactant, chain_id=protein_chain_id, pocket_positions=pocket_positions)
    if len(anchor_idx) == 0:
        raise ValueError("no anchor atoms found for umbrella PMF")

    calc = _get_uma_calculator(model_name, device, workers=calculator_workers)
    steered_indices = np.asarray(mapping["reactant_indices"], dtype=np.int64)
    start_targets = reactant.positions[steered_indices].copy()
    end_targets = mapping["product_aligned_positions"][np.asarray(mapping["product_indices"], dtype=np.int64)].copy()

    if path_lambdas is None or path_targets is None:
        path_lambdas_arr = np.linspace(0.0, 1.0, int(max(2, windows)), dtype=np.float64)
        path_targets_arr = np.stack(
            [(1.0 - lam) * start_targets + lam * end_targets for lam in path_lambdas_arr],
            axis=0,
        )
    else:
        path_lambdas_arr = np.asarray(path_lambdas, dtype=np.float64)
        path_targets_arr = np.asarray(path_targets, dtype=np.float64)
        if path_targets_arr.ndim != 3:
            raise ValueError("path_targets must have shape [n_images, n_atoms, 3]")

    window_lambdas = np.linspace(0.0, 1.0, int(max(2, windows)), dtype=np.float64)
    total_states = len(window_lambdas)
    beta = 1.0 / (0.00198720425864083 * float(temperature_k))

    samples_lambda: list[float] = []
    bias_rows: list[np.ndarray] = []
    window_counts = np.zeros(total_states, dtype=np.int64)
    per_replica_samples: dict[int, dict[str, Any]] = {
        replica_idx: {
            "samples_lambda": [],
            "bias_rows": [],
            "window_counts": np.zeros(total_states, dtype=np.int64),
        }
        for replica_idx in range(int(replicas))
    }
    window_summaries: list[dict[str, Any]] = []

    for state_idx, lam_center in enumerate(window_lambdas):
        target_positions = _interpolate_path_targets(path_lambdas_arr, path_targets_arr, float(lam_center))
        for replica_idx in range(int(replicas)):
            atoms = reactant.to_ase_atoms()
            atoms.positions[steered_indices] = target_positions.copy()
            bias_calc = FixedTargetBiasCalculator(
                calc,
                steered_indices=steered_indices,
                steer_targets=target_positions,
                anchor_indices=anchor_idx,
                anchor_targets=reactant.positions[anchor_idx].copy(),
                k_window_eva2=float(k_window_eva2),
                k_anchor_eva2=float(k_anchor_eva2),
            )
            atoms.calc = bias_calc

            rng = np.random.RandomState(int(seed) + (state_idx * 1000) + replica_idx)
            MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature_k), rng=rng)
            Stationary(atoms)
            ZeroRotation(atoms)
            dyn = Langevin(
                atoms,
                float(timestep_fs) * units.fs,
                temperature_K=float(temperature_k),
                friction=float(friction_per_fs),
            )

            rep_lambdas: list[float] = []
            for step_idx in range(0, int(steps_per_window) + 1):
                if step_idx > 0:
                    dyn.run(1)
                if step_idx % max(1, int(save_every)) != 0 and step_idx != int(steps_per_window):
                    continue
                positions = np.asarray(atoms.positions, dtype=np.float64)
                lam_obs = _project_progress_lambda(
                    positions,
                    steered_indices=steered_indices,
                    start_targets=start_targets,
                    end_targets=end_targets,
                )
                rep_lambdas.append(float(lam_obs))
                reduced_bias = []
                for other_center in window_lambdas:
                    other_targets = _interpolate_path_targets(path_lambdas_arr, path_targets_arr, float(other_center))
                    reduced_bias.append(
                        beta
                        * _window_bias_energy(
                            positions,
                            steered_indices=steered_indices,
                            target_positions=other_targets,
                            k_window_eva2=float(k_window_eva2),
                        )
                    )
                samples_lambda.append(float(lam_obs))
                reduced_bias_arr = np.asarray(reduced_bias, dtype=np.float64)
                bias_rows.append(reduced_bias_arr)
                window_counts[state_idx] += 1
                rep_bundle = per_replica_samples[int(replica_idx)]
                rep_bundle["samples_lambda"].append(float(lam_obs))
                rep_bundle["bias_rows"].append(reduced_bias_arr)
                rep_bundle["window_counts"][state_idx] += 1
            window_summaries.append(
                {
                    "window_index": int(state_idx),
                    "replica": int(replica_idx),
                    "lambda_center": float(lam_center),
                    "mean_sampled_lambda": float(np.mean(rep_lambdas)) if rep_lambdas else float(lam_center),
                    "n_samples": int(len(rep_lambdas)),
                }
            )

    bin_centers, pmf = _reconstruct_path_pmf(
        samples_lambda=samples_lambda,
        bias_rows=bias_rows,
        window_counts=window_counts,
        total_states=total_states,
        beta=beta,
    )
    reactant_mask = bin_centers <= 0.15
    product_mask = bin_centers >= 0.85
    reactant_ref = float(np.min(pmf[reactant_mask])) if np.any(reactant_mask) else float(pmf[0])
    product_ref = float(np.min(pmf[product_mask])) if np.any(product_mask) else float(pmf[-1])
    barrier = float(np.max(pmf))
    delta_g = float(product_ref - reactant_ref)
    replica_barriers: list[float] = []
    replica_delta_gs: list[float] = []
    for rep_bundle in per_replica_samples.values():
        if not rep_bundle["bias_rows"]:
            continue
        rep_centers, rep_pmf = _reconstruct_path_pmf(
            samples_lambda=rep_bundle["samples_lambda"],
            bias_rows=rep_bundle["bias_rows"],
            window_counts=rep_bundle["window_counts"],
            total_states=total_states,
            beta=beta,
        )
        rep_reactant_mask = rep_centers <= 0.15
        rep_product_mask = rep_centers >= 0.85
        rep_reactant_ref = float(np.min(rep_pmf[rep_reactant_mask])) if np.any(rep_reactant_mask) else float(rep_pmf[0])
        rep_product_ref = float(np.min(rep_pmf[rep_product_mask])) if np.any(rep_product_mask) else float(rep_pmf[-1])
        replica_barriers.append(float(np.max(rep_pmf)))
        replica_delta_gs.append(float(rep_product_ref - rep_reactant_ref))

    return {
        "status": "ok",
        "window_lambdas": window_lambdas.tolist(),
        "window_counts": window_counts.tolist(),
        "window_summaries": window_summaries,
        "path_lambdas": path_lambdas_arr.tolist(),
        "pmf_lambda_centers": bin_centers.tolist(),
        "pmf_kcal_mol": pmf.tolist(),
        "delta_g_pmf_barrier_kcal_mol": float(barrier),
        "delta_g_pmf_barrier_std_kcal_mol": float(np.std(replica_barriers, ddof=1)) if len(replica_barriers) > 1 else 0.0,
        "delta_g_pmf_react_to_prod_kcal_mol": float(delta_g),
        "delta_g_pmf_react_to_prod_std_kcal_mol": float(np.std(replica_delta_gs, ddof=1)) if len(replica_delta_gs) > 1 else 0.0,
        "num_samples": int(len(samples_lambda)),
    }


def summarize_catalytic_screen(
    *,
    broad: dict[str, Any],
    smd: dict[str, Any] | None,
    pmf: dict[str, Any] | None,
    temperature_k: float,
) -> dict[str, Any]:
    status = "ok" if broad.get("status") == "ok" and (smd is None or smd.get("status") == "ok") and (pmf is None or pmf.get("status") == "ok") else "error"
    dg_gate = float(broad.get("delta_g_gate_kcal_mol", 0.0))
    dg_bar_smd = float(smd.get("delta_g_smd_barrier_kcal_mol", 0.0)) if smd else 0.0
    dg_bar_pmf = float(pmf.get("delta_g_pmf_barrier_kcal_mol", 0.0)) if pmf else 0.0
    use_pmf = pmf is not None and pmf.get("status") == "ok"
    dg_bar = dg_bar_pmf if use_pmf else dg_bar_smd
    dg_gate_std = float(broad.get("delta_g_gate_std_kcal_mol", 0.0))
    smd_barrier_std = float(smd.get("delta_g_smd_barrier_std_kcal_mol", 0.0)) if smd else 0.0
    pmf_barrier_std = float(pmf.get("delta_g_pmf_barrier_std_kcal_mol", 0.0)) if pmf else 0.0
    log_rate = log10_rate_proxy(dg_gate, dg_bar, temperature_k=float(temperature_k))
    mean_work = float(smd.get("mean_final_work_kcal_mol", 0.0)) if smd else 0.0
    reverse_gap = float(smd.get("forward_reverse_gap_kcal_mol", 0.0)) if smd else 0.0
    barrier_std = pmf_barrier_std if use_pmf else math.sqrt(smd_barrier_std**2 + (0.5 * reverse_gap) ** 2)
    rt_ln10 = KB_KCAL_MOL_K * float(temperature_k) * math.log(10.0)
    uncertainty = float(math.sqrt(dg_gate_std**2 + barrier_std**2) / max(1e-12, rt_ln10))
    near_ts_count = len(smd.get("near_ts_candidates", [])) if smd else 0
    pathway_score = float(max(0.0, near_ts_count) - max(0.0, mean_work) - max(0.0, reverse_gap))
    return {
        "status": status,
        "uma_cat_p_gnac": float(broad.get("p_gnac", 0.0)),
        "uma_cat_p_soft": float(broad.get("p_soft", 0.0)),
        "uma_cat_delta_g_gate_kcal_mol": float(dg_gate),
        "uma_cat_delta_g_gate_std_kcal_mol": float(dg_gate_std),
        "uma_cat_gnac_lcb": float(broad.get("p_gnac_lcb", 0.0)),
        "uma_cat_visits": int(broad.get("productive_visit_count", 0)),
        "uma_cat_dwell_frames": float(broad.get("productive_dwell_frames", 0.0)),
        "uma_cat_first_hit_frame": float(broad.get("first_hit_frame", -1.0)),
        "uma_cat_delta_g_smd_barrier_kcal_mol": float(dg_bar_smd),
        "uma_cat_delta_g_smd_barrier_std_kcal_mol": float(smd_barrier_std),
        "uma_cat_delta_g_react_to_prod_kcal_mol": float(smd.get("delta_g_react_to_prod_kcal_mol", 0.0)) if smd else 0.0,
        "uma_cat_delta_g_react_to_prod_std_kcal_mol": float(smd.get("delta_g_react_to_prod_std_kcal_mol", 0.0)) if smd else 0.0,
        "uma_cat_delta_g_pmf_barrier_kcal_mol": float(dg_bar_pmf),
        "uma_cat_delta_g_pmf_barrier_std_kcal_mol": float(pmf_barrier_std),
        "uma_cat_delta_g_pmf_react_to_prod_kcal_mol": float(pmf.get("delta_g_pmf_react_to_prod_kcal_mol", 0.0)) if pmf else 0.0,
        "uma_cat_delta_g_pmf_react_to_prod_std_kcal_mol": float(pmf.get("delta_g_pmf_react_to_prod_std_kcal_mol", 0.0)) if pmf else 0.0,
        "uma_cat_barrier_source": "pmf" if use_pmf else "smd",
        "uma_cat_delta_g_barrier_std_kcal_mol": float(barrier_std),
        "uma_cat_mean_work_kcal_mol": float(mean_work),
        "uma_cat_work_std_kcal_mol": float(smd.get("std_final_work_kcal_mol", 0.0)) if smd else 0.0,
        "uma_cat_pathway_score": float(pathway_score),
        "uma_cat_near_ts_count": int(near_ts_count),
        "uma_cat_log10_rate_proxy": float(log_rate),
        "uma_cat_log10_rate_std": float(uncertainty),
        "uma_cat_std": float(uncertainty),
        "uma_cat_status": status,
    }


def write_catalytic_artifacts(
    artifact_dir: str | Path,
    *,
    record: dict,
    broad: dict[str, Any],
    smd: dict[str, Any] | None,
    pmf: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, str]:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_path / "summary.json"
    broad_path = artifact_path / "broad_rows.jsonl"
    broad_rep_path = artifact_path / "broad_replicates.json"
    endpoint_path = artifact_path / "smd_work_profile.jsonl"
    smd_path = artifact_path / "smd_summary.json"
    near_ts_path = artifact_path / "smd_near_ts.json"
    pmf_path = artifact_path / "pmf_summary.json"

    summary_payload = {
        "candidate_id": record.get("candidate_id"),
        "record": {
            "reactant_complex_path": record.get("reactant_complex_path"),
            "product_complex_path": record.get("product_complex_path"),
            "reactant_complex_packed_path": record.get("reactant_complex_packed_path"),
            "product_complex_packed_path": record.get("product_complex_packed_path"),
            "pocket_positions": record.get("pocket_positions"),
            "protein_chain_id": record.get("protein_chain_id"),
            "ligand_chain_id": record.get("ligand_chain_id"),
        },
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    with broad_path.open("w", encoding="utf-8") as fh:
        for row in broad.get("broad_rows", []):
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
    broad_rep_path.write_text(json.dumps(broad.get("replicate_summaries", []), indent=2, sort_keys=True))

    if smd is not None:
        with endpoint_path.open("w", encoding="utf-8") as fh:
            for row in smd.get("endpoint_rows", []):
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")
        smd_path.write_text(
            json.dumps(
                {
                    "replicate_summaries": smd.get("replica_summaries", []),
                    "mapping": smd.get("mapping", {}),
                    "lambdas": smd.get("lambdas", []),
                    "jarzynski_free_profile_kcal_mol": smd.get("jarzynski_free_profile_kcal_mol", []),
                    "delta_g_smd_barrier_kcal_mol": smd.get("delta_g_smd_barrier_kcal_mol"),
                    "delta_g_smd_barrier_std_kcal_mol": smd.get("delta_g_smd_barrier_std_kcal_mol"),
                    "delta_g_react_to_prod_kcal_mol": smd.get("delta_g_react_to_prod_kcal_mol"),
                    "delta_g_react_to_prod_std_kcal_mol": smd.get("delta_g_react_to_prod_std_kcal_mol"),
                    "mean_final_work_kcal_mol": smd.get("mean_final_work_kcal_mol"),
                    "std_final_work_kcal_mol": smd.get("std_final_work_kcal_mol"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        near_ts_path.write_text(json.dumps(smd.get("near_ts_candidates", []), indent=2, sort_keys=True))

    if pmf is not None:
        pmf_path.write_text(
            json.dumps(
                {
                    "window_lambdas": pmf.get("window_lambdas", []),
                    "window_counts": pmf.get("window_counts", []),
                    "window_summaries": pmf.get("window_summaries", []),
                    "pmf_lambda_centers": pmf.get("pmf_lambda_centers", []),
                    "pmf_kcal_mol": pmf.get("pmf_kcal_mol", []),
                    "delta_g_pmf_barrier_kcal_mol": pmf.get("delta_g_pmf_barrier_kcal_mol"),
                    "delta_g_pmf_barrier_std_kcal_mol": pmf.get("delta_g_pmf_barrier_std_kcal_mol"),
                    "delta_g_pmf_react_to_prod_kcal_mol": pmf.get("delta_g_pmf_react_to_prod_kcal_mol"),
                    "delta_g_pmf_react_to_prod_std_kcal_mol": pmf.get("delta_g_pmf_react_to_prod_std_kcal_mol"),
                    "num_samples": pmf.get("num_samples"),
                },
                indent=2,
                sort_keys=True,
            )
        )

    return {
        "summary_json": str(summary_path),
        "broad_rows_jsonl": str(broad_path),
        "broad_replicates_json": str(broad_rep_path),
        "smd_work_profile_jsonl": str(endpoint_path) if smd is not None else "",
        "smd_summary_json": str(smd_path) if smd is not None else "",
        "smd_near_ts_json": str(near_ts_path) if smd is not None else "",
        "pmf_summary_json": str(pmf_path) if pmf is not None else "",
    }
