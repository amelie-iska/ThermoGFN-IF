"""Self-contained UMA catalytic runtime for whole-enzyme broad/sMD screening."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation, Slerp
from ase.data import atomic_numbers, covalent_radii

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
_PREPARED_STRUCTURE_CACHE: dict[tuple[str, str, str, tuple[int, ...], float, bool, int, float, float, float], StructureData] = {}
_EQUILIBRATED_STRUCTURE_CACHE: dict[
    tuple[str, str, str, str, int, float, float, float, int, int, float, float, float, float, float],
    StructureData,
] = {}


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


def _copy_structure_with_path(structure: StructureData, *, path: str) -> StructureData:
    return StructureData(
        path=str(path),
        positions=np.asarray(structure.positions, dtype=np.float64, copy=True),
        symbols=list(structure.symbols),
        atom_names=list(structure.atom_names),
        residue_names=list(structure.residue_names),
        chain_ids=list(structure.chain_ids),
        residue_ids=list(structure.residue_ids),
        group_pdb=list(structure.group_pdb),
    )


def _load_openmm_modeller(path: str | Path):
    from openmm.app import Modeller, PDBFile, PDBxFile

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".cif", ".mmcif"} or (suffix == ".gz" and p.name.lower().endswith(".cif.gz")):
        with _open_text(p) as fh, tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=True) as tmp:
            tmp.write(fh.read())
            tmp.flush()
            pdbx = PDBxFile(tmp.name)
            return Modeller(pdbx.topology, pdbx.positions)
    if suffix == ".pdb":
        pdb = PDBFile(str(p))
        return Modeller(pdb.topology, pdb.positions)
    raise ValueError(f"unsupported structure extension for OpenMM preparation: {p}")


def _hydrogenate_structure_openmm(path: str | Path, *, ph: float) -> StructureData:
    from openmm.app import PDBFile

    modeller = _load_openmm_modeller(path)
    modeller.addHydrogens(pH=float(ph))
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=True) as tmp:
        PDBFile.writeFile(modeller.topology, modeller.positions, tmp, keepIds=True)
        tmp.flush()
        prepared = load_structure(tmp.name)
    return _copy_structure_with_path(prepared, path=f"{Path(path).resolve()}::hydrogenated")


def _polar_atom_indices(structure: StructureData, indices: np.ndarray) -> np.ndarray:
    polar = {"N", "O", "S", "P"}
    out = [int(idx) for idx in np.asarray(indices, dtype=np.int64).tolist() if str(structure.symbols[int(idx)]).upper() in polar]
    return np.asarray(out, dtype=np.int64)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / n


def _choose_open_direction(
    structure: StructureData,
    *,
    atom_idx: int,
    reference_positions: np.ndarray | None = None,
    radius_a: float = 3.5,
) -> np.ndarray:
    pos = np.asarray(structure.positions[int(atom_idx)], dtype=np.float64)
    heavy_idx = np.asarray([i for i, sym in enumerate(structure.symbols) if _is_heavy(sym)], dtype=np.int64)
    heavy_idx = heavy_idx[heavy_idx != int(atom_idx)]
    if heavy_idx.size == 0:
        if reference_positions is not None and len(reference_positions) > 0:
            return _normalize(np.mean(np.asarray(reference_positions, dtype=np.float64), axis=0) - pos)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    nbr_pos = structure.positions[heavy_idx]
    d = np.linalg.norm(nbr_pos - pos[None, :], axis=1)
    mask = d <= float(radius_a)
    if np.any(mask):
        vecs = nbr_pos[mask] - pos[None, :]
        inv = 1.0 / np.maximum(np.linalg.norm(vecs, axis=1), 1e-6)
        direction = -np.sum(vecs * inv[:, None], axis=0)
        direction = _normalize(direction)
        if float(np.linalg.norm(direction)) > 0.0:
            return direction
    if reference_positions is not None and len(reference_positions) > 0:
        ref = np.asarray(reference_positions, dtype=np.float64)
        nearest = ref[np.argmin(np.linalg.norm(ref - pos[None, :], axis=1))]
        direction = _normalize(nearest - pos)
        if float(np.linalg.norm(direction)) > 0.0:
            return direction
    centroid = np.mean(np.asarray(structure.positions[heavy_idx], dtype=np.float64), axis=0)
    direction = _normalize(pos - centroid)
    if float(np.linalg.norm(direction)) > 0.0:
        return direction
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def _water_hydrogen_positions(
    oxygen: np.ndarray,
    *,
    bisector_direction: np.ndarray,
    bond_length_a: float = 0.9572,
    angle_deg: float = 104.5,
) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(oxygen, dtype=np.float64)
    bis = _normalize(bisector_direction)
    if float(np.linalg.norm(bis)) <= 1e-12:
        bis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(bis, ref))) > 0.95:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    perp = _normalize(np.cross(bis, ref))
    theta = math.radians(float(angle_deg) / 2.0)
    h1 = o + float(bond_length_a) * (math.cos(theta) * bis + math.sin(theta) * perp)
    h2 = o + float(bond_length_a) * (math.cos(theta) * bis - math.sin(theta) * perp)
    return h1, h2


def _append_water_molecules(
    structure: StructureData,
    *,
    oxygen_positions: list[np.ndarray],
    hydrogen_pairs: list[tuple[np.ndarray, np.ndarray]],
    chain_id: str = "W",
) -> StructureData:
    positions = structure.copy_positions()
    symbols = list(structure.symbols)
    atom_names = list(structure.atom_names)
    residue_names = list(structure.residue_names)
    chain_ids = list(structure.chain_ids)
    residue_ids = list(structure.residue_ids)
    group_pdb = list(structure.group_pdb)

    next_resid = 9000
    if residue_ids:
        numeric = [int(x) for x in residue_ids if x is not None]
        if numeric:
            next_resid = max(max(numeric) + 1, 9000)

    extra_positions: list[np.ndarray] = []
    for water_idx, (o_pos, h_pair) in enumerate(zip(oxygen_positions, hydrogen_pairs, strict=False)):
        resid = next_resid + water_idx
        extra_positions.append(np.asarray(o_pos, dtype=np.float64))
        symbols.append("O")
        atom_names.append("O")
        residue_names.append("HOH")
        chain_ids.append(chain_id)
        residue_ids.append(resid)
        group_pdb.append("HETATM")
        h1, h2 = h_pair
        for atom_name, pos in zip(("H1", "H2"), (h1, h2), strict=False):
            extra_positions.append(np.asarray(pos, dtype=np.float64))
            symbols.append("H")
            atom_names.append(atom_name)
            residue_names.append("HOH")
            chain_ids.append(chain_id)
            residue_ids.append(resid)
            group_pdb.append("HETATM")

    if extra_positions:
        positions = np.vstack([positions, np.asarray(extra_positions, dtype=np.float64)])

    return StructureData(
        path=f"{structure.path}::first_shell_waters",
        positions=np.asarray(positions, dtype=np.float64),
        symbols=symbols,
        atom_names=atom_names,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_ids=residue_ids,
        group_pdb=group_pdb,
    )


def _add_first_shell_waters(
    structure: StructureData,
    *,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    max_waters: int,
    shell_distance_a: float,
    clash_distance_a: float,
    bridge_distance_min_a: float,
    bridge_distance_max_a: float,
) -> StructureData:
    pocket_idx = pocket_heavy_indices(structure, chain_id=protein_chain_id, pocket_positions=pocket_positions)
    ligand_idx = ligand_heavy_indices(structure, chain_id=ligand_chain_id)
    if len(pocket_idx) == 0 or len(ligand_idx) == 0 or int(max_waters) <= 0:
        return structure

    pocket_polar = _polar_atom_indices(structure, pocket_idx)
    ligand_polar = _polar_atom_indices(structure, ligand_idx)
    if len(pocket_polar) == 0 and len(ligand_polar) == 0:
        return structure

    heavy_idx = np.asarray([i for i, sym in enumerate(structure.symbols) if _is_heavy(sym)], dtype=np.int64)
    heavy_pos = np.asarray(structure.positions[heavy_idx], dtype=np.float64)
    ligand_pos = np.asarray(structure.positions[ligand_idx], dtype=np.float64)
    pocket_pos = np.asarray(structure.positions[pocket_idx], dtype=np.float64)

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    for p_idx in pocket_polar.tolist():
        p = np.asarray(structure.positions[int(p_idx)], dtype=np.float64)
        for l_idx in ligand_polar.tolist():
            l = np.asarray(structure.positions[int(l_idx)], dtype=np.float64)
            dist = float(np.linalg.norm(l - p))
            if dist < float(bridge_distance_min_a) or dist > float(bridge_distance_max_a):
                continue
            oxygen = 0.5 * (p + l)
            min_clash = float(np.min(np.linalg.norm(heavy_pos - oxygen[None, :], axis=1)))
            if min_clash < float(clash_distance_a):
                continue
            score = 10.0 - abs(dist - 5.6)
            bisector = _normalize(oxygen - 0.5 * (p + l))
            if float(np.linalg.norm(bisector)) <= 1e-12:
                bisector = _choose_open_direction(structure, atom_idx=int(p_idx), reference_positions=ligand_pos)
            candidates.append((score, oxygen, bisector))

    interface_idx = np.concatenate([pocket_polar, ligand_polar]).astype(np.int64, copy=False)
    ref_lookup: dict[int, np.ndarray] = {}
    for idx in interface_idx.tolist():
        if int(idx) in set(pocket_idx.tolist()):
            ref_lookup[int(idx)] = ligand_pos
        else:
            ref_lookup[int(idx)] = pocket_pos

    for atom_idx in interface_idx.tolist():
        anchor_pos = np.asarray(structure.positions[int(atom_idx)], dtype=np.float64)
        direction = _choose_open_direction(
            structure,
            atom_idx=int(atom_idx),
            reference_positions=ref_lookup.get(int(atom_idx)),
        )
        oxygen = anchor_pos + float(shell_distance_a) * direction
        min_clash = float(np.min(np.linalg.norm(heavy_pos - oxygen[None, :], axis=1)))
        if min_clash < float(clash_distance_a):
            continue
        opposite = ref_lookup.get(int(atom_idx))
        opp_dist = float(np.min(np.linalg.norm(opposite - oxygen[None, :], axis=1))) if opposite is not None and len(opposite) > 0 else 999.0
        score = 4.0 - abs(opp_dist - 2.8)
        candidates.append((score, oxygen, direction))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_o: list[np.ndarray] = []
    selected_h: list[tuple[np.ndarray, np.ndarray]] = []
    for _score, oxygen, bisector in candidates:
        if len(selected_o) >= int(max_waters):
            break
        if selected_o:
            oo = np.linalg.norm(np.asarray(selected_o, dtype=np.float64) - oxygen[None, :], axis=1)
            if float(np.min(oo)) < 2.6:
                continue
        h1, h2 = _water_hydrogen_positions(oxygen, bisector_direction=bisector)
        all_existing = np.asarray(structure.positions, dtype=np.float64)
        if (
            float(np.min(np.linalg.norm(all_existing - h1[None, :], axis=1))) < 1.3
            or float(np.min(np.linalg.norm(all_existing - h2[None, :], axis=1))) < 1.3
        ):
            h1, h2 = _water_hydrogen_positions(oxygen, bisector_direction=-np.asarray(bisector, dtype=np.float64))
        selected_o.append(np.asarray(oxygen, dtype=np.float64))
        selected_h.append((np.asarray(h1, dtype=np.float64), np.asarray(h2, dtype=np.float64)))

    if not selected_o:
        return structure
    return _append_water_molecules(structure, oxygen_positions=selected_o, hydrogen_pairs=selected_h)


def prepare_structure_for_uma(
    path: str | Path,
    *,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    ph: float = 7.4,
    add_first_shell_waters: bool = True,
    max_first_shell_waters: int = 12,
    water_shell_distance_a: float = 2.8,
    water_clash_distance_a: float = 2.1,
    water_bridge_distance_min_a: float = 4.2,
    water_bridge_distance_max_a: float = 6.6,
) -> StructureData:
    resolved = str(Path(path).resolve())
    key = (
        resolved,
        str(protein_chain_id),
        str(ligand_chain_id or ""),
        tuple(int(x) for x in pocket_positions),
        float(ph),
        bool(add_first_shell_waters),
        int(max_first_shell_waters),
        float(water_shell_distance_a),
        float(water_bridge_distance_min_a),
        float(water_bridge_distance_max_a),
    )
    cached = _PREPARED_STRUCTURE_CACHE.get(key)
    if cached is not None:
        return _copy_structure_with_path(cached, path=cached.path)

    prepared = _hydrogenate_structure_openmm(resolved, ph=float(ph))
    if bool(add_first_shell_waters):
        prepared = _add_first_shell_waters(
            prepared,
            protein_chain_id=protein_chain_id,
            ligand_chain_id=ligand_chain_id,
            pocket_positions=pocket_positions,
            max_waters=int(max_first_shell_waters),
            shell_distance_a=float(water_shell_distance_a),
            clash_distance_a=float(water_clash_distance_a),
            bridge_distance_min_a=float(water_bridge_distance_min_a),
            bridge_distance_max_a=float(water_bridge_distance_max_a),
        )
    _PREPARED_STRUCTURE_CACHE[key] = _copy_structure_with_path(prepared, path=prepared.path)
    return prepared


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


def _format_pdb_atom_name(atom_name: str, symbol: str) -> str:
    name = (atom_name or "").strip()[:4]
    if not name:
        name = _normalize_symbol(symbol)
    if len(name) < 4 and len(_normalize_symbol(symbol)) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def write_multimodel_pdb(
    structure: StructureData,
    frames: list[np.ndarray],
    output_path: str | Path,
    *,
    model_metadata: list[dict[str, float | int]] | None = None,
) -> Path:
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = model_metadata or []
    with out.open("w", encoding="utf-8") as fh:
        fh.write("REMARK   Generated by ThermoGFN UMA sMD exporter\n")
        fh.write(f"REMARK   Source topology: {structure.path}\n")
        for model_idx, positions in enumerate(frames, start=1):
            if positions.shape != structure.positions.shape:
                raise ValueError(
                    f"frame {model_idx} has shape {positions.shape}, expected {structure.positions.shape}"
                )
            fh.write(f"MODEL     {model_idx:4d}\n")
            if model_idx - 1 < len(meta):
                md = meta[model_idx - 1]
                fh.write("REMARK   " + " ".join(f"{k}={v}" for k, v in md.items()) + "\n")
            for atom_idx, (xyz, grp, atom_name, res_name, chain_id, resid, symbol) in enumerate(
                zip(
                    positions,
                    structure.group_pdb,
                    structure.atom_names,
                    structure.residue_names,
                    structure.chain_ids,
                    structure.residue_ids,
                    structure.symbols,
                    strict=False,
                ),
                start=1,
            ):
                serial = ((atom_idx - 1) % 99999) + 1
                record = "HETATM" if grp == "HETATM" else "ATOM"
                name_field = _format_pdb_atom_name(atom_name, symbol)
                res_field = (res_name or "UNK")[:3]
                chain_field = (chain_id or " ")[:1]
                resid_field = int(resid) if resid is not None else 0
                elem_field = _normalize_symbol(symbol)[:2].rjust(2)
                fh.write(
                    f"{record:<6}{serial:5d} {name_field} {res_field:>3} {chain_field}{resid_field:4d}    "
                    f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}          {elem_field:>2}\n"
                )
            fh.write("ENDMDL\n")
        fh.write("END\n")
    return out


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


def protein_backbone_heavy_indices(
    structure: StructureData,
    *,
    chain_id: str,
    exclude_positions: set[int] | None = None,
    stride: int = 1,
) -> np.ndarray:
    skip = exclude_positions or set()
    step = max(1, int(stride))
    out = [
        idx
        for idx, (grp, atom_name, ch, resid, sym) in enumerate(
            zip(
                structure.group_pdb,
                structure.atom_names,
                structure.chain_ids,
                structure.residue_ids,
                structure.symbols,
                strict=False,
            )
        )
        if grp == "ATOM"
        and ch == chain_id
        and resid not in skip
        and atom_name.strip() in {"N", "CA", "C", "O"}
        and _is_heavy(sym)
    ]
    if step > 1 and len(out) > step:
        out = out[::step]
    return np.asarray(out, dtype=np.int64)


def protein_ca_indices(
    structure: StructureData,
    *,
    chain_id: str,
    exclude_positions: set[int] | None = None,
    stride: int = 1,
) -> np.ndarray:
    skip = exclude_positions or set()
    step = max(1, int(stride))
    out = [
        idx
        for idx, (grp, atom_name, ch, resid, sym) in enumerate(
            zip(
                structure.group_pdb,
                structure.atom_names,
                structure.chain_ids,
                structure.residue_ids,
                structure.symbols,
                strict=False,
            )
        )
        if grp == "ATOM"
        and ch == chain_id
        and resid not in skip
        and atom_name.strip() == "CA"
        and _is_heavy(sym)
    ]
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


def _is_backbone_atom(atom_name: str) -> bool:
    return atom_name.strip() in {"N", "CA", "C", "O"}


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
    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}
    shared = [key for key in left_map if key in right_map]
    shared.sort(key=lambda k: (k[1] if k[1] is not None else -1, atom_order.get(k[2], 99), k[2]))
    if not shared:
        raise ValueError("no shared pocket atom identities between endpoint structures")
    return (
        np.asarray([left_map[k] for k in shared], dtype=np.int64),
        np.asarray([right_map[k] for k in shared], dtype=np.int64),
    )


def _pocket_guidance_mask(structure: StructureData, indices: np.ndarray) -> np.ndarray:
    allowed = {"N", "CA", "C", "O", "CB"}
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.asarray(
        [str(structure.atom_names[int(i)]).strip() in allowed for i in idx.tolist()],
        dtype=bool,
    )


def match_backbone_ca_indices(
    left: StructureData,
    right: StructureData,
    *,
    chain_id: str,
    exclude_positions: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    skip = exclude_positions or set()
    left_map: dict[tuple[str, int | None], int] = {}
    right_map: dict[tuple[str, int | None], int] = {}
    for idx, (grp, atom_name, ch, resid, sym) in enumerate(
        zip(
            left.group_pdb,
            left.atom_names,
            left.chain_ids,
            left.residue_ids,
            left.symbols,
            strict=False,
        )
    ):
        if (
            grp == "ATOM"
            and ch == chain_id
            and resid not in skip
            and atom_name.strip() == "CA"
            and _is_heavy(sym)
        ):
            left_map[(ch, resid)] = idx
    for idx, (grp, atom_name, ch, resid, sym) in enumerate(
        zip(
            right.group_pdb,
            right.atom_names,
            right.chain_ids,
            right.residue_ids,
            right.symbols,
            strict=False,
        )
    ):
        if (
            grp == "ATOM"
            and ch == chain_id
            and resid not in skip
            and atom_name.strip() == "CA"
            and _is_heavy(sym)
        ):
            right_map[(ch, resid)] = idx
    shared = [key for key in left_map if key in right_map]
    shared.sort(key=lambda k: (k[1] if k[1] is not None else -1))
    if not shared:
        raise ValueError("no shared backbone CA atoms between endpoint structures")
    return (
        np.asarray([left_map[k] for k in shared], dtype=np.int64),
        np.asarray([right_map[k] for k in shared], dtype=np.int64),
    )


def match_backbone_heavy_indices(
    left: StructureData,
    right: StructureData,
    *,
    chain_id: str,
    exclude_positions: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    skip = exclude_positions or set()
    allowed_names = {"N", "CA", "C", "O"}
    left_map: dict[tuple[str, int | None, str], int] = {}
    right_map: dict[tuple[str, int | None, str], int] = {}
    for idx, (grp, atom_name, ch, resid, sym) in enumerate(
        zip(
            left.group_pdb,
            left.atom_names,
            left.chain_ids,
            left.residue_ids,
            left.symbols,
            strict=False,
        )
    ):
        atom_name = atom_name.strip()
        if (
            grp == "ATOM"
            and ch == chain_id
            and resid not in skip
            and atom_name in allowed_names
            and _is_heavy(sym)
        ):
            left_map[(ch, resid, atom_name)] = idx
    for idx, (grp, atom_name, ch, resid, sym) in enumerate(
        zip(
            right.group_pdb,
            right.atom_names,
            right.chain_ids,
            right.residue_ids,
            right.symbols,
            strict=False,
        )
    ):
        atom_name = atom_name.strip()
        if (
            grp == "ATOM"
            and ch == chain_id
            and resid not in skip
            and atom_name in allowed_names
            and _is_heavy(sym)
        ):
            right_map[(ch, resid, atom_name)] = idx
    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}
    shared = [key for key in left_map if key in right_map]
    shared.sort(key=lambda k: (k[1] if k[1] is not None else -1, atom_order.get(k[2], 99), k[2]))
    if not shared:
        raise ValueError("no shared backbone heavy atoms between endpoint structures")
    return (
        np.asarray([left_map[k] for k in shared], dtype=np.int64),
        np.asarray([right_map[k] for k in shared], dtype=np.int64),
    )


def build_interpolated_ca_elastic_network(
    *,
    reactant: StructureData,
    product_aligned_positions: np.ndarray,
    reactant_ca_indices: np.ndarray,
    product_ca_indices: np.ndarray,
    sequential_k_eva2: float = 6.0,
    midrange_k_eva2: float = 1.25,
    contact_k_eva2: float = 0.35,
    contact_cutoff_a: float = 8.0,
) -> dict[str, np.ndarray]:
    reactant_ca_indices = np.asarray(reactant_ca_indices, dtype=np.int64)
    product_ca_indices = np.asarray(product_ca_indices, dtype=np.int64)
    if reactant_ca_indices.shape != product_ca_indices.shape or reactant_ca_indices.size == 0:
        return {
            "pairs": np.zeros((0, 2), dtype=np.int64),
            "start_distances": np.zeros((0,), dtype=np.float64),
            "end_distances": np.zeros((0,), dtype=np.float64),
            "force_constants": np.zeros((0,), dtype=np.float64),
        }

    start = np.asarray(reactant.positions[reactant_ca_indices], dtype=np.float64)
    end = np.asarray(product_aligned_positions[product_ca_indices], dtype=np.float64)
    resids = [
        int(reactant.residue_ids[int(idx)]) if reactant.residue_ids[int(idx)] is not None else pos
        for pos, idx in enumerate(reactant_ca_indices.tolist(), start=1)
    ]

    pair_k: dict[tuple[int, int], float] = {}

    def add_pair(i: int, j: int, k_val: float) -> None:
        if i == j:
            return
        key = (min(i, j), max(i, j))
        cur = pair_k.get(key)
        if cur is None or float(k_val) > float(cur):
            pair_k[key] = float(k_val)

    for i in range(len(reactant_ca_indices) - 1):
        add_pair(i, i + 1, float(sequential_k_eva2))
    for i in range(len(reactant_ca_indices) - 2):
        add_pair(i, i + 2, float(sequential_k_eva2) * 0.5)
    if float(midrange_k_eva2) > 0.0:
        for spacing in (4, 8, 16):
            if spacing >= len(reactant_ca_indices):
                continue
            k_val = float(midrange_k_eva2) * (4.0 / float(spacing))
            for i in range(len(reactant_ca_indices) - spacing):
                add_pair(i, i + spacing, k_val)

    cutoff = float(contact_cutoff_a)
    for i in range(len(reactant_ca_indices)):
        for j in range(i + 3, len(reactant_ca_indices)):
            d0 = float(np.linalg.norm(start[i] - start[j]))
            d1 = float(np.linalg.norm(end[i] - end[j]))
            if d0 <= cutoff or d1 <= cutoff:
                add_pair(i, j, float(contact_k_eva2))

    pairs: list[list[int]] = []
    start_distances: list[float] = []
    end_distances: list[float] = []
    force_constants: list[float] = []
    for i, j in sorted(pair_k):
        pairs.append([int(reactant_ca_indices[i]), int(reactant_ca_indices[j])])
        start_distances.append(float(np.linalg.norm(start[i] - start[j])))
        end_distances.append(float(np.linalg.norm(end[i] - end[j])))
        force_constants.append(float(pair_k[(i, j)]))

    return {
        "pairs": np.asarray(pairs, dtype=np.int64),
        "start_distances": np.asarray(start_distances, dtype=np.float64),
        "end_distances": np.asarray(end_distances, dtype=np.float64),
        "force_constants": np.asarray(force_constants, dtype=np.float64),
    }


def build_backbone_geometry_restraints(
    *,
    structure: StructureData,
    chain_id: str,
    intrares_k_eva2: float = 30.0,
    peptide_k_eva2: float = 25.0,
    ca_next_k_eva2: float = 12.0,
) -> dict[str, np.ndarray]:
    residue_atoms: dict[int | None, dict[str, int]] = {}
    residue_order: list[int | None] = []
    for idx, (grp, ch, resid, atom_name, sym) in enumerate(
        zip(
            structure.group_pdb,
            structure.chain_ids,
            structure.residue_ids,
            structure.atom_names,
            structure.symbols,
            strict=False,
        )
    ):
        if grp != "ATOM" or ch != chain_id or not _is_heavy(sym):
            continue
        atom = str(atom_name).strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        if resid not in residue_atoms:
            residue_atoms[resid] = {}
            residue_order.append(resid)
        residue_atoms[resid][atom] = int(idx)

    pairs: list[list[int]] = []
    target_distances: list[float] = []
    force_constants: list[float] = []

    def add_pair(i: int, j: int, k_val: float) -> None:
        dist = float(np.linalg.norm(np.asarray(structure.positions[i] - structure.positions[j], dtype=np.float64)))
        pairs.append([int(i), int(j)])
        target_distances.append(float(dist))
        force_constants.append(float(k_val))

    for resid in residue_order:
        atoms = residue_atoms[resid]
        if "N" in atoms and "CA" in atoms:
            add_pair(atoms["N"], atoms["CA"], intrares_k_eva2)
        if "CA" in atoms and "C" in atoms:
            add_pair(atoms["CA"], atoms["C"], intrares_k_eva2)
        if "C" in atoms and "O" in atoms:
            add_pair(atoms["C"], atoms["O"], intrares_k_eva2)

    for left_resid, right_resid in zip(residue_order[:-1], residue_order[1:], strict=False):
        left_atoms = residue_atoms[left_resid]
        right_atoms = residue_atoms[right_resid]
        if "C" in left_atoms and "N" in right_atoms:
            add_pair(left_atoms["C"], right_atoms["N"], peptide_k_eva2)
        if "CA" in left_atoms and "CA" in right_atoms:
            add_pair(left_atoms["CA"], right_atoms["CA"], ca_next_k_eva2)

    return {
        "pairs": np.asarray(pairs, dtype=np.int64),
        "target_distances": np.asarray(target_distances, dtype=np.float64),
        "force_constants": np.asarray(force_constants, dtype=np.float64),
    }


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


def align_positions_on_reference(
    positions: np.ndarray,
    *,
    reference_positions: np.ndarray,
    mobile_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    mobile_indices = np.asarray(mobile_indices, dtype=np.int64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if mobile_indices.size == 0 or reference_indices.size == 0 or mobile_indices.shape != reference_indices.shape:
        return np.asarray(positions, dtype=np.float64).copy()
    rot, trans = kabsch_align(
        np.asarray(reference_positions[reference_indices], dtype=np.float64),
        np.asarray(positions[mobile_indices], dtype=np.float64),
    )
    return apply_transform(np.asarray(positions, dtype=np.float64), rot, trans)


def build_rigid_ligand_path_targets(
    *,
    ligand_positions: np.ndarray,
    matched_start_positions: np.ndarray,
    matched_end_positions: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    ligand_positions = np.asarray(ligand_positions, dtype=np.float64)
    matched_start_positions = np.asarray(matched_start_positions, dtype=np.float64)
    matched_end_positions = np.asarray(matched_end_positions, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if ligand_positions.ndim != 2 or ligand_positions.shape[1] != 3:
        raise ValueError("ligand_positions must have shape [n_atoms, 3]")
    if matched_start_positions.shape != matched_end_positions.shape or matched_start_positions.ndim != 2:
        raise ValueError("matched_start_positions and matched_end_positions must have matching shape [n_atoms, 3]")
    if matched_start_positions.shape[0] == 0:
        raise ValueError("at least one matched ligand atom is required")

    start_cent = matched_start_positions.mean(axis=0)
    end_cent = matched_end_positions.mean(axis=0)

    use_rotation = matched_start_positions.shape[0] >= 3
    rot_end = np.eye(3, dtype=np.float64)
    if use_rotation:
        try:
            rot_end, _ = kabsch_align(matched_end_positions, matched_start_positions)
        except Exception:  # noqa: BLE001
            rot_end = np.eye(3, dtype=np.float64)
            use_rotation = False

    if use_rotation:
        rots = Rotation.from_matrix(np.stack([np.eye(3, dtype=np.float64), rot_end], axis=0))
        slerp = Slerp([0.0, 1.0], rots)
        interp_rots = slerp(np.clip(lambdas, 0.0, 1.0)).as_matrix()
    else:
        interp_rots = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(lambdas), axis=0)

    centered = ligand_positions - start_cent[None, :]
    out = []
    for lam, rot in zip(lambdas.tolist(), interp_rots, strict=False):
        center = (1.0 - float(lam)) * start_cent + float(lam) * end_cent
        out.append(centered @ np.asarray(rot, dtype=np.float64).T + center[None, :])
    return np.asarray(out, dtype=np.float64)


def infer_ligand_bonds(
    *,
    symbols: list[str],
    positions: np.ndarray,
    scale: float = 1.25,
    max_bond_a: float = 2.2,
) -> set[tuple[int, int]]:
    pos = np.asarray(positions, dtype=np.float64)
    bonds: set[tuple[int, int]] = set()
    for i in range(len(symbols)):
        zi = atomic_numbers[str(symbols[i])]
        ri = float(covalent_radii[zi])
        for j in range(i + 1, len(symbols)):
            zj = atomic_numbers[str(symbols[j])]
            rj = float(covalent_radii[zj])
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cutoff = float(scale) * (ri + rj)
            if d <= cutoff and d < float(max_bond_a):
                bonds.add((i, j))
    return bonds


def connected_components_from_bonds(n_atoms: int, bonds: set[tuple[int, int]]) -> list[list[int]]:
    adj = {i: set() for i in range(int(n_atoms))}
    for i, j in bonds:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in range(int(n_atoms)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in sorted(adj[node]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (c[0], len(c)))
    return comps


def infer_ligand_graph_edits(
    *,
    symbols: list[str],
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
) -> dict[str, Any]:
    react_bonds = infer_ligand_bonds(symbols=symbols, positions=reactant_positions)
    prod_bonds = infer_ligand_bonds(symbols=symbols, positions=product_positions)
    stable_bonds = react_bonds & prod_bonds
    reactant_components = connected_components_from_bonds(len(symbols), react_bonds)
    product_components = connected_components_from_bonds(len(symbols), prod_bonds)
    broken_bonds = react_bonds - prod_bonds
    formed_bonds = prod_bonds - react_bonds
    stable_components = connected_components_from_bonds(len(symbols), stable_bonds)
    reactive_atoms_local: set[int] = set()
    for i, j in sorted(broken_bonds | formed_bonds):
        reactive_atoms_local.add(int(i))
        reactive_atoms_local.add(int(j))
    return {
        "reactant_bonds": react_bonds,
        "product_bonds": prod_bonds,
        "stable_bonds": stable_bonds,
        "reactant_components": reactant_components,
        "product_components": product_components,
        "broken_bonds": broken_bonds,
        "formed_bonds": formed_bonds,
        "stable_components": stable_components,
        "reactive_atoms_local": np.asarray(sorted(reactive_atoms_local), dtype=np.int64),
    }


def count_ligand_excess_bonds(
    *,
    symbols: list[str],
    positions: np.ndarray,
    allowed_bonds: set[tuple[int, int]],
) -> int:
    current = infer_ligand_bonds(symbols=symbols, positions=positions)
    return int(sum(1 for pair in current if pair not in allowed_bonds))


def count_close_contacts(
    positions: np.ndarray,
    *,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    threshold_a: float,
) -> int:
    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.size == 0 or right.size == 0:
        return 0
    dist = pairwise_distances(
        np.asarray(positions[left], dtype=np.float64),
        np.asarray(positions[right], dtype=np.float64),
    )
    return int(np.sum(dist < float(threshold_a)))


def elastic_network_rms_deviation(
    positions: np.ndarray,
    *,
    pairs: np.ndarray,
    target_distances: np.ndarray,
) -> float:
    pair_idx = np.asarray(pairs, dtype=np.int64)
    if pair_idx.size == 0:
        return 0.0
    current = []
    for pair in pair_idx.tolist():
        i, j = int(pair[0]), int(pair[1])
        current.append(float(np.linalg.norm(np.asarray(positions[i] - positions[j], dtype=np.float64))))
    current_arr = np.asarray(current, dtype=np.float64)
    target_arr = np.asarray(target_distances, dtype=np.float64)
    if current_arr.shape != target_arr.shape or current_arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((current_arr - target_arr) ** 2)))


def build_connectivity_aware_ligand_path_targets(
    *,
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
    symbols: list[str],
    lambdas: np.ndarray,
) -> np.ndarray:
    reactant_positions = np.asarray(reactant_positions, dtype=np.float64)
    product_positions = np.asarray(product_positions, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    react_bonds = infer_ligand_bonds(symbols=symbols, positions=reactant_positions)
    prod_bonds = infer_ligand_bonds(symbols=symbols, positions=product_positions)
    stable_bonds = react_bonds & prod_bonds
    components = connected_components_from_bonds(len(symbols), stable_bonds)

    frames: list[np.ndarray] = []
    for lam in lambdas.tolist():
        frame = np.zeros_like(reactant_positions, dtype=np.float64)
        for comp in components:
            comp_idx = np.asarray(comp, dtype=np.int64)
            start = reactant_positions[comp_idx]
            end = product_positions[comp_idx]
            if len(comp_idx) >= 3:
                rot_end, _trans_end = kabsch_align(end, start)
                rots = Rotation.from_matrix(np.stack([np.eye(3, dtype=np.float64), rot_end], axis=0))
                rot = Slerp([0.0, 1.0], rots)([float(lam)]).as_matrix()[0]
                start_cent = start.mean(axis=0)
                end_cent = end.mean(axis=0)
                centered = start - start_cent[None, :]
                center = (1.0 - float(lam)) * start_cent + float(lam) * end_cent
                frame[comp_idx] = centered @ rot.T + center[None, :]
            else:
                frame[comp_idx] = (1.0 - float(lam)) * start + float(lam) * end
        frames.append(frame)
    return np.asarray(frames, dtype=np.float64)


def build_component_rigid_ligand_path_targets(
    *,
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
    components: list[list[int]],
    lambdas: np.ndarray,
) -> np.ndarray:
    reactant_positions = np.asarray(reactant_positions, dtype=np.float64)
    product_positions = np.asarray(product_positions, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)

    frames: list[np.ndarray] = []
    for lam in lambdas.tolist():
        frame = np.zeros_like(reactant_positions, dtype=np.float64)
        for comp in components:
            comp_idx = np.asarray(comp, dtype=np.int64)
            start = reactant_positions[comp_idx]
            end = product_positions[comp_idx]
            if len(comp_idx) >= 3:
                rot_end, _trans_end = kabsch_align(end, start)
                rots = Rotation.from_matrix(np.stack([np.eye(3, dtype=np.float64), rot_end], axis=0))
                rot = Slerp([0.0, 1.0], rots)([float(lam)]).as_matrix()[0]
                start_cent = start.mean(axis=0)
                end_cent = end.mean(axis=0)
                centered = start - start_cent[None, :]
                center = (1.0 - float(lam)) * start_cent + float(lam) * end_cent
                frame[comp_idx] = centered @ rot.T + center[None, :]
            else:
                frame[comp_idx] = (1.0 - float(lam)) * start + float(lam) * end
        frames.append(frame)
    return np.asarray(frames, dtype=np.float64)


def build_reaction_center_hybrid_ligand_path_targets(
    *,
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
    components: list[list[int]],
    reactive_atoms_local: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    reactant_positions = np.asarray(reactant_positions, dtype=np.float64)
    product_positions = np.asarray(product_positions, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    reactive_set = set(np.asarray(reactive_atoms_local, dtype=np.int64).tolist())

    frames: list[np.ndarray] = []
    for lam in lambdas.tolist():
        frame = np.zeros_like(reactant_positions, dtype=np.float64)
        for comp in components:
            comp_idx = np.asarray(comp, dtype=np.int64)
            start = reactant_positions[comp_idx]
            end = product_positions[comp_idx]
            if reactive_set.intersection(int(idx) for idx in comp_idx.tolist()):
                frame[comp_idx] = (1.0 - float(lam)) * start + float(lam) * end
                continue
            if len(comp_idx) >= 3:
                rot_end, _trans_end = kabsch_align(end, start)
                rots = Rotation.from_matrix(np.stack([np.eye(3, dtype=np.float64), rot_end], axis=0))
                rot = Slerp([0.0, 1.0], rots)([float(lam)]).as_matrix()[0]
                start_cent = start.mean(axis=0)
                end_cent = end.mean(axis=0)
                centered = start - start_cent[None, :]
                center = (1.0 - float(lam)) * start_cent + float(lam) * end_cent
                frame[comp_idx] = centered @ rot.T + center[None, :]
            else:
                frame[comp_idx] = (1.0 - float(lam)) * start + float(lam) * end
        frames.append(frame)
    return np.asarray(frames, dtype=np.float64)


def build_internal_morph_ligand_path_targets(
    *,
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    reactant_positions = np.asarray(reactant_positions, dtype=np.float64)
    product_positions = np.asarray(product_positions, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if reactant_positions.shape != product_positions.shape or reactant_positions.ndim != 2:
        raise ValueError("reactant_positions and product_positions must have matching shape [n_atoms, 3]")
    frames = []
    for lam in lambdas.tolist():
        frames.append((1.0 - float(lam)) * reactant_positions + float(lam) * product_positions)
    return np.asarray(frames, dtype=np.float64)


def _empty_support_model() -> dict[str, np.ndarray]:
    return {
        "support_pairs": np.zeros((0, 2), dtype=np.int64),
        "support_start_distances": np.zeros((0,), dtype=np.float64),
        "support_end_distances": np.zeros((0,), dtype=np.float64),
        "support_force_constants": np.zeros((0,), dtype=np.float64),
    }


def _bond_adjacency(
    n_atoms: int,
    bonds: set[tuple[int, int]],
) -> list[set[int]]:
    adjacency = [set() for _ in range(int(n_atoms))]
    for i, j in bonds:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    return adjacency


def _ordered_pair(i: int, j: int) -> tuple[int, int]:
    if int(i) <= int(j):
        return int(i), int(j)
    return int(j), int(i)


def _choose_anchor_neighbor(
    atom_idx: int,
    *,
    primary_adjacency: list[set[int]],
    secondary_adjacency: list[set[int]],
    forbidden_idx: int,
) -> int | None:
    primary = sorted(int(x) for x in primary_adjacency[int(atom_idx)] if int(x) != int(forbidden_idx))
    if primary:
        return int(primary[0])
    secondary = sorted(int(x) for x in secondary_adjacency[int(atom_idx)] if int(x) != int(forbidden_idx))
    if secondary:
        return int(secondary[0])
    return None


def build_reaction_cv_model(
    *,
    reactant_positions: np.ndarray,
    product_positions: np.ndarray,
    reactant_bonds: set[tuple[int, int]],
    product_bonds: set[tuple[int, int]],
    stable_bonds: set[tuple[int, int]],
    reactant_components: list[np.ndarray],
    broken_bonds: set[tuple[int, int]],
    formed_bonds: set[tuple[int, int]],
) -> dict[str, Any]:
    start = np.asarray(reactant_positions, dtype=np.float64)
    end = np.asarray(product_positions, dtype=np.float64)
    n_atoms = int(start.shape[0])
    react_adj = _bond_adjacency(n_atoms, reactant_bonds)
    prod_adj = _bond_adjacency(n_atoms, product_bonds)
    stable_adj = _bond_adjacency(n_atoms, stable_bonds)

    component_lookup: dict[int, int] = {}
    component_start_coms: list[np.ndarray] = []
    component_end_coms: list[np.ndarray] = []
    for comp_idx, comp in enumerate(reactant_components):
        comp_arr = np.asarray(comp, dtype=np.int64)
        for atom_idx in comp_arr.tolist():
            component_lookup[int(atom_idx)] = int(comp_idx)
        component_start_coms.append(np.mean(start[comp_arr], axis=0))
        component_end_coms.append(np.mean(end[comp_arr], axis=0))

    component_pair_indices: list[list[int]] = []
    component_pair_start_distances: list[float] = []
    component_pair_end_distances: list[float] = []
    component_pair_force_constants: list[float] = []
    seen_component_pairs: set[tuple[int, int]] = set()

    aux_pairs: list[list[int]] = []
    aux_start_distances: list[float] = []
    aux_end_distances: list[float] = []
    aux_force_constants: list[float] = []
    seen_aux_pairs: set[tuple[int, int]] = set()

    def _add_aux_pair(i: int, j: int, *, force_constant: float) -> None:
        pair = _ordered_pair(i, j)
        if pair in seen_aux_pairs:
            return
        start_d = float(np.linalg.norm(start[pair[0]] - start[pair[1]]))
        end_d = float(np.linalg.norm(end[pair[0]] - end[pair[1]]))
        if abs(end_d - start_d) <= 1e-3:
            return
        seen_aux_pairs.add(pair)
        aux_pairs.append([int(pair[0]), int(pair[1])])
        aux_start_distances.append(float(start_d))
        aux_end_distances.append(float(end_d))
        aux_force_constants.append(float(force_constant))

    for i, j in sorted(formed_bonds | broken_bonds):
        pair = _ordered_pair(i, j)
        if pair in formed_bonds:
            left_comp = component_lookup.get(int(pair[0]))
            right_comp = component_lookup.get(int(pair[1]))
            if left_comp is not None and right_comp is not None and left_comp != right_comp:
                comp_pair = _ordered_pair(left_comp, right_comp)
                if comp_pair not in seen_component_pairs:
                    seen_component_pairs.add(comp_pair)
                    start_dist = float(
                        np.linalg.norm(
                            np.asarray(component_start_coms[comp_pair[0]], dtype=np.float64)
                            - np.asarray(component_start_coms[comp_pair[1]], dtype=np.float64),
                        )
                    )
                    end_dist = float(
                        np.linalg.norm(
                            np.asarray(component_end_coms[comp_pair[0]], dtype=np.float64)
                            - np.asarray(component_end_coms[comp_pair[1]], dtype=np.float64),
                        )
                    )
                    component_pair_indices.append([int(comp_pair[0]), int(comp_pair[1])])
                    component_pair_start_distances.append(float(start_dist))
                    component_pair_end_distances.append(float(end_dist))
                    component_pair_force_constants.append(2.5)

        left_neighbor = _choose_anchor_neighbor(
            int(pair[0]),
            primary_adjacency=stable_adj,
            secondary_adjacency=react_adj if pair in broken_bonds else prod_adj,
            forbidden_idx=int(pair[1]),
        )
        if left_neighbor is not None:
            _add_aux_pair(left_neighbor, int(pair[1]), force_constant=2.0)
        right_neighbor = _choose_anchor_neighbor(
            int(pair[1]),
            primary_adjacency=stable_adj,
            secondary_adjacency=react_adj if pair in broken_bonds else prod_adj,
            forbidden_idx=int(pair[0]),
        )
        if right_neighbor is not None:
            _add_aux_pair(int(pair[0]), right_neighbor, force_constant=2.0)

    return {
        "component_pair_indices": np.asarray(component_pair_indices, dtype=np.int64),
        "component_pair_start_distances": np.asarray(component_pair_start_distances, dtype=np.float64),
        "component_pair_end_distances": np.asarray(component_pair_end_distances, dtype=np.float64),
        "component_pair_force_constants": np.asarray(component_pair_force_constants, dtype=np.float64),
        "aux_pairs": np.asarray(aux_pairs, dtype=np.int64),
        "aux_start_distances": np.asarray(aux_start_distances, dtype=np.float64),
        "aux_end_distances": np.asarray(aux_end_distances, dtype=np.float64),
        "aux_force_constants": np.asarray(aux_force_constants, dtype=np.float64),
    }


def build_component_pocket_support_model(
    *,
    reactant: StructureData,
    product_aligned_positions: np.ndarray,
    ligand_indices: np.ndarray,
    pocket_react_indices: np.ndarray,
    pocket_prod_indices: np.ndarray,
    graph_model: dict[str, Any],
    max_pocket_anchors_per_component: int = 2,
) -> dict[str, np.ndarray]:
    if not bool(graph_model.get("full_mapping", False)):
        return {
            "support_pairs": np.zeros((0, 2), dtype=np.int64),
            "support_start_distances": np.zeros((0,), dtype=np.float64),
            "support_end_distances": np.zeros((0,), dtype=np.float64),
            "support_force_constants": np.zeros((0,), dtype=np.float64),
        }

    ligand_indices = np.asarray(ligand_indices, dtype=np.int64)
    pocket_react = np.asarray(pocket_react_indices, dtype=np.int64)
    pocket_prod = np.asarray(pocket_prod_indices, dtype=np.int64)
    if pocket_react.size == 0 or pocket_prod.size == 0:
        return {
            "support_pairs": np.zeros((0, 2), dtype=np.int64),
            "support_start_distances": np.zeros((0,), dtype=np.float64),
            "support_end_distances": np.zeros((0,), dtype=np.float64),
            "support_force_constants": np.zeros((0,), dtype=np.float64),
        }

    react_local = np.asarray(graph_model["reactant_positions_local"], dtype=np.float64)
    prod_local = np.asarray(graph_model["product_positions_local"], dtype=np.float64)
    steering_mode = str(graph_model.get("steering_mode", "component_pose_only"))
    if steering_mode == "reactive_center":
        base_force_constant = 0.75
    elif steering_mode == "component_pose_only":
        base_force_constant = 2.0
    else:
        base_force_constant = 2.5

    support_pairs: list[list[int]] = []
    support_start_distances: list[float] = []
    support_end_distances: list[float] = []
    support_force_constants: list[float] = []
    seen: set[tuple[int, int]] = set()

    for comp in graph_model.get("reactant_components", []):
        comp_local = np.asarray(comp, dtype=np.int64)
        if comp_local.size == 0:
            continue
        comp_start = react_local[comp_local]
        comp_end = prod_local[comp_local]
        comp_start_com = np.mean(comp_start, axis=0)
        comp_end_com = np.mean(comp_end, axis=0)

        rep_local = int(comp_local[np.argmin(np.linalg.norm(comp_start - comp_start_com[None, :], axis=1))])
        rep_locals = [rep_local]
        if comp_local.size >= 3:
            rep_pos = react_local[rep_local]
            d = np.linalg.norm(comp_start - rep_pos[None, :], axis=1)
            far_local = int(comp_local[np.argmax(d)])
            if far_local != rep_local and float(np.max(d)) > 0.5:
                rep_locals.append(far_local)

        avg_pocket_dist = 0.5 * (
            np.linalg.norm(reactant.positions[pocket_react] - comp_start_com[None, :], axis=1)
            + np.linalg.norm(product_aligned_positions[pocket_prod] - comp_end_com[None, :], axis=1)
        )
        anchor_order = np.argsort(avg_pocket_dist)[: max(1, int(max_pocket_anchors_per_component))]
        for rep_rank, rep_local_idx in enumerate(rep_locals):
            lig_global = int(ligand_indices[int(rep_local_idx)])
            lig_start = np.asarray(reactant.positions[lig_global], dtype=np.float64)
            lig_end = np.asarray(prod_local[int(rep_local_idx)], dtype=np.float64)
            for anchor_rank, anchor_pos in enumerate(anchor_order.tolist()):
                pocket_global = int(pocket_react[int(anchor_pos)])
                pocket_prod_global = int(pocket_prod[int(anchor_pos)])
                pair = (lig_global, pocket_global)
                if pair in seen:
                    continue
                start_d = float(np.linalg.norm(lig_start - np.asarray(reactant.positions[pocket_global], dtype=np.float64)))
                end_d = float(np.linalg.norm(lig_end - np.asarray(product_aligned_positions[pocket_prod_global], dtype=np.float64)))
                if abs(end_d - start_d) <= 0.05:
                    continue
                weight = 1.0 / float(1 + rep_rank + anchor_rank)
                support_pairs.append([lig_global, pocket_global])
                support_start_distances.append(start_d)
                support_end_distances.append(end_d)
                support_force_constants.append(float(base_force_constant * weight))
                seen.add(pair)

    return {
        "support_pairs": np.asarray(support_pairs, dtype=np.int64),
        "support_start_distances": np.asarray(support_start_distances, dtype=np.float64),
        "support_end_distances": np.asarray(support_end_distances, dtype=np.float64),
        "support_force_constants": np.asarray(support_force_constants, dtype=np.float64),
    }


def build_ligand_graph_model(
    *,
    reactant: StructureData,
    ligand_indices: np.ndarray,
    mapping: dict[str, Any],
    max_reactive_bonds: int = 8,
    max_reactive_atoms: int = 12,
    max_reactive_fraction: float = 0.35,
) -> dict[str, Any]:
    ligand_indices = np.asarray(ligand_indices, dtype=np.int64)
    mapped_react = np.asarray(mapping["reactant_indices"], dtype=np.int64)
    mapped_prod = np.asarray(mapping["product_indices"], dtype=np.int64)
    product_aligned = np.asarray(mapping["product_aligned_positions"], dtype=np.float64)

    fallback_groups = [np.asarray([int(idx)], dtype=np.int64) for idx in ligand_indices.tolist()]
    fallback_start_coms = np.asarray(
        [np.asarray(reactant.positions[int(idx)], dtype=np.float64) for idx in ligand_indices.tolist()],
        dtype=np.float64,
    )
    if len(mapped_react) != len(ligand_indices) or not np.array_equal(np.sort(mapped_react), np.sort(ligand_indices)):
        return {
            "full_mapping": False,
            "reactant_positions_local": np.asarray(reactant.positions[ligand_indices], dtype=np.float64),
            "product_positions_local": np.asarray(reactant.positions[ligand_indices], dtype=np.float64),
            "symbols": [str(reactant.symbols[int(idx)]) for idx in ligand_indices.tolist()],
            "reactant_components": [list(group.tolist()) for group in fallback_groups],
            "stable_components": [list(group.tolist()) for group in fallback_groups],
            "component_groups": fallback_groups,
            "component_start_coms": fallback_start_coms,
            "component_end_coms": fallback_start_coms.copy(),
            "steering_mode": "unmapped_fallback",
            "steering_confident": False,
            "broken_bond_count": 0,
            "formed_bond_count": 0,
            "reactive_atom_count": 0,
            "reactive_fraction": 0.0,
            "reactive_atoms_local": np.zeros((0,), dtype=np.int64),
            "reactant_bonds": set(),
            "product_bonds": set(),
            "stable_bonds": set(),
            "allowed_bonds": set(),
            "formed_bonds": set(),
            "broken_bonds": set(),
            "component_pair_indices": np.zeros((0, 2), dtype=np.int64),
            "component_pair_start_distances": np.zeros((0,), dtype=np.float64),
            "component_pair_end_distances": np.zeros((0,), dtype=np.float64),
            "component_pair_force_constants": np.zeros((0,), dtype=np.float64),
            "aux_pairs": np.zeros((0, 2), dtype=np.int64),
            "aux_start_distances": np.zeros((0,), dtype=np.float64),
            "aux_end_distances": np.zeros((0,), dtype=np.float64),
            "aux_force_constants": np.zeros((0,), dtype=np.float64),
            "quality_reason": "incomplete_atom_mapping",
        }

    prod_lookup = {int(ridx): int(pidx) for ridx, pidx in zip(mapped_react.tolist(), mapped_prod.tolist(), strict=False)}
    reordered_prod = np.asarray([prod_lookup[int(ridx)] for ridx in ligand_indices.tolist()], dtype=np.int64)
    start = np.asarray(reactant.positions[ligand_indices], dtype=np.float64)
    end = np.asarray(product_aligned[reordered_prod], dtype=np.float64)
    symbols = [str(reactant.symbols[int(idx)]) for idx in ligand_indices.tolist()]

    graph_edits = infer_ligand_graph_edits(
        symbols=symbols,
        reactant_positions=start,
        product_positions=end,
    )
    broken_bond_count = int(len(graph_edits["broken_bonds"]))
    formed_bond_count = int(len(graph_edits["formed_bonds"]))
    reactive_atoms_local = np.asarray(graph_edits["reactive_atoms_local"], dtype=np.int64)
    reactive_atom_count = int(len(reactive_atoms_local))
    reactive_fraction = float(reactive_atom_count) / max(1.0, float(len(ligand_indices)))
    reactive_bond_count = broken_bond_count + formed_bond_count
    fraction_gate_ok = (
        reactive_fraction <= float(max_reactive_fraction)
        or int(len(ligand_indices)) <= int(max_reactive_atoms)
    )
    steering_confident = (
        reactive_bond_count <= int(max_reactive_bonds)
        and reactive_atom_count <= int(max_reactive_atoms)
        and fraction_gate_ok
    )
    steering_mode = "reactive_center" if steering_confident and reactive_bond_count > 0 else "component_pose_only"
    quality_reasons: list[str] = []
    if reactive_bond_count > int(max_reactive_bonds):
        quality_reasons.append("too_many_graph_edits")
    if reactive_atom_count > int(max_reactive_atoms):
        quality_reasons.append("too_many_reactive_atoms")
    if not fraction_gate_ok:
        quality_reasons.append("reactive_fraction_too_large")
    if reactive_bond_count == 0:
        quality_reasons.append("no_detected_graph_edits")

    reactant_components = [np.asarray(comp, dtype=np.int64) for comp in graph_edits["reactant_components"]]
    component_start_coms = np.asarray(
        [np.mean(start[comp], axis=0) for comp in reactant_components],
        dtype=np.float64,
    )
    component_end_coms = np.asarray(
        [np.mean(end[comp], axis=0) for comp in reactant_components],
        dtype=np.float64,
    )
    reaction_cv_model = build_reaction_cv_model(
        reactant_positions=start,
        product_positions=end,
        reactant_bonds=set(graph_edits["reactant_bonds"]),
        product_bonds=set(graph_edits["product_bonds"]),
        stable_bonds=set(graph_edits["stable_bonds"]),
        reactant_components=reactant_components,
        broken_bonds=set(graph_edits["broken_bonds"]),
        formed_bonds=set(graph_edits["formed_bonds"]),
    )

    return {
        "full_mapping": True,
        "reactant_positions_local": start,
        "product_positions_local": end,
        "symbols": symbols,
        "reactant_components": [comp.copy() for comp in reactant_components],
        "stable_components": [np.asarray(comp, dtype=np.int64) for comp in graph_edits["stable_components"]],
        "component_groups": [
            np.asarray([int(ligand_indices[int(i)]) for i in comp.tolist()], dtype=np.int64)
            for comp in reactant_components
        ],
        "component_start_coms": component_start_coms,
        "component_end_coms": component_end_coms,
        "steering_mode": steering_mode,
        "steering_confident": bool(steering_confident),
        "broken_bond_count": broken_bond_count,
        "formed_bond_count": formed_bond_count,
        "reactive_atom_count": reactive_atom_count,
        "reactive_fraction": float(reactive_fraction),
        "reactive_atoms_local": reactive_atoms_local,
        "reactant_bonds": set(graph_edits["reactant_bonds"]),
        "product_bonds": set(graph_edits["product_bonds"]),
        "stable_bonds": set(graph_edits["stable_bonds"]),
        "formed_bonds": set(graph_edits["formed_bonds"]),
        "broken_bonds": set(graph_edits["broken_bonds"]),
        "allowed_bonds": set(graph_edits["reactant_bonds"]) | set(graph_edits["product_bonds"]),
        "component_pair_indices": np.asarray(reaction_cv_model["component_pair_indices"], dtype=np.int64),
        "component_pair_start_distances": np.asarray(reaction_cv_model["component_pair_start_distances"], dtype=np.float64),
        "component_pair_end_distances": np.asarray(reaction_cv_model["component_pair_end_distances"], dtype=np.float64),
        "component_pair_force_constants": np.asarray(reaction_cv_model["component_pair_force_constants"], dtype=np.float64),
        "aux_pairs": np.asarray(reaction_cv_model["aux_pairs"], dtype=np.int64),
        "aux_start_distances": np.asarray(reaction_cv_model["aux_start_distances"], dtype=np.float64),
        "aux_end_distances": np.asarray(reaction_cv_model["aux_end_distances"], dtype=np.float64),
        "aux_force_constants": np.asarray(reaction_cv_model["aux_force_constants"], dtype=np.float64),
        "quality_reason": ",".join(quality_reasons) if quality_reasons else "trusted_reactive_center",
    }


def classify_ligand_protocol_mode(
    *,
    steering_mode: str,
    steering_confident: bool,
    quality_reason: str,
) -> dict[str, Any]:
    steering_mode = str(steering_mode or "").strip() or "unknown"
    quality_reason = str(quality_reason or "").strip()
    if steering_mode == "reactive_center" and bool(steering_confident):
        return {
            "protocol_mode": "reactive_center",
            "protocol_reason": quality_reason or "trusted_reactive_center",
            "reactive_barrier_valid": True,
            "pmf_eligible": True,
        }
    if steering_mode == "component_pose_only":
        if quality_reason == "incomplete_atom_mapping":
            return {
                "protocol_mode": "unsupported_reactive_path",
                "protocol_reason": quality_reason,
                "reactive_barrier_valid": False,
                "pmf_eligible": False,
            }
        return {
            "protocol_mode": "conformational_endpoint",
            "protocol_reason": quality_reason or "component_pose_only",
            "reactive_barrier_valid": False,
            "pmf_eligible": False,
        }
    return {
        "protocol_mode": "unsupported_reactive_path",
        "protocol_reason": quality_reason or steering_mode,
        "reactive_barrier_valid": False,
        "pmf_eligible": False,
    }


def analyze_endpoint_protocol(
    *,
    reactant: StructureData,
    product: StructureData,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    max_reactive_bonds: int = 8,
    max_reactive_atoms: int = 12,
    max_reactive_fraction: float = 0.35,
) -> dict[str, Any]:
    mapping = map_ligand_atoms_between_endpoints(
        reactant,
        product,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pocket_positions=pocket_positions,
    )
    ligand_indices = ligand_heavy_indices(reactant, chain_id=ligand_chain_id)
    if len(ligand_indices) == 0:
        raise ValueError("no ligand atoms selected for endpoint protocol analysis")
    graph_model = build_ligand_graph_model(
        reactant=reactant,
        ligand_indices=ligand_indices,
        mapping=mapping,
        max_reactive_bonds=int(max_reactive_bonds),
        max_reactive_atoms=int(max_reactive_atoms),
        max_reactive_fraction=float(max_reactive_fraction),
    )
    ligand_restraints = build_ligand_restraint_model(
        reactant=reactant,
        ligand_indices=ligand_indices,
        mapping=mapping,
        graph_model=graph_model,
    )
    steering_mode = str(ligand_restraints.get("steering_mode", graph_model.get("steering_mode", "unknown")))
    protocol_meta = classify_ligand_protocol_mode(
        steering_mode=steering_mode,
        steering_confident=bool(ligand_restraints.get("steering_confident", False)),
        quality_reason=str(ligand_restraints.get("quality_reason", "")),
    )
    return {
        "mapping": mapping,
        "ligand_indices": np.asarray(ligand_indices, dtype=np.int64),
        "graph_model": graph_model,
        "ligand_restraints": ligand_restraints,
        "steering_mode": steering_mode,
        "protocol_meta": protocol_meta,
    }


def build_guided_ligand_path_targets(
    *,
    reactant: StructureData,
    ligand_indices: np.ndarray,
    mapping: dict[str, Any],
    lambdas: np.ndarray,
    graph_model: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    ligand_indices = np.asarray(ligand_indices, dtype=np.int64)
    mapped_react = np.asarray(mapping["reactant_indices"], dtype=np.int64)
    mapped_prod = np.asarray(mapping["product_indices"], dtype=np.int64)
    product_aligned = np.asarray(mapping["product_aligned_positions"], dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    graph_model = build_ligand_graph_model(
        reactant=reactant,
        ligand_indices=ligand_indices,
        mapping=mapping,
    ) if graph_model is None else graph_model

    if len(mapped_react) == len(ligand_indices) and np.array_equal(np.sort(mapped_react), np.sort(ligand_indices)):
        reorder = []
        prod_lookup = {int(ridx): int(pidx) for ridx, pidx in zip(mapped_react.tolist(), mapped_prod.tolist(), strict=False)}
        for ridx in ligand_indices.tolist():
            reorder.append(prod_lookup[int(ridx)])
        start = reactant.positions[ligand_indices].copy()
        end = product_aligned[np.asarray(reorder, dtype=np.int64)].copy()
        if graph_model["steering_mode"] == "reactive_center":
            path = build_internal_morph_ligand_path_targets(
                reactant_positions=start,
                product_positions=end,
                lambdas=lambdas,
            )
            return np.asarray(path, dtype=np.float64), "reactive_center_internal_morph", graph_model
        path = build_connectivity_aware_ligand_path_targets(
            reactant_positions=start,
            product_positions=end,
            symbols=list(graph_model["symbols"]),
            lambdas=lambdas,
        )
        return np.asarray(path, dtype=np.float64), "connectivity_aware_endpoint", graph_model

    path = build_rigid_ligand_path_targets(
        ligand_positions=reactant.positions[ligand_indices],
        matched_start_positions=reactant.positions[mapped_react],
        matched_end_positions=product_aligned[mapped_prod],
        lambdas=lambdas,
    )
    return path, "rigid_fallback", graph_model


def build_ligand_restraint_model(
    *,
    reactant: StructureData,
    ligand_indices: np.ndarray,
    mapping: dict[str, Any],
    reactive_atom_steer_scale: float = 0.02,
    context_atom_steer_scale: float = 0.08,
    graph_model: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    ligand_indices = np.asarray(ligand_indices, dtype=np.int64)
    graph_model = build_ligand_graph_model(
        reactant=reactant,
        ligand_indices=ligand_indices,
        mapping=mapping,
    ) if graph_model is None else graph_model

    if not bool(graph_model.get("full_mapping", False)):
        return {
            "bond_pairs": np.zeros((0, 2), dtype=np.int64),
            "bond_start_distances": np.zeros((0,), dtype=np.float64),
            "bond_end_distances": np.zeros((0,), dtype=np.float64),
            "bond_start_lambdas": np.zeros((0,), dtype=np.float64),
            "bond_end_lambdas": np.zeros((0,), dtype=np.float64),
            "bond_start_force_constants": np.zeros((0,), dtype=np.float64),
            "bond_end_force_constants": np.zeros((0,), dtype=np.float64),
            "repulsion_pairs": np.zeros((0, 2), dtype=np.int64),
            "repulsion_min_distances": np.zeros((0,), dtype=np.float64),
            "repulsion_force_constants": np.zeros((0,), dtype=np.float64),
            "steer_weights": np.full(len(ligand_indices), max(float(context_atom_steer_scale), 0.50), dtype=np.float64),
            "component_groups": list(graph_model["component_groups"]),
            "component_start_coms": np.asarray(graph_model["component_start_coms"], dtype=np.float64),
            "component_end_coms": np.asarray(graph_model["component_end_coms"], dtype=np.float64),
            "broken_bond_count": 0,
            "formed_bond_count": 0,
            "reactive_atom_count": 0,
            "allowed_bonds": set(),
            "ligand_symbols": list(graph_model["symbols"]),
            "steering_mode": str(graph_model.get("steering_mode", "unmapped_fallback")),
            "steering_confident": bool(graph_model.get("steering_confident", False)),
            "component_pair_indices": np.zeros((0, 2), dtype=np.int64),
            "component_pair_start_distances": np.zeros((0,), dtype=np.float64),
            "component_pair_end_distances": np.zeros((0,), dtype=np.float64),
            "component_pair_force_constants": np.zeros((0,), dtype=np.float64),
            "aux_pairs": np.zeros((0, 2), dtype=np.int64),
            "aux_start_distances": np.zeros((0,), dtype=np.float64),
            "aux_end_distances": np.zeros((0,), dtype=np.float64),
            "aux_force_constants": np.zeros((0,), dtype=np.float64),
            "quality_reason": str(graph_model.get("quality_reason", "incomplete_atom_mapping")),
        }
    start = np.asarray(graph_model["reactant_positions_local"], dtype=np.float64)
    end = np.asarray(graph_model["product_positions_local"], dtype=np.float64)
    symbols = list(graph_model["symbols"])
    react_bonds = set(graph_model["reactant_bonds"])
    prod_bonds = set(graph_model["product_bonds"])
    reactive_atoms_local = np.asarray(graph_model["reactive_atoms_local"], dtype=np.int64)
    union_bonds = sorted(react_bonds | prod_bonds)
    bond_pairs: list[list[int]] = []
    bond_start_distances: list[float] = []
    bond_end_distances: list[float] = []
    bond_start_lambdas: list[float] = []
    bond_end_lambdas: list[float] = []
    bond_start_force_constants: list[float] = []
    bond_end_force_constants: list[float] = []

    for i, j in union_bonds:
        start_d = float(np.linalg.norm(start[i] - start[j]))
        end_d = float(np.linalg.norm(end[i] - end[j]))
        if str(graph_model.get("steering_mode", "")) != "reactive_center":
            continue
        if (i, j) in react_bonds and (i, j) in prod_bonds:
            lam0, lam1 = 0.0, 1.0
            k_start, k_end = 10.0, 10.0
        elif (i, j) in react_bonds:
            lam0, lam1 = 0.0, 0.50
            k_start, k_end = 6.0, 0.0
            end_d = min(2.0, max(start_d + 0.35, 1.65))
        else:
            lam0, lam1 = 0.50, 1.0
            k_start, k_end = 0.0, 6.0
            start_d = min(2.0, max(end_d + 0.35, 1.65))
        bond_pairs.append([int(ligand_indices[i]), int(ligand_indices[j])])
        bond_start_distances.append(float(start_d))
        bond_end_distances.append(float(end_d))
        bond_start_lambdas.append(float(lam0))
        bond_end_lambdas.append(float(lam1))
        bond_start_force_constants.append(float(k_start))
        bond_end_force_constants.append(float(k_end))

    repulsion_pairs: list[list[int]] = []
    repulsion_min_distances: list[float] = []
    repulsion_force_constants: list[float] = []
    bonded = set(union_bonds)
    for i in range(len(ligand_indices)):
        for j in range(i + 1, len(ligand_indices)):
            if (i, j) in bonded:
                continue
            repulsion_pairs.append([int(ligand_indices[i]), int(ligand_indices[j])])
            repulsion_min_distances.append(2.30)
            repulsion_force_constants.append(12.0)

    if str(graph_model.get("steering_mode", "")) == "reactive_center":
        steer_weights = np.zeros(len(ligand_indices), dtype=np.float64)
        if reactive_atoms_local.size > 0:
            steer_weights[reactive_atoms_local] = 1.0
    else:
        steer_weights = np.full(len(ligand_indices), max(float(context_atom_steer_scale), 0.50), dtype=np.float64)

    return {
        "bond_pairs": np.asarray(bond_pairs, dtype=np.int64),
        "bond_start_distances": np.asarray(bond_start_distances, dtype=np.float64),
        "bond_end_distances": np.asarray(bond_end_distances, dtype=np.float64),
        "bond_start_lambdas": np.asarray(bond_start_lambdas, dtype=np.float64),
        "bond_end_lambdas": np.asarray(bond_end_lambdas, dtype=np.float64),
        "bond_start_force_constants": np.asarray(bond_start_force_constants, dtype=np.float64),
        "bond_end_force_constants": np.asarray(bond_end_force_constants, dtype=np.float64),
        "repulsion_pairs": np.asarray(repulsion_pairs, dtype=np.int64),
        "repulsion_min_distances": np.asarray(repulsion_min_distances, dtype=np.float64),
        "repulsion_force_constants": np.asarray(repulsion_force_constants, dtype=np.float64),
        "steer_weights": np.asarray(steer_weights, dtype=np.float64),
        "component_groups": list(graph_model["component_groups"]),
        "component_start_coms": np.asarray(graph_model["component_start_coms"], dtype=np.float64),
        "component_end_coms": np.asarray(graph_model["component_end_coms"], dtype=np.float64),
        "broken_bond_count": int(graph_model["broken_bond_count"]),
        "formed_bond_count": int(graph_model["formed_bond_count"]),
        "reactive_atom_count": int(len(reactive_atoms_local)),
        "allowed_bonds": set(graph_model["allowed_bonds"]),
        "ligand_symbols": list(symbols),
        "steering_mode": str(graph_model.get("steering_mode", "component_pose_only")),
        "steering_confident": bool(graph_model.get("steering_confident", False)),
        "component_pair_indices": np.asarray(graph_model.get("component_pair_indices", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64),
        "component_pair_start_distances": np.asarray(graph_model.get("component_pair_start_distances", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "component_pair_end_distances": np.asarray(graph_model.get("component_pair_end_distances", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "component_pair_force_constants": np.asarray(graph_model.get("component_pair_force_constants", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "aux_pairs": np.asarray(graph_model.get("aux_pairs", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64),
        "aux_start_distances": np.asarray(graph_model.get("aux_start_distances", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "aux_end_distances": np.asarray(graph_model.get("aux_end_distances", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "aux_force_constants": np.asarray(graph_model.get("aux_force_constants", np.zeros((0,), dtype=np.float64)), dtype=np.float64),
        "quality_reason": str(graph_model.get("quality_reason", "trusted_reactive_center")),
    }


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
    alignment_mode = "pocket_identity"
    try:
        react_align_idx, prod_align_idx = match_backbone_ca_indices(
            reactant,
            product,
            chain_id=protein_chain_id,
        )
        alignment_mode = "backbone_ca"
    except Exception:  # noqa: BLE001
        try:
            react_align_idx, prod_align_idx = match_backbone_heavy_indices(
                reactant,
                product,
                chain_id=protein_chain_id,
            )
            alignment_mode = "backbone_heavy"
        except Exception:  # noqa: BLE001
            react_align_idx, prod_align_idx = match_identity_indices(
                reactant,
                product,
                chain_id=protein_chain_id,
                residue_positions=pocket_positions,
            )
    rot, trans = kabsch_align(
        reactant.positions[react_align_idx],
        product.positions[prod_align_idx],
    )
    product_aligned = apply_transform(product.positions, rot, trans)

    react_lig_idx = ligand_heavy_indices(reactant, chain_id=ligand_chain_id)
    prod_lig_idx = ligand_heavy_indices(product, chain_id=ligand_chain_id)
    if len(react_lig_idx) == 0 or len(prod_lig_idx) == 0:
        raise ValueError("ligand atoms missing in reactant or product structure")

    react_sym = np.asarray([reactant.symbols[i] for i in react_lig_idx], dtype=object)
    prod_sym = np.asarray([product.symbols[i] for i in prod_lig_idx], dtype=object)
    react_names = np.asarray([reactant.atom_names[i].strip() for i in react_lig_idx], dtype=object)
    prod_names = np.asarray([product.atom_names[i].strip() for i in prod_lig_idx], dtype=object)

    pair_left: list[int] = []
    pair_right: list[int] = []
    per_element: dict[str, int] = {}
    matched_left: set[int] = set()
    matched_right: set[int] = set()

    react_name_map: dict[tuple[str, str], list[int]] = {}
    prod_name_map: dict[tuple[str, str], list[int]] = {}
    for local_idx, (atom_name, element) in enumerate(zip(react_names.tolist(), react_sym.tolist(), strict=False)):
        react_name_map.setdefault((str(atom_name), str(element)), []).append(int(local_idx))
    for local_idx, (atom_name, element) in enumerate(zip(prod_names.tolist(), prod_sym.tolist(), strict=False)):
        prod_name_map.setdefault((str(atom_name), str(element)), []).append(int(local_idx))

    exact_name_matches = 0
    for key in sorted(set(react_name_map) & set(prod_name_map)):
        left_local = react_name_map[key]
        right_local = prod_name_map[key]
        n_take = min(len(left_local), len(right_local))
        if n_take <= 0:
            continue
        for offset in range(n_take):
            li = int(left_local[offset])
            ri = int(right_local[offset])
            pair_left.append(int(react_lig_idx[li]))
            pair_right.append(int(prod_lig_idx[ri]))
            matched_left.add(li)
            matched_right.add(ri)
            exact_name_matches += 1
        per_element[key[1]] = int(per_element.get(key[1], 0) + n_take)

    for element in sorted(set(react_sym.tolist()) & set(prod_sym.tolist())):
        left_local = np.asarray(
            [idx for idx in np.where(react_sym == element)[0].tolist() if int(idx) not in matched_left],
            dtype=np.int64,
        )
        right_local = np.asarray(
            [idx for idx in np.where(prod_sym == element)[0].tolist() if int(idx) not in matched_right],
            dtype=np.int64,
        )
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
        for row_idx, col_idx in zip(rows.tolist(), cols.tolist(), strict=False):
            matched_left.add(int(left_local[row_idx]))
            matched_right.add(int(right_local[col_idx]))
        per_element[element] = int(per_element.get(element, 0) + n_take)

    if not pair_left:
        raise ValueError("no ligand atom correspondence found between reactant and product")

    return {
        "reactant_indices": np.asarray(pair_left, dtype=np.int64),
        "product_indices": np.asarray(pair_right, dtype=np.int64),
        "product_aligned_positions": product_aligned,
        "shared_atom_count": int(len(pair_left)),
        "element_counts": per_element,
        "exact_name_matches": int(exact_name_matches),
        "alignment_mode": alignment_mode,
        "alignment_atom_count": int(len(react_align_idx)),
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


def _friction_ps_inv_to_ase(friction_ps_inv: float) -> float:
    from ase import units

    return float(friction_ps_inv) / (1000.0 * float(units.fs))


def _scheduled_pair_energy_forces(
    positions: np.ndarray,
    *,
    pairs: np.ndarray,
    start_distances: np.ndarray,
    end_distances: np.ndarray,
    force_constants: np.ndarray,
    lam: float,
    start_lambdas: np.ndarray | None = None,
    end_lambdas: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    pair_idx = np.asarray(pairs, dtype=np.int64)
    if pair_idx.size == 0:
        return float(energy), forces
    start_d = np.asarray(start_distances, dtype=np.float64).reshape(-1)
    end_d = np.asarray(end_distances, dtype=np.float64).reshape(-1)
    ks = np.asarray(force_constants, dtype=np.float64)
    lam0 = np.asarray(start_lambdas, dtype=np.float64).reshape(-1) if start_lambdas is not None else None
    lam1 = np.asarray(end_lambdas, dtype=np.float64).reshape(-1) if end_lambdas is not None else None
    for idx, pair in enumerate(pair_idx.tolist()):
        i, j = int(pair[0]), int(pair[1])
        vec = np.asarray(positions[i] - positions[j], dtype=np.float64)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-8:
            continue
        if lam0 is None or lam1 is None:
            alpha = float(lam)
            if ks.ndim == 2:
                k_pair = float((1.0 - alpha) * ks[idx, 0] + alpha * ks[idx, 1])
            else:
                k_pair = float(ks.reshape(-1)[idx])
        else:
            lo = float(lam0[idx])
            hi = float(lam1[idx])
            if hi <= lo:
                alpha = 1.0
            else:
                alpha = min(1.0, max(0.0, (float(lam) - lo) / (hi - lo)))
            if ks.ndim == 2:
                k_pair = float((1.0 - alpha) * ks[idx, 0] + alpha * ks[idx, 1])
            else:
                k_pair = float(ks.reshape(-1)[idx])
        if k_pair <= 1e-12:
            continue
        target = (1.0 - alpha) * float(start_d[idx]) + alpha * float(end_d[idx])
        delta = dist - target
        unit_vec = vec / dist
        energy += 0.5 * float(k_pair) * float(delta * delta)
        pair_force = -float(k_pair) * float(delta) * unit_vec
        forces[i] += pair_force
        forces[j] -= pair_force
    return float(energy), forces


def _scheduled_component_pair_distance_energy_forces(
    positions: np.ndarray,
    *,
    component_groups: list[np.ndarray],
    component_pair_indices: np.ndarray,
    start_distances: np.ndarray,
    end_distances: np.ndarray,
    force_constants: np.ndarray,
    lam: float,
) -> tuple[float, np.ndarray]:
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    pair_idx = np.asarray(component_pair_indices, dtype=np.int64)
    if pair_idx.size == 0:
        return float(energy), forces
    start_d = np.asarray(start_distances, dtype=np.float64).reshape(-1)
    end_d = np.asarray(end_distances, dtype=np.float64).reshape(-1)
    ks = np.asarray(force_constants, dtype=np.float64).reshape(-1)
    for idx, comp_pair in enumerate(pair_idx.tolist()):
        left_idx, right_idx = int(comp_pair[0]), int(comp_pair[1])
        if left_idx >= len(component_groups) or right_idx >= len(component_groups):
            continue
        left_members = np.asarray(component_groups[left_idx], dtype=np.int64)
        right_members = np.asarray(component_groups[right_idx], dtype=np.int64)
        if left_members.size == 0 or right_members.size == 0:
            continue
        left_com = np.mean(np.asarray(positions[left_members], dtype=np.float64), axis=0)
        right_com = np.mean(np.asarray(positions[right_members], dtype=np.float64), axis=0)
        vec = left_com - right_com
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-8:
            continue
        target = (1.0 - float(lam)) * float(start_d[idx]) + float(lam) * float(end_d[idx])
        k_pair = float(ks[idx])
        if k_pair <= 1e-12:
            continue
        delta = dist - target
        unit_vec = vec / dist
        energy += 0.5 * k_pair * float(delta * delta)
        com_force = -k_pair * float(delta) * unit_vec
        forces[left_members] += com_force / float(left_members.size)
        forces[right_members] -= com_force / float(right_members.size)
    return float(energy), forces


def relax_structure_under_uma(
    *,
    structure: StructureData,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    steps: int = 25,
    fmax_eva: float = 0.20,
) -> StructureData:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    n_steps = max(0, int(steps))
    if n_steps <= 0:
        return _copy_structure_with_path(structure, path=structure.path)
    atoms = structure.to_ase_atoms()
    atoms.calc = _get_uma_calculator(model_name, device, workers=calculator_workers)
    rng = np.random.RandomState(int(abs(hash((structure.path, model_name, device, n_steps))) % (2**31 - 1)))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = Langevin(
        atoms,
        0.05 * units.fs,
        temperature_K=300.0,
        friction=_friction_ps_inv_to_ase(2.0),
    )
    dyn.run(n_steps)
    relaxed = _copy_structure_with_path(structure, path=f"{structure.path}::relaxed")
    relaxed.positions = np.asarray(atoms.positions, dtype=np.float64)
    return relaxed


def equilibrate_structure_under_uma(
    *,
    structure: StructureData,
    chain_id: str,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    temperature_k: float = 300.0,
    timestep_fs: float = 0.05,
    friction_ps_inv: float = 1.0,
    steps: int = 200,
    fire_steps: int = 40,
    fmax_eva: float = 0.15,
    ca_network_sequential_k_eva2: float = 18.0,
    ca_network_midrange_k_eva2: float = 2.0,
    ca_network_contact_k_eva2: float = 0.75,
    ca_network_contact_cutoff_a: float = 10.0,
) -> StructureData:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    n_steps = max(0, int(steps))
    pre_md_steps = max(0, int(fire_steps))
    total_md_steps = n_steps + pre_md_steps
    if total_md_steps <= 0:
        return _copy_structure_with_path(structure, path=structure.path)
    key = (
        str(structure.path),
        str(chain_id),
        str(model_name),
        str(device),
        int(calculator_workers),
        float(temperature_k),
        float(timestep_fs),
        float(friction_ps_inv),
        int(steps),
        int(fire_steps),
        float(fmax_eva),
        float(ca_network_sequential_k_eva2),
        float(ca_network_midrange_k_eva2),
        float(ca_network_contact_k_eva2),
        float(ca_network_contact_cutoff_a),
    )
    cached = _EQUILIBRATED_STRUCTURE_CACHE.get(key)
    if cached is not None:
        return _copy_structure_with_path(cached, path=cached.path)

    try:
        ca_idx = protein_ca_indices(
            structure,
            chain_id=chain_id,
            exclude_positions=set(),
            stride=1,
        )
    except Exception:  # noqa: BLE001
        ca_idx = np.zeros((0,), dtype=np.int64)

    elastic_network = build_interpolated_ca_elastic_network(
        reactant=structure,
        product_aligned_positions=np.asarray(structure.positions, dtype=np.float64),
        reactant_ca_indices=np.asarray(ca_idx, dtype=np.int64),
        product_ca_indices=np.asarray(ca_idx, dtype=np.int64),
        sequential_k_eva2=float(ca_network_sequential_k_eva2),
        midrange_k_eva2=float(ca_network_midrange_k_eva2),
        contact_k_eva2=float(ca_network_contact_k_eva2),
        contact_cutoff_a=float(ca_network_contact_cutoff_a),
    )
    backbone_geometry = build_backbone_geometry_restraints(
        structure=structure,
        chain_id=chain_id,
    )
    elastic_pairs = np.concatenate(
        [
            np.asarray(elastic_network["pairs"], dtype=np.int64),
            np.asarray(backbone_geometry["pairs"], dtype=np.int64),
        ],
        axis=0,
    )
    elastic_targets = np.concatenate(
        [
            np.asarray(elastic_network["start_distances"], dtype=np.float64),
            np.asarray(backbone_geometry["target_distances"], dtype=np.float64),
        ],
        axis=0,
    )
    elastic_force_constants = np.concatenate(
        [
            np.asarray(elastic_network["force_constants"], dtype=np.float64),
            np.asarray(backbone_geometry["force_constants"], dtype=np.float64),
        ],
        axis=0,
    )

    atoms = structure.to_ase_atoms()
    bias_calc = FixedTargetBiasCalculator(
        _get_uma_calculator(model_name, device, workers=calculator_workers),
        steered_indices=np.zeros((0,), dtype=np.int64),
        steer_targets=np.zeros((0, 3), dtype=np.float64),
        steered_weights=np.zeros((0,), dtype=np.float64),
        global_indices=np.zeros((0,), dtype=np.int64),
        global_targets=np.zeros((0, 3), dtype=np.float64),
        local_indices=np.zeros((0,), dtype=np.int64),
        local_targets=np.zeros((0, 3), dtype=np.float64),
        anchor_indices=np.zeros((0,), dtype=np.int64),
        anchor_targets=np.zeros((0, 3), dtype=np.float64),
        k_window_eva2=0.0,
        k_global_eva2=0.0,
        k_local_eva2=0.0,
        k_anchor_eva2=0.0,
        elastic_pairs=elastic_pairs,
        elastic_target_distances=elastic_targets,
        elastic_force_constants=elastic_force_constants,
    )
    atoms.calc = bias_calc

    rng = np.random.RandomState(int(abs(hash((structure.path, chain_id, n_steps))) % (2**31 - 1)))
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature_k), rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = Langevin(
        atoms,
        float(timestep_fs) * units.fs,
        temperature_K=float(temperature_k),
        friction=_friction_ps_inv_to_ase(float(friction_ps_inv)),
    )
    dyn.run(int(total_md_steps))

    equilibrated = _copy_structure_with_path(structure, path=f"{structure.path}::equilibrated")
    equilibrated.positions = np.asarray(atoms.positions, dtype=np.float64)
    _EQUILIBRATED_STRUCTURE_CACHE[key] = _copy_structure_with_path(equilibrated, path=equilibrated.path)
    return equilibrated


def _build_smd_record_steps(total_images: int, steps_per_image: int) -> np.ndarray:
    n_images = max(1, int(total_images))
    image_steps = max(1, int(steps_per_image))
    total_md_steps = n_images * image_steps
    return np.arange(0, total_md_steps + 1, image_steps, dtype=np.int64)


def radius_of_gyration(
    positions: np.ndarray,
    *,
    indices: np.ndarray,
) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0.0
    coords = np.asarray(positions[idx], dtype=np.float64)
    center = np.mean(coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center[None, :]) ** 2, axis=1))))


def run_broad_uma_screen(
    *,
    reactant_complex_path: str | Path,
    reactant_structure: StructureData | None = None,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_ps_inv: float,
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
    prepare_hydrogens: bool = True,
    add_first_shell_waters: bool = True,
    preparation_ph: float = 7.4,
    max_first_shell_waters: int = 12,
    water_shell_distance_a: float = 2.8,
    water_clash_distance_a: float = 2.1,
    water_bridge_distance_min_a: float = 4.2,
    water_bridge_distance_max_a: float = 6.6,
    seed: int = 13,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    structure = reactant_structure
    if structure is None:
        if bool(prepare_hydrogens):
            structure = prepare_structure_for_uma(
                reactant_complex_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(preparation_ph),
                add_first_shell_waters=bool(add_first_shell_waters),
                max_first_shell_waters=int(max_first_shell_waters),
                water_shell_distance_a=float(water_shell_distance_a),
                water_clash_distance_a=float(water_clash_distance_a),
                water_bridge_distance_min_a=float(water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(water_bridge_distance_max_a),
            )
        else:
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
            friction=_friction_ps_inv_to_ase(float(friction_ps_inv)),
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
    path_lambdas: np.ndarray | None
    path_targets: np.ndarray | None
    steered_weights: np.ndarray
    global_indices: np.ndarray
    global_start_targets: np.ndarray
    global_end_targets: np.ndarray
    local_indices: np.ndarray
    local_start_targets: np.ndarray
    local_end_targets: np.ndarray
    anchor_indices: np.ndarray
    anchor_targets: np.ndarray
    k_steer_eva2: float
    k_global_eva2: float
    k_local_eva2: float
    k_anchor_eva2: float
    total_steps: int
    force_clip_eva: float | None = None
    elastic_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    elastic_start_distances: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    elastic_end_distances: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    elastic_force_constants: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    ligand_bond_start_distances: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_end_distances: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_start_lambdas: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_end_lambdas: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_start_force_constants: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    ligand_bond_end_force_constants: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    step_idx: int = 0
    cumulative_work: float = 0.0
    work_profile: list[dict[str, float]] = field(default_factory=list)

    def _lambda_for_step(self, step: int) -> float:
        if self.total_steps <= 1:
            return 1.0
        raw = float(step) / float(self.total_steps - 1)
        return 0.5 - 0.5 * math.cos(math.pi * raw)

    def current_targets(self, step: int | None = None) -> np.ndarray:
        use_step = self.step_idx if step is None else int(step)
        lam = self._lambda_for_step(use_step)
        if self.path_lambdas is not None and self.path_targets is not None:
            return _interpolate_path_targets(
                np.asarray(self.path_lambdas, dtype=np.float64),
                np.asarray(self.path_targets, dtype=np.float64),
                float(lam),
            )
        return (1.0 - lam) * self.start_targets + lam * self.end_targets

    def current_local_targets(self, step: int | None = None) -> np.ndarray:
        use_step = self.step_idx if step is None else int(step)
        lam = self._lambda_for_step(use_step)
        return (1.0 - lam) * self.local_start_targets + lam * self.local_end_targets

    def current_global_targets(self, step: int | None = None) -> np.ndarray:
        use_step = self.step_idx if step is None else int(step)
        lam = self._lambda_for_step(use_step)
        return (1.0 - lam) * self.global_start_targets + lam * self.global_end_targets

    def bias_energy_and_forces(self, positions: np.ndarray, step: int | None = None) -> tuple[float, np.ndarray]:
        use_step = self.step_idx if step is None else int(step)
        lam = float(self._lambda_for_step(use_step))
        targets = self.current_targets(use_step)
        forces = np.zeros_like(positions, dtype=np.float64)
        energy = 0.0

        disp = positions[self.steered_indices] - targets
        weights = np.asarray(self.steered_weights, dtype=np.float64).reshape(-1, 1)
        if weights.shape[0] != disp.shape[0]:
            weights = np.ones((disp.shape[0], 1), dtype=np.float64)
        energy += 0.5 * float(self.k_steer_eva2) * float(np.sum(weights * disp * disp))
        steer_force = -float(self.k_steer_eva2) * weights * disp
        if self.force_clip_eva is not None and self.force_clip_eva > 0:
            norms = np.linalg.norm(steer_force, axis=1, keepdims=True)
            mask = norms > float(self.force_clip_eva)
            steer_force[mask[:, 0]] *= float(self.force_clip_eva) / np.maximum(norms[mask[:, 0]], 1e-12)
        forces[self.steered_indices] += steer_force

        if len(self.local_indices) > 0 and self.k_local_eva2 > 0:
            local_targets = self.current_local_targets(use_step)
            ldisp = positions[self.local_indices] - local_targets
            energy += 0.5 * float(self.k_local_eva2) * float(np.sum(ldisp * ldisp))
            forces[self.local_indices] += -float(self.k_local_eva2) * ldisp

        if len(self.global_indices) > 0 and self.k_global_eva2 > 0:
            global_targets = self.current_global_targets(use_step)
            gdisp = positions[self.global_indices] - global_targets
            energy += 0.5 * float(self.k_global_eva2) * float(np.sum(gdisp * gdisp))
            forces[self.global_indices] += -float(self.k_global_eva2) * gdisp

        if len(self.anchor_indices) > 0 and self.k_anchor_eva2 > 0:
            adisp = positions[self.anchor_indices] - self.anchor_targets
            energy += 0.5 * float(self.k_anchor_eva2) * float(np.sum(adisp * adisp))
            forces[self.anchor_indices] += -float(self.k_anchor_eva2) * adisp

        if len(self.elastic_pairs) > 0:
            for pair, start_d, end_d, k_elastic in zip(
                self.elastic_pairs.tolist(),
                self.elastic_start_distances.tolist(),
                self.elastic_end_distances.tolist(),
                self.elastic_force_constants.tolist(),
                strict=False,
            ):
                i, j = int(pair[0]), int(pair[1])
                vec = np.asarray(positions[i] - positions[j], dtype=np.float64)
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-8:
                    continue
                target_d = (1.0 - lam) * float(start_d) + lam * float(end_d)
                delta = dist - target_d
                unit_vec = vec / dist
                energy += 0.5 * float(k_elastic) * float(delta * delta)
                pair_force = -float(k_elastic) * float(delta) * unit_vec
                forces[i] += pair_force
                forces[j] -= pair_force

        if len(self.ligand_bond_pairs) > 0:
            pair_energy, pair_forces = _scheduled_pair_energy_forces(
                positions,
                pairs=self.ligand_bond_pairs,
                start_distances=self.ligand_bond_start_distances,
                end_distances=self.ligand_bond_end_distances,
                force_constants=np.stack(
                    [
                        np.asarray(self.ligand_bond_start_force_constants, dtype=np.float64),
                        np.asarray(self.ligand_bond_end_force_constants, dtype=np.float64),
                    ],
                    axis=1,
                ),
                lam=lam,
                start_lambdas=self.ligand_bond_start_lambdas,
                end_lambdas=self.ligand_bond_end_lambdas,
            )
            energy += pair_energy
            forces += pair_forces

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
        steered_weights: np.ndarray,
        global_indices: np.ndarray,
        global_targets: np.ndarray,
        local_indices: np.ndarray,
        local_targets: np.ndarray,
        anchor_indices: np.ndarray,
        anchor_targets: np.ndarray,
        k_window_eva2: float,
        k_global_eva2: float,
        k_local_eva2: float,
        k_anchor_eva2: float,
        elastic_pairs: np.ndarray | None = None,
        elastic_target_distances: np.ndarray | None = None,
        elastic_force_constants: np.ndarray | None = None,
        ligand_bond_pairs: np.ndarray | None = None,
        ligand_bond_target_distances: np.ndarray | None = None,
        ligand_bond_force_constants: np.ndarray | None = None,
    ):
        self.base_calc = base_calc
        self.steered_indices = np.asarray(steered_indices, dtype=np.int64)
        self.steer_targets = np.asarray(steer_targets, dtype=np.float64)
        self.steered_weights = np.asarray(steered_weights, dtype=np.float64).reshape(-1)
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.global_targets = np.asarray(global_targets, dtype=np.float64)
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.local_targets = np.asarray(local_targets, dtype=np.float64)
        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
        self.anchor_targets = np.asarray(anchor_targets, dtype=np.float64)
        self.k_window_eva2 = float(k_window_eva2)
        self.k_global_eva2 = float(k_global_eva2)
        self.k_local_eva2 = float(k_local_eva2)
        self.k_anchor_eva2 = float(k_anchor_eva2)
        self.elastic_pairs = np.asarray(elastic_pairs if elastic_pairs is not None else np.zeros((0, 2), dtype=np.int64), dtype=np.int64)
        self.elastic_target_distances = np.asarray(elastic_target_distances if elastic_target_distances is not None else np.zeros((0,), dtype=np.float64), dtype=np.float64)
        self.elastic_force_constants = np.asarray(elastic_force_constants if elastic_force_constants is not None else np.zeros((0,), dtype=np.float64), dtype=np.float64)
        self.ligand_bond_pairs = np.asarray(ligand_bond_pairs if ligand_bond_pairs is not None else np.zeros((0, 2), dtype=np.int64), dtype=np.int64)
        self.ligand_bond_target_distances = np.asarray(ligand_bond_target_distances if ligand_bond_target_distances is not None else np.zeros((0,), dtype=np.float64), dtype=np.float64)
        self.ligand_bond_force_constants = np.asarray(ligand_bond_force_constants if ligand_bond_force_constants is not None else np.zeros((0,), dtype=np.float64), dtype=np.float64)
        self.results: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.base_calc.calculate(atoms, properties=["energy", "forces"], system_changes=system_changes)
        phys_e = float(self.base_calc.results["energy"])
        phys_f = np.asarray(self.base_calc.results["forces"], dtype=np.float64)
        positions = np.asarray(atoms.positions, dtype=np.float64)

        forces = np.zeros_like(positions, dtype=np.float64)
        energy = 0.0

        sdisp = positions[self.steered_indices] - self.steer_targets
        weights = self.steered_weights.reshape(-1, 1)
        if weights.shape[0] != sdisp.shape[0]:
            weights = np.ones((sdisp.shape[0], 1), dtype=np.float64)
        energy += 0.5 * self.k_window_eva2 * float(np.sum(weights * sdisp * sdisp))
        forces[self.steered_indices] += -self.k_window_eva2 * weights * sdisp

        if len(self.global_indices) > 0 and self.k_global_eva2 > 0:
            gdisp = positions[self.global_indices] - self.global_targets
            energy += 0.5 * self.k_global_eva2 * float(np.sum(gdisp * gdisp))
            forces[self.global_indices] += -self.k_global_eva2 * gdisp

        if len(self.local_indices) > 0 and self.k_local_eva2 > 0:
            ldisp = positions[self.local_indices] - self.local_targets
            energy += 0.5 * self.k_local_eva2 * float(np.sum(ldisp * ldisp))
            forces[self.local_indices] += -self.k_local_eva2 * ldisp

        if len(self.anchor_indices) > 0 and self.k_anchor_eva2 > 0:
            adisp = positions[self.anchor_indices] - self.anchor_targets
            energy += 0.5 * self.k_anchor_eva2 * float(np.sum(adisp * adisp))
            forces[self.anchor_indices] += -self.k_anchor_eva2 * adisp

        if len(self.elastic_pairs) > 0:
            for pair, target_d, k_elastic in zip(
                self.elastic_pairs.tolist(),
                self.elastic_target_distances.tolist(),
                self.elastic_force_constants.tolist(),
                strict=False,
            ):
                i, j = int(pair[0]), int(pair[1])
                vec = np.asarray(positions[i] - positions[j], dtype=np.float64)
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-8:
                    continue
                unit_vec = vec / dist
                delta = dist - float(target_d)
                energy += 0.5 * float(k_elastic) * float(delta * delta)
                pair_force = -float(k_elastic) * float(delta) * unit_vec
                forces[i] += pair_force
                forces[j] -= pair_force

        if len(self.ligand_bond_pairs) > 0:
            pair_energy, pair_forces = _scheduled_pair_energy_forces(
                positions,
                pairs=self.ligand_bond_pairs,
                start_distances=self.ligand_bond_target_distances,
                end_distances=self.ligand_bond_target_distances,
                force_constants=self.ligand_bond_force_constants,
                lam=0.0,
            )
            energy += pair_energy
            forces += pair_forces

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


def _collect_progress_terms(
    current_values: np.ndarray,
    *,
    start_values: np.ndarray,
    end_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[float], list[float]]:
    terms: list[float] = []
    term_weights: list[float] = []
    current_arr = np.asarray(current_values, dtype=np.float64).reshape(-1)
    start_arr = np.asarray(start_values, dtype=np.float64).reshape(-1)
    end_arr = np.asarray(end_values, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    for cur, start, end, weight in zip(current_arr.tolist(), start_arr.tolist(), end_arr.tolist(), weight_arr.tolist(), strict=False):
        delta = float(end) - float(start)
        if abs(delta) <= 1e-4 or float(weight) <= 1e-12:
            continue
        lam = (float(cur) - float(start)) / delta
        terms.append(float(min(1.0, max(0.0, lam))))
        term_weights.append(float(abs(delta) * weight))
    return terms, term_weights


def _project_reaction_progress_lambda(
    positions: np.ndarray,
    *,
    steered_indices: np.ndarray,
    start_targets: np.ndarray,
    end_targets: np.ndarray,
    bond_pairs: np.ndarray,
    bond_start_distances: np.ndarray,
    bond_end_distances: np.ndarray,
    bond_force_constants: np.ndarray,
) -> float:
    terms: list[float] = []
    term_weights: list[float] = []

    pair_idx = np.asarray(bond_pairs, dtype=np.int64)
    if pair_idx.size > 0:
        current = np.asarray(
            [float(np.linalg.norm(np.asarray(positions[int(i)] - positions[int(j)], dtype=np.float64))) for i, j in pair_idx.tolist()],
            dtype=np.float64,
        )
        t, w = _collect_progress_terms(
            current,
            start_values=np.asarray(bond_start_distances, dtype=np.float64),
            end_values=np.asarray(bond_end_distances, dtype=np.float64),
            weights=np.max(np.asarray(bond_force_constants, dtype=np.float64), axis=1)
            if np.asarray(bond_force_constants).ndim == 2
            else np.asarray(bond_force_constants, dtype=np.float64),
        )
        terms.extend(t)
        term_weights.extend(w)

    if term_weights:
        return float(np.average(np.asarray(terms, dtype=np.float64), weights=np.asarray(term_weights, dtype=np.float64)))
    return _project_progress_lambda(
        positions,
        steered_indices=steered_indices,
        start_targets=start_targets,
        end_targets=end_targets,
    )


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
    log_mean_exp = logsumexp(-beta * shifted, axis=0) - math.log(float(stacked.shape[0]))
    free = -log_mean_exp / beta
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
    reactant_structure: StructureData | None = None,
    product_structure: StructureData | None = None,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_ps_inv: float,
    images: int,
    steps_per_image: int,
    replicas: int,
    k_steer_eva2: float,
    k_global_eva2: float,
    k_local_eva2: float,
    k_anchor_eva2: float,
    ca_network_sequential_k_eva2: float = 6.0,
    ca_network_midrange_k_eva2: float = 1.25,
    ca_network_contact_k_eva2: float = 0.35,
    ca_network_contact_cutoff_a: float = 8.0,
    equilibrate_endpoint_steps: int = 160,
    equilibrate_endpoint_fire_steps: int = 40,
    equilibrate_endpoint_fmax_eva: float = 0.15,
    equilibrate_endpoint_ca_network_sequential_k_eva2: float = 18.0,
    equilibrate_endpoint_ca_network_midrange_k_eva2: float = 2.0,
    equilibrate_endpoint_ca_network_contact_k_eva2: float = 0.75,
    equilibrate_endpoint_ca_network_contact_cutoff_a: float = 10.0,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    anchor_stride: int = 12,
    force_clip_eva: float | None = 25.0,
    production_warmup_steps: int = 40,
    prepare_hydrogens: bool = True,
    add_first_shell_waters: bool = True,
    preparation_ph: float = 7.4,
    max_first_shell_waters: int = 12,
    water_shell_distance_a: float = 2.8,
    water_clash_distance_a: float = 2.1,
    water_bridge_distance_min_a: float = 4.2,
    water_bridge_distance_max_a: float = 6.6,
    max_reactive_bonds: int = 8,
    max_reactive_atoms: int = 12,
    max_reactive_fraction: float = 0.35,
    seed: int = 13,
    export_multimodel_pdb_path: str | Path | None = None,
    export_replica_index: int = 0,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    if reactant_structure is None:
        reactant = (
            prepare_structure_for_uma(
                reactant_complex_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(preparation_ph),
                add_first_shell_waters=bool(add_first_shell_waters),
                max_first_shell_waters=int(max_first_shell_waters),
                water_shell_distance_a=float(water_shell_distance_a),
                water_clash_distance_a=float(water_clash_distance_a),
                water_bridge_distance_min_a=float(water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(water_bridge_distance_max_a),
            )
            if bool(prepare_hydrogens)
            else load_structure(reactant_complex_path)
        )
    else:
        reactant = reactant_structure
    if product_structure is None:
        product = (
            prepare_structure_for_uma(
                product_complex_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(preparation_ph),
                add_first_shell_waters=bool(add_first_shell_waters),
                max_first_shell_waters=int(max_first_shell_waters),
                water_shell_distance_a=float(water_shell_distance_a),
                water_clash_distance_a=float(water_clash_distance_a),
                water_bridge_distance_min_a=float(water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(water_bridge_distance_max_a),
            )
            if bool(prepare_hydrogens)
            else load_structure(product_complex_path)
        )
    else:
        product = product_structure
    if int(equilibrate_endpoint_steps) > 0:
        reactant = equilibrate_structure_under_uma(
            structure=reactant,
            chain_id=protein_chain_id,
            model_name=model_name,
            device=device,
            calculator_workers=int(calculator_workers),
            temperature_k=float(temperature_k),
            timestep_fs=float(timestep_fs),
            friction_ps_inv=float(friction_ps_inv),
            steps=int(equilibrate_endpoint_steps),
            fire_steps=int(equilibrate_endpoint_fire_steps),
            fmax_eva=float(equilibrate_endpoint_fmax_eva),
            ca_network_sequential_k_eva2=float(equilibrate_endpoint_ca_network_sequential_k_eva2),
            ca_network_midrange_k_eva2=float(equilibrate_endpoint_ca_network_midrange_k_eva2),
            ca_network_contact_k_eva2=float(equilibrate_endpoint_ca_network_contact_k_eva2),
            ca_network_contact_cutoff_a=float(equilibrate_endpoint_ca_network_contact_cutoff_a),
        )
        product = equilibrate_structure_under_uma(
            structure=product,
            chain_id=protein_chain_id,
            model_name=model_name,
            device=device,
            calculator_workers=int(calculator_workers),
            temperature_k=float(temperature_k),
            timestep_fs=float(timestep_fs),
            friction_ps_inv=float(friction_ps_inv),
            steps=int(equilibrate_endpoint_steps),
            fire_steps=int(equilibrate_endpoint_fire_steps),
            fmax_eva=float(equilibrate_endpoint_fmax_eva),
            ca_network_sequential_k_eva2=float(equilibrate_endpoint_ca_network_sequential_k_eva2),
            ca_network_midrange_k_eva2=float(equilibrate_endpoint_ca_network_midrange_k_eva2),
            ca_network_contact_k_eva2=float(equilibrate_endpoint_ca_network_contact_k_eva2),
            ca_network_contact_cutoff_a=float(equilibrate_endpoint_ca_network_contact_cutoff_a),
        )
    protocol_bundle = analyze_endpoint_protocol(
        reactant=reactant,
        product=product,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pocket_positions=pocket_positions,
        max_reactive_bonds=int(max_reactive_bonds),
        max_reactive_atoms=int(max_reactive_atoms),
        max_reactive_fraction=float(max_reactive_fraction),
    )
    mapping = protocol_bundle["mapping"]
    pocket_react_idx, pocket_prod_idx = match_identity_indices(
        reactant,
        product,
        chain_id=protein_chain_id,
        residue_positions=pocket_positions,
    )
    local_mask = _pocket_guidance_mask(reactant, pocket_react_idx)
    local_react_idx = pocket_react_idx[local_mask]
    local_prod_idx = pocket_prod_idx[local_mask]
    if len(local_react_idx) == 0:
        local_react_idx = pocket_react_idx.copy()
        local_prod_idx = pocket_prod_idx.copy()

    try:
        global_react_idx, global_prod_idx = match_backbone_ca_indices(
            reactant,
            product,
            chain_id=protein_chain_id,
            exclude_positions=set(),
        )
    except Exception:  # noqa: BLE001
        try:
            global_react_idx, global_prod_idx = match_backbone_heavy_indices(
                reactant,
                product,
                chain_id=protein_chain_id,
                exclude_positions=set(),
            )
        except Exception:  # noqa: BLE001
            global_react_idx = local_react_idx.copy()
            global_prod_idx = local_prod_idx.copy()

    try:
        ca_react_idx, ca_prod_idx = match_backbone_ca_indices(
            reactant,
            product,
            chain_id=protein_chain_id,
            exclude_positions=set(),
        )
    except Exception:  # noqa: BLE001
        ca_react_idx = np.asarray([idx for idx in global_react_idx.tolist() if reactant.atom_names[int(idx)].strip() == "CA"], dtype=np.int64)
        ca_prod_idx = np.asarray([idx for idx in global_prod_idx.tolist() if product.atom_names[int(idx)].strip() == "CA"], dtype=np.int64)

    anchor_idx = protein_ca_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=max(1, int(anchor_stride)),
    )
    if len(anchor_idx) == 0:
        anchor_idx = protein_heavy_indices(
            reactant,
            chain_id=protein_chain_id,
            exclude_positions=set(int(x) for x in pocket_positions),
            stride=max(1, int(anchor_stride)),
        )
    if float(k_anchor_eva2) <= 0.0:
        anchor_idx = np.zeros((0,), dtype=np.int64)
    if len(anchor_idx) == 0 and float(k_anchor_eva2) > 0.0:
        anchor_idx = global_react_idx.copy()
    protein_nonpocket_backbone = protein_backbone_heavy_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=1,
    )
    if len(protein_nonpocket_backbone) == 0:
        protein_nonpocket_backbone = global_react_idx.copy()
    protein_heavy_all = protein_heavy_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(),
        stride=1,
    )
    if len(protein_heavy_all) == 0:
        protein_heavy_all = global_react_idx.copy()
    reference_protein_rg = radius_of_gyration(
        reactant.positions,
        indices=protein_heavy_all,
    )

    ligand_idx = np.asarray(protocol_bundle["ligand_indices"], dtype=np.int64)
    if len(ligand_idx) == 0:
        raise ValueError("no ligand atoms selected for steered UMA dynamics")
    rigid_path_lambdas = np.linspace(0.0, 1.0, max(2, int(images)) + 1, dtype=np.float64)
    guided_path_targets, ligand_path_mode, ligand_graph_model = build_guided_ligand_path_targets(
        reactant=reactant,
        ligand_indices=ligand_idx,
        mapping=mapping,
        lambdas=rigid_path_lambdas,
        graph_model=protocol_bundle["graph_model"],
    )
    ligand_restraints = protocol_bundle["ligand_restraints"]
    steering_mode = str(protocol_bundle["steering_mode"] or ligand_path_mode)
    protocol_meta = dict(protocol_bundle["protocol_meta"])
    effective_k_steer = float(k_steer_eva2)
    effective_k_global = float(k_global_eva2)
    effective_k_local = float(k_local_eva2)
    effective_k_anchor = float(k_anchor_eva2)
    if steering_mode != "reactive_center":
        effective_k_steer = max(effective_k_steer, 0.03)
        effective_k_global = max(effective_k_global, 0.08)
        effective_k_local = max(effective_k_local, 0.35)
        effective_k_anchor = max(effective_k_anchor, 0.02)
    elastic_network = build_interpolated_ca_elastic_network(
        reactant=reactant,
        product_aligned_positions=np.asarray(mapping["product_aligned_positions"], dtype=np.float64),
        reactant_ca_indices=ca_react_idx,
        product_ca_indices=ca_prod_idx,
        sequential_k_eva2=float(ca_network_sequential_k_eva2),
        midrange_k_eva2=float(ca_network_midrange_k_eva2),
        contact_k_eva2=float(ca_network_contact_k_eva2),
        contact_cutoff_a=float(ca_network_contact_cutoff_a),
    )
    backbone_geometry = build_backbone_geometry_restraints(
        structure=reactant,
        chain_id=protein_chain_id,
    )
    elastic_pairs = np.concatenate(
        [
            np.asarray(elastic_network["pairs"], dtype=np.int64),
            np.asarray(backbone_geometry["pairs"], dtype=np.int64),
        ],
        axis=0,
    )
    elastic_start_distances = np.concatenate(
        [
            np.asarray(elastic_network["start_distances"], dtype=np.float64),
            np.asarray(backbone_geometry["target_distances"], dtype=np.float64),
        ],
        axis=0,
    )
    elastic_end_distances = np.concatenate(
        [
            np.asarray(elastic_network["end_distances"], dtype=np.float64),
            np.asarray(backbone_geometry["target_distances"], dtype=np.float64),
        ],
        axis=0,
    )
    elastic_force_constants = np.concatenate(
        [
            np.asarray(elastic_network["force_constants"], dtype=np.float64),
            np.asarray(backbone_geometry["force_constants"], dtype=np.float64),
        ],
        axis=0,
    )

    total_images = max(2, int(images))
    total_md_steps = total_images * max(1, int(steps_per_image))
    record_steps = _build_smd_record_steps(total_images, int(steps_per_image))
    calc = _get_uma_calculator(model_name, device, workers=calculator_workers)

    production_path_lambdas = rigid_path_lambdas.copy()
    production_path_targets = guided_path_targets.copy()
    active_local_indices = np.arange(len(ligand_idx), dtype=np.int64)
    if steering_mode == "reactive_center":
        reactive_local = np.asarray(ligand_graph_model.get("reactive_atoms_local", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if reactive_local.size > 0:
            active_local_indices = reactive_local.copy()
    active_steered_indices = np.asarray(ligand_idx[active_local_indices], dtype=np.int64)
    active_path_targets = np.asarray(production_path_targets[:, active_local_indices, :], dtype=np.float64)
    active_steer_weights = np.asarray(ligand_restraints["steer_weights"][active_local_indices], dtype=np.float64)

    work_traces: list[np.ndarray] = []
    replica_summaries: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    path_snapshots: list[list[np.ndarray]] = []
    near_ts_rows: list[dict[str, Any]] = []
    export_frames: list[np.ndarray] | None = None
    export_meta: list[dict[str, float | int]] | None = None

    for replica_idx in range(int(replicas)):
        atoms = reactant.to_ase_atoms()
        protocol = SteeringProtocol(
            steered_indices=active_steered_indices.copy(),
            start_targets=active_path_targets[0].copy(),
            end_targets=active_path_targets[-1].copy(),
            path_lambdas=production_path_lambdas.copy(),
            path_targets=active_path_targets.copy(),
            steered_weights=active_steer_weights.copy(),
            global_indices=global_react_idx.copy(),
            global_start_targets=reactant.positions[global_react_idx].copy(),
            global_end_targets=mapping["product_aligned_positions"][global_prod_idx].copy(),
            local_indices=local_react_idx.copy(),
            local_start_targets=reactant.positions[local_react_idx].copy(),
            local_end_targets=mapping["product_aligned_positions"][local_prod_idx].copy(),
            anchor_indices=anchor_idx,
            anchor_targets=reactant.positions[anchor_idx].copy(),
            k_steer_eva2=float(effective_k_steer),
            k_global_eva2=float(effective_k_global),
            k_local_eva2=float(effective_k_local),
            k_anchor_eva2=float(effective_k_anchor),
            total_steps=int(total_md_steps + 1),
            force_clip_eva=force_clip_eva,
            elastic_pairs=elastic_pairs,
            elastic_start_distances=elastic_start_distances,
            elastic_end_distances=elastic_end_distances,
            elastic_force_constants=elastic_force_constants,
            ligand_bond_pairs=ligand_restraints["bond_pairs"],
            ligand_bond_start_distances=ligand_restraints["bond_start_distances"],
            ligand_bond_end_distances=ligand_restraints["bond_end_distances"],
            ligand_bond_start_lambdas=ligand_restraints["bond_start_lambdas"],
            ligand_bond_end_lambdas=ligand_restraints["bond_end_lambdas"],
            ligand_bond_start_force_constants=ligand_restraints["bond_start_force_constants"],
            ligand_bond_end_force_constants=ligand_restraints["bond_end_force_constants"],
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
            friction=_friction_ps_inv_to_ase(float(friction_ps_inv)),
        )

        pre_md_warmup_steps = max(0, int(production_warmup_steps))
        if pre_md_warmup_steps > 0:
            dyn.run(pre_md_warmup_steps)

        protocol.work_profile.append(
            {
                "step": 0.0,
                "lambda": 0.0,
                "work_increment_kcal_mol": 0.0,
                "cumulative_work_kcal_mol": 0.0,
            }
        )
        rep_snapshots = [atoms.positions[np.asarray(ligand_idx, dtype=np.int64)].copy()]
        rep_full_snapshots = [np.asarray(atoms.positions, dtype=np.float64).copy()]
        image_md_steps = [0]
        record_step_set = {int(x) for x in record_steps[1:].tolist()}
        for md_step in range(1, int(total_md_steps) + 1):
            dyn.run(1)
            protocol.advance_protocol(np.asarray(atoms.positions, dtype=np.float64))
            if md_step in record_step_set:
                rep_snapshots.append(atoms.positions[np.asarray(ligand_idx, dtype=np.int64)].copy())
                rep_full_snapshots.append(np.asarray(atoms.positions, dtype=np.float64).copy())
                image_md_steps.append(int(md_step))

        work_trace = np.asarray([row["cumulative_work_kcal_mol"] for row in protocol.work_profile], dtype=np.float64)
        work_traces.append(work_trace)
        path_snapshots.append(rep_snapshots)

        aligned_snapshots = [
            align_positions_on_reference(
                frame,
                reference_positions=reactant.positions,
                mobile_indices=global_react_idx,
                reference_indices=global_react_idx,
            )
            for frame in rep_full_snapshots
        ]
        react_lig = aligned_snapshots[-1][np.asarray(ligand_idx, dtype=np.int64)].copy()
        prod_lig = guided_path_targets[-1].copy()
        final_rmsd = rmsd(react_lig, prod_lig)
        final_pocket_rmsd = rmsd(aligned_snapshots[-1][pocket_react_idx], mapping["product_aligned_positions"][pocket_prod_idx])
        full_backbone_idx = protein_backbone_heavy_indices(
            reactant,
            chain_id=protein_chain_id,
            exclude_positions=set(),
            stride=1,
        )
        final_backbone_rmsd = rmsd(aligned_snapshots[-1][full_backbone_idx], reactant.positions[full_backbone_idx])
        final_global_rmsd = rmsd(aligned_snapshots[-1][global_react_idx], mapping["product_aligned_positions"][global_prod_idx])
        max_product_rmsd = max(
            rmsd(frame[np.asarray(ligand_idx, dtype=np.int64)], prod_lig)
            for frame in aligned_snapshots
        )
        max_pocket_rmsd = max(
            rmsd(frame[pocket_react_idx], mapping["product_aligned_positions"][pocket_prod_idx])
            for frame in aligned_snapshots
        )
        max_backbone_rmsd = max(
            rmsd(frame[full_backbone_idx], reactant.positions[full_backbone_idx])
            for frame in aligned_snapshots
        )
        max_global_rmsd = max(
            rmsd(frame[global_react_idx], mapping["product_aligned_positions"][global_prod_idx])
            for frame in aligned_snapshots
        )
        final_nonpocket_backbone_rmsd = rmsd(
            aligned_snapshots[-1][protein_nonpocket_backbone],
            reactant.positions[protein_nonpocket_backbone],
        )
        max_nonpocket_backbone_rmsd = max(
            rmsd(frame[protein_nonpocket_backbone], reactant.positions[protein_nonpocket_backbone])
            for frame in aligned_snapshots
        )
        target_elastic_distances = 0.5 * (elastic_network["start_distances"] + elastic_network["end_distances"])
        max_ca_network_rms = max(
            elastic_network_rms_deviation(
                frame,
                pairs=elastic_network["pairs"],
                target_distances=target_elastic_distances,
            )
            for frame in rep_full_snapshots
        )
        protein_nonpocket_heavy = protein_heavy_indices(
            reactant,
            chain_id=protein_chain_id,
            exclude_positions=set(int(x) for x in pocket_positions),
            stride=1,
        )
        max_close_contacts = max(
            count_close_contacts(
                frame,
                left_indices=ligand_idx,
                right_indices=protein_nonpocket_heavy,
                threshold_a=1.2,
            )
            for frame in rep_full_snapshots
        )
        max_excess_bonds = max(
            count_ligand_excess_bonds(
                symbols=ligand_restraints["ligand_symbols"],
                positions=np.asarray(frame[ligand_idx], dtype=np.float64),
                allowed_bonds=ligand_restraints["allowed_bonds"],
            )
            for frame in rep_full_snapshots
        )
        max_protein_rg_drift = max(
            abs(radius_of_gyration(frame, indices=protein_heavy_all) - reference_protein_rg)
            for frame in aligned_snapshots
        )
        summary = {
            "replica": int(replica_idx),
            "final_cumulative_work_kcal_mol": float(work_trace[-1]),
            "final_product_rmsd_a": float(final_rmsd),
            "final_pocket_rmsd_a": float(final_pocket_rmsd),
            "final_backbone_rmsd_a": float(final_backbone_rmsd),
            "final_global_backbone_rmsd_a": float(final_global_rmsd),
            "final_nonpocket_backbone_rmsd_a": float(final_nonpocket_backbone_rmsd),
            "max_product_rmsd_a": float(max_product_rmsd),
            "max_pocket_rmsd_a": float(max_pocket_rmsd),
            "max_backbone_rmsd_a": float(max_backbone_rmsd),
            "max_global_backbone_rmsd_a": float(max_global_rmsd),
            "max_nonpocket_backbone_rmsd_a": float(max_nonpocket_backbone_rmsd),
            "max_ca_network_rms_a": float(max_ca_network_rms),
            "max_protein_rg_drift_a": float(max_protein_rg_drift),
            "max_close_contacts": int(max_close_contacts),
            "max_excess_bond_count": int(max_excess_bonds),
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

        step_grid = image_md_steps[: len(rep_snapshots)]
        last_cum = 0.0
        for snap_idx, step_idx in enumerate(step_grid[: len(rep_snapshots)]):
            scheduled_lam = float(protocol._lambda_for_step(snap_idx))
            observed_lam = float(
                _project_reaction_progress_lambda(
                    aligned_snapshots[snap_idx],
                    steered_indices=np.asarray(ligand_idx, dtype=np.int64),
                    start_targets=protocol.start_targets,
                    end_targets=protocol.end_targets,
                    bond_pairs=protocol.ligand_bond_pairs,
                    bond_start_distances=protocol.ligand_bond_start_distances,
                    bond_end_distances=protocol.ligand_bond_end_distances,
                    bond_force_constants=np.maximum(
                        protocol.ligand_bond_start_force_constants,
                        protocol.ligand_bond_end_force_constants,
                    ),
                )
            )
            cum = float(work_trace[min(snap_idx, len(work_trace) - 1)])
            local_work = float(cum - last_cum)
            last_cum = cum
            snap_positions = aligned_snapshots[snap_idx][np.asarray(protocol.steered_indices, dtype=np.int64)]
            start_rmsd = rmsd(snap_positions, protocol.start_targets)
            end_rmsd = rmsd(snap_positions, protocol.end_targets)
            symmetry = max(0.0, 1.0 - abs(2.0 * float(observed_lam) - 1.0))
            score = float((1.0 + max(0.0, local_work)) * symmetry)
            near_ts_rows.append(
                {
                    "replica": int(replica_idx),
                    "image_index": int(snap_idx),
                    "step": int(step_idx),
                    "lambda": float(observed_lam),
                    "scheduled_lambda": float(scheduled_lam),
                    "local_work_kcal_mol": float(local_work),
                    "cumulative_work_kcal_mol": float(cum),
                    "start_ligand_rmsd_a": float(start_rmsd),
                    "end_ligand_rmsd_a": float(end_rmsd),
                    "near_ts_score": float(score),
                }
            )
        if export_multimodel_pdb_path is not None and int(replica_idx) == int(export_replica_index):
            export_frames = aligned_snapshots[: len(step_grid)]
            export_meta = []
            for snap_idx, step_idx in enumerate(step_grid[: len(export_frames)]):
                observed_lam = float(
                    _project_reaction_progress_lambda(
                        aligned_snapshots[snap_idx],
                        steered_indices=np.asarray(ligand_idx, dtype=np.int64),
                        start_targets=protocol.start_targets,
                        end_targets=protocol.end_targets,
                        bond_pairs=protocol.ligand_bond_pairs,
                        bond_start_distances=protocol.ligand_bond_start_distances,
                        bond_end_distances=protocol.ligand_bond_end_distances,
                        bond_force_constants=np.maximum(
                            protocol.ligand_bond_start_force_constants,
                            protocol.ligand_bond_end_force_constants,
                        ),
                    )
                )
                export_meta.append(
                    {
                        "replica": int(replica_idx),
                        "image_index": int(snap_idx),
                        "step": int(step_idx),
                        "lambda": float(observed_lam),
                        "scheduled_lambda": float(protocol._lambda_for_step(snap_idx)),
                        "cumulative_work_kcal_mol": float(work_trace[min(snap_idx, len(work_trace) - 1)]),
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
    exported_pdb = ""
    if export_multimodel_pdb_path is not None and export_frames:
        exported_pdb = str(
            write_multimodel_pdb(
                reactant,
                export_frames,
                export_multimodel_pdb_path,
                model_metadata=export_meta,
            )
        )

    return {
        "status": "ok",
        "mapping": {
            "shared_atom_count": int(mapping["shared_atom_count"]),
            "element_counts": mapping["element_counts"],
            "exact_name_matches": int(mapping.get("exact_name_matches", 0)),
            "ligand_path_mode": ligand_path_mode,
            "alignment_mode": str(mapping.get("alignment_mode", "unknown")),
            "alignment_atom_count": int(mapping.get("alignment_atom_count", 0)),
            "global_backbone_atom_count": int(len(global_react_idx)),
            "ca_network_atom_count": int(len(ca_react_idx)),
            "elastic_pair_count": int(len(elastic_network["pairs"])),
            "broken_bond_count": int(ligand_restraints["broken_bond_count"]),
            "formed_bond_count": int(ligand_restraints["formed_bond_count"]),
            "reactive_atom_count": int(ligand_restraints["reactive_atom_count"]),
            "steering_mode": steering_mode,
            "steering_confident": bool(ligand_restraints.get("steering_confident", False)),
            "quality_reason": str(ligand_restraints.get("quality_reason", "")),
            "protocol_mode": str(protocol_meta["protocol_mode"]),
            "protocol_reason": str(protocol_meta["protocol_reason"]),
            "reactive_barrier_valid": bool(protocol_meta["reactive_barrier_valid"]),
            "pmf_eligible": bool(protocol_meta["pmf_eligible"]),
        },
        "trajectory_protocol": {
            "mode": "continuous_uma_biased_langevin",
            "total_md_steps": int(total_md_steps),
            "record_steps": [int(x) for x in record_steps.tolist()],
            "production_warmup_steps": int(max(0, production_warmup_steps)),
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
        "trajectory_multimodel_pdb": exported_pdb,
    }


def run_path_umbrella_pmf(
    *,
    reactant_complex_path: str | Path,
    product_complex_path: str | Path,
    reactant_structure: StructureData | None = None,
    product_structure: StructureData | None = None,
    protein_chain_id: str,
    ligand_chain_id: str | None,
    pocket_positions: list[int],
    temperature_k: float,
    timestep_fs: float,
    friction_ps_inv: float,
    windows: int,
    steps_per_window: int,
    save_every: int,
    replicas: int,
    k_window_eva2: float,
    k_global_eva2: float,
    k_local_eva2: float,
    k_anchor_eva2: float,
    ca_network_sequential_k_eva2: float = 6.0,
    ca_network_midrange_k_eva2: float = 1.25,
    ca_network_contact_k_eva2: float = 0.35,
    ca_network_contact_cutoff_a: float = 8.0,
    equilibrate_endpoint_steps: int = 160,
    equilibrate_endpoint_fire_steps: int = 40,
    equilibrate_endpoint_fmax_eva: float = 0.15,
    equilibrate_endpoint_ca_network_sequential_k_eva2: float = 18.0,
    equilibrate_endpoint_ca_network_midrange_k_eva2: float = 2.0,
    equilibrate_endpoint_ca_network_contact_k_eva2: float = 0.75,
    equilibrate_endpoint_ca_network_contact_cutoff_a: float = 10.0,
    model_name: str,
    device: str,
    calculator_workers: int = 1,
    anchor_stride: int = 12,
    path_lambdas: list[float] | np.ndarray | None = None,
    path_targets: list[Any] | np.ndarray | None = None,
    prepare_hydrogens: bool = True,
    add_first_shell_waters: bool = True,
    preparation_ph: float = 7.4,
    max_first_shell_waters: int = 12,
    water_shell_distance_a: float = 2.8,
    water_clash_distance_a: float = 2.1,
    water_bridge_distance_min_a: float = 4.2,
    water_bridge_distance_max_a: float = 6.6,
    max_reactive_bonds: int = 8,
    max_reactive_atoms: int = 12,
    max_reactive_fraction: float = 0.35,
    seed: int = 13,
    window_relax_steps: int = 32,
    window_equilibrate_steps: int = 64,
) -> dict[str, Any]:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    if reactant_structure is None:
        reactant = (
            prepare_structure_for_uma(
                reactant_complex_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(preparation_ph),
                add_first_shell_waters=bool(add_first_shell_waters),
                max_first_shell_waters=int(max_first_shell_waters),
                water_shell_distance_a=float(water_shell_distance_a),
                water_clash_distance_a=float(water_clash_distance_a),
                water_bridge_distance_min_a=float(water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(water_bridge_distance_max_a),
            )
            if bool(prepare_hydrogens)
            else load_structure(reactant_complex_path)
        )
    else:
        reactant = reactant_structure
    if product_structure is None:
        product = (
            prepare_structure_for_uma(
                product_complex_path,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=ligand_chain_id,
                pocket_positions=pocket_positions,
                ph=float(preparation_ph),
                add_first_shell_waters=bool(add_first_shell_waters),
                max_first_shell_waters=int(max_first_shell_waters),
                water_shell_distance_a=float(water_shell_distance_a),
                water_clash_distance_a=float(water_clash_distance_a),
                water_bridge_distance_min_a=float(water_bridge_distance_min_a),
                water_bridge_distance_max_a=float(water_bridge_distance_max_a),
            )
            if bool(prepare_hydrogens)
            else load_structure(product_complex_path)
        )
    else:
        product = product_structure
    if int(equilibrate_endpoint_steps) > 0:
        reactant = equilibrate_structure_under_uma(
            structure=reactant,
            chain_id=protein_chain_id,
            model_name=model_name,
            device=device,
            calculator_workers=int(calculator_workers),
            temperature_k=float(temperature_k),
            timestep_fs=float(timestep_fs),
            friction_ps_inv=float(friction_ps_inv),
            steps=int(equilibrate_endpoint_steps),
            fire_steps=int(equilibrate_endpoint_fire_steps),
            fmax_eva=float(equilibrate_endpoint_fmax_eva),
            ca_network_sequential_k_eva2=float(equilibrate_endpoint_ca_network_sequential_k_eva2),
            ca_network_midrange_k_eva2=float(equilibrate_endpoint_ca_network_midrange_k_eva2),
            ca_network_contact_k_eva2=float(equilibrate_endpoint_ca_network_contact_k_eva2),
            ca_network_contact_cutoff_a=float(equilibrate_endpoint_ca_network_contact_cutoff_a),
        )
        product = equilibrate_structure_under_uma(
            structure=product,
            chain_id=protein_chain_id,
            model_name=model_name,
            device=device,
            calculator_workers=int(calculator_workers),
            temperature_k=float(temperature_k),
            timestep_fs=float(timestep_fs),
            friction_ps_inv=float(friction_ps_inv),
            steps=int(equilibrate_endpoint_steps),
            fire_steps=int(equilibrate_endpoint_fire_steps),
            fmax_eva=float(equilibrate_endpoint_fmax_eva),
            ca_network_sequential_k_eva2=float(equilibrate_endpoint_ca_network_sequential_k_eva2),
            ca_network_midrange_k_eva2=float(equilibrate_endpoint_ca_network_midrange_k_eva2),
            ca_network_contact_k_eva2=float(equilibrate_endpoint_ca_network_contact_k_eva2),
            ca_network_contact_cutoff_a=float(equilibrate_endpoint_ca_network_contact_cutoff_a),
        )
    protocol_bundle = analyze_endpoint_protocol(
        reactant=reactant,
        product=product,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pocket_positions=pocket_positions,
        max_reactive_bonds=int(max_reactive_bonds),
        max_reactive_atoms=int(max_reactive_atoms),
        max_reactive_fraction=float(max_reactive_fraction),
    )
    mapping = protocol_bundle["mapping"]
    pocket_react_idx, pocket_prod_idx = match_identity_indices(
        reactant,
        product,
        chain_id=protein_chain_id,
        residue_positions=pocket_positions,
    )
    local_mask = _pocket_guidance_mask(reactant, pocket_react_idx)
    local_react_idx = pocket_react_idx[local_mask]
    local_prod_idx = pocket_prod_idx[local_mask]
    if len(local_react_idx) == 0:
        local_react_idx = pocket_react_idx.copy()
        local_prod_idx = pocket_prod_idx.copy()
    try:
        global_react_idx, global_prod_idx = match_backbone_ca_indices(
            reactant,
            product,
            chain_id=protein_chain_id,
            exclude_positions=set(),
        )
    except Exception:  # noqa: BLE001
        try:
            global_react_idx, global_prod_idx = match_backbone_heavy_indices(
                reactant,
                product,
                chain_id=protein_chain_id,
                exclude_positions=set(),
            )
        except Exception:  # noqa: BLE001
            global_react_idx = local_react_idx.copy()
            global_prod_idx = local_prod_idx.copy()
    try:
        ca_react_idx, ca_prod_idx = match_backbone_ca_indices(
            reactant,
            product,
            chain_id=protein_chain_id,
            exclude_positions=set(),
        )
    except Exception:  # noqa: BLE001
        ca_react_idx = np.asarray([idx for idx in global_react_idx.tolist() if reactant.atom_names[int(idx)].strip() == "CA"], dtype=np.int64)
        ca_prod_idx = np.asarray([idx for idx in global_prod_idx.tolist() if product.atom_names[int(idx)].strip() == "CA"], dtype=np.int64)
    anchor_idx = protein_ca_indices(
        reactant,
        chain_id=protein_chain_id,
        exclude_positions=set(int(x) for x in pocket_positions),
        stride=max(1, int(anchor_stride)),
    )
    if len(anchor_idx) == 0:
        anchor_idx = protein_heavy_indices(
            reactant,
            chain_id=protein_chain_id,
            exclude_positions=set(int(x) for x in pocket_positions),
            stride=max(1, int(anchor_stride)),
        )
    if float(k_anchor_eva2) <= 0.0:
        anchor_idx = np.zeros((0,), dtype=np.int64)
    if len(anchor_idx) == 0 and float(k_anchor_eva2) > 0.0:
        anchor_idx = global_react_idx.copy()

    calc = _get_uma_calculator(model_name, device, workers=calculator_workers)
    steered_indices = np.asarray(protocol_bundle["ligand_indices"], dtype=np.int64)
    if len(steered_indices) == 0:
        raise ValueError("no ligand atoms selected for umbrella PMF")
    rigid_path_lambdas = np.linspace(0.0, 1.0, int(max(2, windows)), dtype=np.float64)
    guided_path_targets, _ligand_path_mode, ligand_graph_model = build_guided_ligand_path_targets(
        reactant=reactant,
        ligand_indices=steered_indices,
        mapping=mapping,
        lambdas=rigid_path_lambdas,
        graph_model=protocol_bundle["graph_model"],
    )
    ligand_restraints = protocol_bundle["ligand_restraints"]
    steering_mode = str(protocol_bundle["steering_mode"] or _ligand_path_mode)
    protocol_meta = dict(protocol_bundle["protocol_meta"])
    if not bool(protocol_meta["pmf_eligible"]):
        raise ValueError(
            "PMF requires reactive-center protocol mode; "
            f"got protocol_mode={protocol_meta['protocol_mode']} "
            f"reason={protocol_meta['protocol_reason']}"
        )
    effective_k_global = float(k_global_eva2)
    effective_k_local = float(k_local_eva2)
    effective_k_anchor = float(k_anchor_eva2)
    if steering_mode != "reactive_center":
        effective_k_global = max(effective_k_global, 0.08)
        effective_k_local = max(effective_k_local, 0.35)
        effective_k_anchor = max(effective_k_anchor, 0.02)
    elastic_network = build_interpolated_ca_elastic_network(
        reactant=reactant,
        product_aligned_positions=np.asarray(mapping["product_aligned_positions"], dtype=np.float64),
        reactant_ca_indices=ca_react_idx,
        product_ca_indices=ca_prod_idx,
        sequential_k_eva2=float(ca_network_sequential_k_eva2),
        midrange_k_eva2=float(ca_network_midrange_k_eva2),
        contact_k_eva2=float(ca_network_contact_k_eva2),
        contact_cutoff_a=float(ca_network_contact_cutoff_a),
    )
    backbone_geometry = build_backbone_geometry_restraints(
        structure=reactant,
        chain_id=protein_chain_id,
    )
    elastic_pairs = np.concatenate(
        [
            np.asarray(elastic_network["pairs"], dtype=np.int64),
            np.asarray(backbone_geometry["pairs"], dtype=np.int64),
        ],
        axis=0,
    )
    elastic_force_constants = np.concatenate(
        [
            np.asarray(elastic_network["force_constants"], dtype=np.float64),
            np.asarray(backbone_geometry["force_constants"], dtype=np.float64),
        ],
        axis=0,
    )
    start_targets = guided_path_targets[0].copy()
    end_targets = guided_path_targets[-1].copy()
    local_start_targets = reactant.positions[local_react_idx].copy()
    local_end_targets = mapping["product_aligned_positions"][local_prod_idx].copy()

    if path_lambdas is None or path_targets is None:
        path_lambdas_arr = rigid_path_lambdas
        path_targets_arr = guided_path_targets
    else:
        path_lambdas_arr = np.asarray(path_lambdas, dtype=np.float64)
        path_targets_arr = np.asarray(path_targets, dtype=np.float64)
        if path_targets_arr.ndim != 3:
            raise ValueError("path_targets must have shape [n_images, n_atoms, 3]")

    active_local_indices = np.arange(len(steered_indices), dtype=np.int64)
    if steering_mode == "reactive_center":
        reactive_local = np.asarray(ligand_graph_model.get("reactive_atoms_local", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if reactive_local.size > 0:
            active_local_indices = reactive_local.copy()
    active_steered_indices = np.asarray(steered_indices[active_local_indices], dtype=np.int64)
    active_path_targets_arr = np.asarray(path_targets_arr[:, active_local_indices, :], dtype=np.float64)
    active_start_targets = np.asarray(start_targets[active_local_indices], dtype=np.float64)
    active_end_targets = np.asarray(end_targets[active_local_indices], dtype=np.float64)
    active_steer_weights = np.asarray(ligand_restraints["steer_weights"][active_local_indices], dtype=np.float64)

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
        target_positions = _interpolate_path_targets(path_lambdas_arr, active_path_targets_arr, float(lam_center))
        global_target_positions = (1.0 - float(lam_center)) * reactant.positions[global_react_idx] + float(lam_center) * mapping["product_aligned_positions"][global_prod_idx]
        local_target_positions = (1.0 - float(lam_center)) * local_start_targets + float(lam_center) * local_end_targets
        elastic_target_distances = (
            (1.0 - float(lam_center)) * np.asarray(elastic_network["start_distances"], dtype=np.float64)
            + float(lam_center) * np.asarray(elastic_network["end_distances"], dtype=np.float64)
        )
        elastic_target_distances = np.concatenate(
            [
                elastic_target_distances,
                np.asarray(backbone_geometry["target_distances"], dtype=np.float64),
            ],
            axis=0,
        )
        for replica_idx in range(int(replicas)):
            atoms = reactant.to_ase_atoms()
            atoms.positions[steered_indices] = target_positions.copy()
            atoms.positions[global_react_idx] = global_target_positions.copy()
            atoms.positions[local_react_idx] = local_target_positions.copy()
            bias_calc = FixedTargetBiasCalculator(
                calc,
                steered_indices=active_steered_indices,
                steer_targets=target_positions,
                steered_weights=active_steer_weights,
                global_indices=global_react_idx,
                global_targets=global_target_positions,
                local_indices=local_react_idx,
                local_targets=local_target_positions,
                anchor_indices=anchor_idx,
                anchor_targets=reactant.positions[anchor_idx].copy(),
                k_window_eva2=float(k_window_eva2),
                k_global_eva2=float(effective_k_global),
                k_local_eva2=float(effective_k_local),
                k_anchor_eva2=float(effective_k_anchor),
                elastic_pairs=elastic_pairs,
                elastic_target_distances=elastic_target_distances,
                elastic_force_constants=elastic_force_constants,
                ligand_bond_pairs=ligand_restraints["bond_pairs"],
                ligand_bond_target_distances=(1.0 - float(lam_center)) * ligand_restraints["bond_start_distances"] + float(lam_center) * ligand_restraints["bond_end_distances"],
                ligand_bond_force_constants=(1.0 - float(lam_center)) * ligand_restraints["bond_start_force_constants"] + float(lam_center) * ligand_restraints["bond_end_force_constants"],
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
                friction=_friction_ps_inv_to_ase(float(friction_ps_inv)),
            )
            pre_window_md_steps = max(0, int(window_relax_steps)) + max(0, int(window_equilibrate_steps))
            if pre_window_md_steps > 0:
                dyn.run(pre_window_md_steps)

            rep_lambdas: list[float] = []
            for step_idx in range(0, int(steps_per_window) + 1):
                if step_idx > 0:
                    dyn.run(1)
                if step_idx % max(1, int(save_every)) != 0 and step_idx != int(steps_per_window):
                    continue
                positions = np.asarray(atoms.positions, dtype=np.float64)
                lam_obs = _project_reaction_progress_lambda(
                    positions,
                    steered_indices=active_steered_indices,
                    start_targets=active_start_targets,
                    end_targets=active_end_targets,
                    bond_pairs=ligand_restraints["bond_pairs"],
                    bond_start_distances=ligand_restraints["bond_start_distances"],
                    bond_end_distances=ligand_restraints["bond_end_distances"],
                    bond_force_constants=np.stack(
                        [
                            np.asarray(ligand_restraints["bond_start_force_constants"], dtype=np.float64),
                            np.asarray(ligand_restraints["bond_end_force_constants"], dtype=np.float64),
                        ],
                        axis=1,
                    )
                    if np.asarray(ligand_restraints["bond_start_force_constants"]).size > 0
                    else np.zeros((0, 2), dtype=np.float64),
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
        "protocol_mode": str(protocol_meta["protocol_mode"]),
        "protocol_reason": str(protocol_meta["protocol_reason"]),
        "pmf_eligible": bool(protocol_meta["pmf_eligible"]),
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
    rep_summaries = list(smd.get("replica_summaries", [])) if smd else []
    mapping = dict(smd.get("mapping", {})) if smd else {}
    def _rep_mean(key: str) -> float:
        vals = [float(row.get(key, 0.0)) for row in rep_summaries if key in row]
        return float(np.mean(vals)) if vals else 0.0
    dg_gate = float(broad.get("delta_g_gate_kcal_mol", 0.0))
    dg_bar_smd = float(smd.get("delta_g_smd_barrier_kcal_mol", 0.0)) if smd else 0.0
    dg_bar_pmf = float(pmf.get("delta_g_pmf_barrier_kcal_mol", 0.0)) if pmf else 0.0
    protocol_mode = str(mapping.get("protocol_mode", "none" if smd is None else "unknown"))
    protocol_reason = str(mapping.get("protocol_reason", ""))
    reactive_barrier_valid = bool(mapping.get("reactive_barrier_valid", smd is not None))
    pmf_eligible = bool(mapping.get("pmf_eligible", reactive_barrier_valid))
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
        "uma_cat_barrier_source": "pmf" if use_pmf else ("smd" if reactive_barrier_valid else "diagnostic_smd"),
        "uma_cat_delta_g_barrier_std_kcal_mol": float(barrier_std),
        "uma_cat_protocol_mode": protocol_mode,
        "uma_cat_protocol_reason": protocol_reason,
        "uma_cat_reactive_barrier_valid": bool(reactive_barrier_valid),
        "uma_cat_pmf_eligible": bool(pmf_eligible),
        "uma_cat_mean_work_kcal_mol": float(mean_work),
        "uma_cat_work_std_kcal_mol": float(smd.get("std_final_work_kcal_mol", 0.0)) if smd else 0.0,
        "uma_cat_pathway_score": float(pathway_score),
        "uma_cat_near_ts_count": int(near_ts_count),
        "uma_cat_final_product_rmsd_a": _rep_mean("final_product_rmsd_a"),
        "uma_cat_final_pocket_rmsd_a": _rep_mean("final_pocket_rmsd_a"),
        "uma_cat_final_backbone_rmsd_a": _rep_mean("final_backbone_rmsd_a"),
        "uma_cat_final_global_backbone_rmsd_a": _rep_mean("final_global_backbone_rmsd_a"),
        "uma_cat_max_product_rmsd_a": _rep_mean("max_product_rmsd_a"),
        "uma_cat_max_pocket_rmsd_a": _rep_mean("max_pocket_rmsd_a"),
        "uma_cat_max_backbone_rmsd_a": _rep_mean("max_backbone_rmsd_a"),
        "uma_cat_max_global_backbone_rmsd_a": _rep_mean("max_global_backbone_rmsd_a"),
        "uma_cat_max_ca_network_rms_a": _rep_mean("max_ca_network_rms_a"),
        "uma_cat_max_close_contacts": _rep_mean("max_close_contacts"),
        "uma_cat_max_excess_bond_count": _rep_mean("max_excess_bond_count"),
        "uma_cat_log10_rate_proxy": float(log_rate),
        "uma_cat_log10_rate_std": float(uncertainty),
        "uma_cat_std": float(uncertainty),
        "uma_cat_status": status,
    }


def assess_smd_quality(
    smd: dict[str, Any] | None,
    *,
    max_final_product_rmsd_a: float,
    max_max_product_rmsd_a: float,
    max_max_pocket_rmsd_a: float,
    max_max_backbone_rmsd_a: float,
    max_max_ca_network_rms_a: float = 0.75,
    max_max_close_contacts: int = 0,
    max_max_excess_bond_count: int = 0,
    max_max_nonpocket_backbone_rmsd_a: float = float("inf"),
    max_max_protein_rg_drift_a: float = float("inf"),
) -> dict[str, Any]:
    if not smd or smd.get("status") != "ok":
        return {
            "pass": False,
            "reasons": ["missing_or_failed_smd"],
            "protocol_mode": "missing",
            "protocol_reason": "missing_or_failed_smd",
            "reactive_barrier_valid": False,
            "pmf_eligible": False,
            "worst_final_product_rmsd_a": float("inf"),
            "worst_max_product_rmsd_a": float("inf"),
            "worst_max_pocket_rmsd_a": float("inf"),
            "worst_max_backbone_rmsd_a": float("inf"),
            "worst_max_ca_network_rms_a": float("inf"),
            "worst_max_close_contacts": float("inf"),
            "worst_max_excess_bond_count": float("inf"),
            "worst_max_nonpocket_backbone_rmsd_a": float("inf"),
            "worst_max_protein_rg_drift_a": float("inf"),
        }
    reps = list(smd.get("replica_summaries", []))
    if not reps:
        return {
            "pass": False,
            "reasons": ["missing_replica_summaries"],
            "protocol_mode": str((smd or {}).get("mapping", {}).get("protocol_mode", "unknown")),
            "protocol_reason": str((smd or {}).get("mapping", {}).get("protocol_reason", "")),
            "reactive_barrier_valid": bool((smd or {}).get("mapping", {}).get("reactive_barrier_valid", False)),
            "pmf_eligible": bool((smd or {}).get("mapping", {}).get("pmf_eligible", False)),
            "worst_final_product_rmsd_a": float("inf"),
            "worst_max_product_rmsd_a": float("inf"),
            "worst_max_pocket_rmsd_a": float("inf"),
            "worst_max_backbone_rmsd_a": float("inf"),
            "worst_max_ca_network_rms_a": float("inf"),
            "worst_max_close_contacts": float("inf"),
            "worst_max_excess_bond_count": float("inf"),
            "worst_max_nonpocket_backbone_rmsd_a": float("inf"),
            "worst_max_protein_rg_drift_a": float("inf"),
        }
    worst_final_product = max(float(row.get("final_product_rmsd_a", 0.0)) for row in reps)
    worst_max_product = max(float(row.get("max_product_rmsd_a", 0.0)) for row in reps)
    worst_max_pocket = max(float(row.get("max_pocket_rmsd_a", 0.0)) for row in reps)
    worst_max_backbone = max(float(row.get("max_backbone_rmsd_a", 0.0)) for row in reps)
    worst_ca_network_rms = max(float(row.get("max_ca_network_rms_a", 0.0)) for row in reps)
    worst_close_contacts = max(int(row.get("max_close_contacts", 0)) for row in reps)
    worst_excess_bonds = max(int(row.get("max_excess_bond_count", 0)) for row in reps)
    worst_nonpocket_backbone = max(float(row.get("max_nonpocket_backbone_rmsd_a", 0.0)) for row in reps)
    worst_protein_rg_drift = max(float(row.get("max_protein_rg_drift_a", 0.0)) for row in reps)
    mapping = dict(smd.get("mapping", {}))
    protocol_mode = str(mapping.get("protocol_mode", "unknown"))
    protocol_reason = str(mapping.get("protocol_reason", ""))
    reactive_barrier_valid = bool(mapping.get("reactive_barrier_valid", False))
    pmf_eligible = bool(mapping.get("pmf_eligible", False))
    reasons: list[str] = []
    if worst_final_product > float(max_final_product_rmsd_a):
        reasons.append("final_product_rmsd")
    if worst_max_product > float(max_max_product_rmsd_a):
        reasons.append("max_product_rmsd")
    if worst_max_pocket > float(max_max_pocket_rmsd_a):
        reasons.append("max_pocket_rmsd")
    if worst_max_backbone > float(max_max_backbone_rmsd_a):
        reasons.append("max_backbone_rmsd")
    if worst_ca_network_rms > float(max_max_ca_network_rms_a):
        reasons.append("max_ca_network_rms")
    if worst_close_contacts > int(max_max_close_contacts):
        reasons.append("max_close_contacts")
    if worst_excess_bonds > int(max_max_excess_bond_count):
        reasons.append("max_excess_bond_count")
    if worst_nonpocket_backbone > float(max_max_nonpocket_backbone_rmsd_a):
        reasons.append("max_nonpocket_backbone_rmsd")
    if worst_protein_rg_drift > float(max_max_protein_rg_drift_a):
        reasons.append("max_protein_rg_drift")
    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "protocol_mode": protocol_mode,
        "protocol_reason": protocol_reason,
        "reactive_barrier_valid": bool(reactive_barrier_valid),
        "pmf_eligible": bool(pmf_eligible),
        "worst_final_product_rmsd_a": float(worst_final_product),
        "worst_max_product_rmsd_a": float(worst_max_product),
        "worst_max_pocket_rmsd_a": float(worst_max_pocket),
        "worst_max_backbone_rmsd_a": float(worst_max_backbone),
        "worst_max_ca_network_rms_a": float(worst_ca_network_rms),
        "worst_max_close_contacts": int(worst_close_contacts),
        "worst_max_excess_bond_count": int(worst_excess_bonds),
        "worst_max_nonpocket_backbone_rmsd_a": float(worst_nonpocket_backbone),
        "worst_max_protein_rg_drift_a": float(worst_protein_rg_drift),
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
