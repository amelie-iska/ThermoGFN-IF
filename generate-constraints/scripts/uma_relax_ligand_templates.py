#!/usr/bin/env python
"""
Generate UMA-relaxed ligand conformers and inject them as Boltz-TS templates.

Usage:
  python scripts/uma_relax_ligand_templates.py \
      --yaml models/boltz-ts/examples/product_state_template.yaml \
      --out-yaml product_with_templates.yaml \
      --out-dir lig_sdf \
      --device cuda

What it does:
- Parses a Boltz-TS YAML (expects a ligand entry under `sequences` with a `smiles` field).
- Attempts RDKit ETKDG embedding; on failure tries Open Babel.
- Refines coordinates with UMA forces (gradient steps).
- Writes an SDF per ligand and updates/creates a `templates` entry pointing to that SDF.
"""

import argparse
import os
import sys
import subprocess
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
MDGEN_ROOT = os.path.join(ROOT, "models", "mdgen")
if MDGEN_ROOT not in sys.path:
    sys.path.append(MDGEN_ROOT)

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False

try:
    from mdgen.uma_energy import UMAEnergy, ELEMENT_NUMBERS  # type: ignore
except Exception:
    UMAEnergy = None  # type: ignore
    # Minimal periodic table fallback for obabel parsing
    ELEMENT_NUMBERS = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "BR": 35,
        "Br": 35,
        "I": 53,
        "Na": 11,
        "K": 19,
        "Ca": 20,
        "Fe": 26,
        "Cu": 29,
        "Zn": 30,
        "Mg": 12,
    }

# Reverse map for writing element symbols when we only have atomic numbers.
ELEMENT_SYMBOLS = {v: k for k, v in ELEMENT_NUMBERS.items()}


def rdkit_embed(smiles: str, minimize: bool = True, attempts: int = 64):
    if not RDKit_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    has_wildcards = mol.HasSubstructMatch(Chem.MolFromSmarts("[*]"))
    if has_wildcards:
        # RDKit embeddings with wildcard attachment points are brittle; let obabel handle instead.
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Continue with partially sanitized mol; embedding may still work.
        pass
    mol = Chem.AddHs(mol, addCoords=False)
    params = AllChem.ETKDGv3()
    if hasattr(params, "numThreads"):
        params.numThreads = 0
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 2000
    if hasattr(params, "useRandomCoords") and has_wildcards:
        params.useRandomCoords = True
    if hasattr(params, "enforceChirality") and has_wildcards:
        params.enforceChirality = False
    cid = -1
    for _ in range(attempts):
        try:
            cid = AllChem.EmbedMolecule(mol, params=params)
        except Exception:
            cid = -1
        if cid != -1:
            break
        if hasattr(params, "randomSeed"):
            params.randomSeed = np.random.randint(0, 1_000_000)
    if cid == -1 or has_wildcards:
        # Try distance geometry with bounds adjustments
        try:
            if hasattr(params, "useRandomCoords"):
                params.useRandomCoords = True
            if hasattr(params, "randomSeed"):
                params.randomSeed = np.random.randint(0, 1_000_000)
            cid = AllChem.EmbedMolecule(mol, params=params)
        except Exception:
            cid = -1
    if cid == -1:
        return None
    if minimize:
        # UFF can't type wildcard atoms ("*"), so skip minimization when they are present to avoid warnings.
        has_wildcards = mol.HasSubstructMatch(Chem.MolFromSmarts("[*]"))
        if not has_wildcards:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=800)
            except Exception:
                pass
    conf = mol.GetConformer(cid)
    coords = np.array(conf.GetPositions(), dtype=np.float32)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    return mol, coords, z


def obabel_embed(smiles: str, tries: int = 3, nconf: int = 1):
    """Use obabel to generate 3D coordinates; returns coords, atomic_numbers, raw_lines."""
    for _ in range(tries):
        try:
            result = subprocess.run(
                [
                    "obabel",
                    f"-:{smiles}",
                    "-osdf",
                    "--gen3d",
                    "--conformer",
                    "--weighted",
                    "--fastest",
                    "--nconf",
                    str(nconf),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue
        sdf = result.stdout.splitlines()
        if len(sdf) < 4:
            continue
        counts = sdf[3]
        try:
            num_atoms = int(counts[:3])
        except Exception:
            parts = counts.split()
            if not parts:
                continue
            num_atoms = int(parts[0])
        atom_lines = sdf[4 : 4 + num_atoms]
        coords = []
        zs = []
        ok = True
        for line in atom_lines:
            toks = line.split()
            if len(toks) < 4:
                ok = False
                break
            x, y, zc = map(float, toks[:3])
            elem = toks[3].strip()
            zs.append(ELEMENT_NUMBERS.get(elem, 6))
            coords.append([x, y, zc])
        if ok:
            return np.array(coords, dtype=np.float32), np.array(zs, dtype=np.int64), sdf
    return None


def write_sdf_from_lines(lines: List[str], coords: np.ndarray, out_path: str):
    """Replace atom block coordinates in an SDF line list and write to disk."""
    lines = lines.copy()
    try:
        num_atoms = int(lines[3][:3])
    except Exception:
        parts = lines[3].split()
        num_atoms = int(parts[0])
    atom_start = 4
    for i in range(num_atoms):
        toks = lines[atom_start + i].split()
        tail = " ".join(toks[3:]) if len(toks) > 3 else ""
        x, y, zc = coords[i]
        lines[atom_start + i] = f"{x:>10.4f}{y:>10.4f}{zc:>10.4f} {tail}\n"
    with open(out_path, "w") as f:
        for ln in lines:
            if not ln.endswith("\n"):
                ln = ln + "\n"
            f.write(ln)


def refine_with_uma(pos: torch.Tensor, z: torch.Tensor, uma: UMAEnergy, steps: int, lr: float, clip: float):
    pos = pos.clone()
    energy_val = None
    for _ in range(steps):
        energy, forces = uma.energy_forces(pos, z)
        pos = pos - lr * forces
        if clip is not None:
            pos = pos.clamp(min=-clip, max=clip)
        energy_val = energy.item()
    return pos, energy_val if energy_val is not None else float("nan")


def coords_are_bad(pos: torch.Tensor, clip: float):
    if not torch.isfinite(pos).all():
        return True
    max_abs = pos.abs().max().item()
    if clip is not None and max_abs >= clip * 0.95:
        return True
    return False


def upsert_template(templates: List[Dict], chain_id: str, sdf_path: str, threshold: float = 0.01):
    updated = False
    for t in templates:
        if t.get("chain_id") == chain_id:
            t.update({"sdf": sdf_path, "atom_map": "identical", "force": True, "threshold": threshold})
            updated = True
            break
    if not updated:
        templates.append(
            {"sdf": sdf_path, "chain_id": chain_id, "atom_map": "identical", "force": True, "threshold": threshold}
        )
    return templates


def main():
    ap = argparse.ArgumentParser(description="UMA-relax ligands and add Boltz-TS templates")
    ap.add_argument("--yaml", required=True, help="Input Boltz-TS YAML")
    ap.add_argument("--out-yaml", required=True, help="Output YAML with template paths injected")
    ap.add_argument("--out-dir", required=True, help="Directory to write SDFs")
    ap.add_argument("--skip-uma", action="store_true", help="Skip UMA loading/refinement (embed-only fallback)")
    ap.add_argument("--obabel-tries", type=int, default=8, help="Attempts for obabel embedding")
    ap.add_argument(
        "--obabel-nconf",
        type=int,
        default=1,
        help="Number of conformers to sample per obabel attempt (keep at 1 to avoid multiple writes)",
    )
    ap.add_argument("--rdkit-attempts", type=int, default=64, help="Number of RDKit embedding attempts before fallback")
    ap.add_argument("--uma_model_name", default="uma-s-1p1")
    ap.add_argument("--uma_task", default="omol")
    ap.add_argument("--uma_cutoff", type=float, default=4.0)
    ap.add_argument("--uma_max_neighbors", type=int, default=32)
    ap.add_argument("--uma_pos_clip", type=float, default=16.0)
    ap.add_argument("--uma_radius_version", type=int, default=1)
    ap.add_argument("--uma_enforce_max_neighbors", action="store_true")
    ap.add_argument("--uma_always_use_pbc", action="store_true")
    ap.add_argument("--refine_steps", type=int, default=12)
    ap.add_argument("--refine_lr", type=float, default=0.01)
    ap.add_argument("--refine_clip", type=float, default=8.0)
    ap.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto (prefers cuda when available)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.yaml, "r") as f:
        doc = yaml.safe_load(f)

    sequences = doc.get("sequences", [])
    templates = doc.get("templates", []) or []

    resolved_device = args.device
    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    uma = None
    if not args.skip_uma:
        if UMAEnergy is None:
            tqdm.write("[error] UMAEnergy unavailable; run with --skip-uma or install mdgen deps")
            sys.exit(1)
        try:
            uma = UMAEnergy(
                model_name=args.uma_model_name,
                task=args.uma_task,
                device=resolved_device,
                cutoff=args.uma_cutoff,
                max_neighbors=args.uma_max_neighbors,
                pos_clip=args.uma_pos_clip,
                radius_version=args.uma_radius_version,
                enforce_max_neighbors=args.uma_enforce_max_neighbors,
                always_use_pbc=args.uma_always_use_pbc,
            )
            tqdm.write(f"[uma] loaded model on device={resolved_device}")
        except Exception as exc:
            tqdm.write(f"[warn] UMA load failed ({exc}); continuing without refinement")
            uma = None
    else:
        tqdm.write("[info] UMA skipped; using embed-only coordinates")

    base_name = os.path.splitext(os.path.basename(args.yaml))[0]

    for entry in tqdm(sequences, desc="ligands"):
        if "ligand" not in entry:
            continue
        lig = entry["ligand"]
        chain_id = lig.get("id", "L")
        smiles = lig.get("smiles")
        if not smiles:
            tqdm.write(f"[skip] ligand {chain_id} has no SMILES")
            continue

        rd = rdkit_embed(smiles, attempts=args.rdkit_attempts)
        raw_sdf_lines = None
        if rd is None:
            ob = obabel_embed(smiles, tries=args.obabel_tries, nconf=args.obabel_nconf)
            if ob is None:
                tqdm.write(f"[skip] failed to generate 3D for ligand {chain_id} with RDKit and obabel; skipping")
                continue
            else:
                coords, z, raw_sdf_lines = ob
        else:
            mol, coords, z = rd

        pos_embed = torch.tensor(coords, device=resolved_device, dtype=torch.float32)
        pos_refined = pos_embed
        e_val = float("nan")
        if uma is not None:
            z_t = torch.tensor(z, device=resolved_device, dtype=torch.long)
            pos_refined, e_val = refine_with_uma(
                pos_refined, z_t, uma, steps=args.refine_steps, lr=args.refine_lr, clip=args.refine_clip
            )
            if coords_are_bad(pos_refined, args.refine_clip):
                tqdm.write(
                    f"[warn] UMA refinement diverged for ligand {chain_id} (max_abs "
                    f"{pos_refined.abs().max().item():.2f}); reverting to embedded coords"
                )
                pos_refined = pos_embed
                e_val = float("nan")

        sdf_path = os.path.join(args.out_dir, f"{base_name}_{chain_id}.sdf")
        os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
        # Debug which path is writing
        branch = "rdkit" if rd is not None else ("obabel" if raw_sdf_lines is not None else "minimal")
        tqdm.write(f"[write] branch={branch} ligand={chain_id} -> {sdf_path}")

        try:
            if rd is not None:
                # Assign refined coords to RDKit mol for writing SDF
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i in range(mol.GetNumAtoms()):
                    x, y, zc = pos_refined[i].cpu().numpy().tolist()
                    conf.SetAtomPosition(i, (float(x), float(y), float(zc)))
                mol.RemoveAllConformers()
                mol.AddConformer(conf, assignId=True)
                w = Chem.SDWriter(sdf_path)
                w.write(mol)
                w.close()
            elif raw_sdf_lines is not None:
                write_sdf_from_lines(raw_sdf_lines, pos_refined.cpu().numpy(), sdf_path)
            else:
                # Minimal SDF writer from coords and atomic numbers
                with open(sdf_path, "w") as f:
                    f.write("\n" * 3)
                    n = len(z)
                    f.write(f"{n:>3d}{0:>3d}  0  0  0  0            999 V2000\n")
                    for coord, atomic_num in zip(pos_refined.cpu().numpy(), z):
                        x, y, zc = coord
                        elem = ELEMENT_SYMBOLS.get(int(atomic_num), "C")
                        f.write(
                            f"{x:>10.4f}{y:>10.4f}{zc:>10.4f} {elem:>3s}  0  0  0  0  0  0  0  0  0  0\n"
                        )
                    f.write("M  END\n")
            if not os.path.exists(sdf_path):
                tqdm.write(f"[warn] expected SDF not found after write: {sdf_path}")
            else:
                tqdm.write(f"[ok] SDF written: {sdf_path}")
        except Exception as exc:
            tqdm.write(f"[error] writing SDF for ligand {chain_id}: {exc}")
        tqdm.write(f"[saved] ligand {chain_id} -> {sdf_path} (UMA E={e_val:.3f})")

        templates = upsert_template(templates, chain_id=chain_id, sdf_path=os.path.abspath(sdf_path))

    # Prepare pocket contacts with flow style ([A, 12]) for readability/consistency.
    class FlowList(list):
        pass

    class MyDumper(yaml.SafeDumper):
        pass

    def flow_seq_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    MyDumper.add_representer(FlowList, flow_seq_representer)

    for constraint in doc.get("constraints", []):
        if "pocket" in constraint:
            contacts = constraint["pocket"].get("contacts", [])
            constraint["pocket"]["contacts"] = [FlowList(contact) for contact in contacts]

    doc["templates"] = templates
    with open(args.out_yaml, "w") as f:
        yaml.dump(doc, f, sort_keys=False, Dumper=MyDumper)
    print(f"[done] wrote {args.out_yaml}")


if __name__ == "__main__":
    main()
