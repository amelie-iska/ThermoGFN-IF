"""Prediction postprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

from boltz.data.types import Record, StructureV2


def _load_template_mol(record: Record, template_dir: Path) -> Optional[Chem.Mol]:
    """Load the first ligand template matching this record."""
    template_paths = list(template_dir.glob(f"{record.id}_*.npz"))
    if not template_paths:
        return None
    data = np.load(template_paths[0])
    atoms = data["atoms"]
    bonds = data["bonds"]
    mol = Chem.RWMol()
    for idx, atom_row in enumerate(atoms):
        symbol = atom_row["name"].decode()[:1]
        atom = Chem.Atom(symbol)
        mol_idx = mol.AddAtom(atom)
        assert mol_idx == idx
    for bond in bonds:
        a1, a2, btype = int(bond["atom_1"]), int(bond["atom_2"]), int(bond["type"])
        # Map Boltz bond types: 0 unknown, 1 SINGLE, 2 DOUBLE, 3 TRIPLE, 4 AROMATIC
        if btype == 2:
            bt = Chem.BondType.DOUBLE
        elif btype == 3:
            bt = Chem.BondType.TRIPLE
        elif btype == 4:
            bt = Chem.BondType.AROMATIC
        else:
            bt = Chem.BondType.SINGLE
        mol.AddBond(a1, a2, bt)
    return mol.GetMol()


def _assign_bonds_to_pred(pred: StructureV2, template: Chem.Mol) -> StructureV2:
    """Assign bond orders from template to predicted ligand atoms in-place."""
    atoms = pred.atoms
    bonds = pred.bonds
    if atoms.shape[0] != template.GetNumAtoms():
        return pred

    rw = Chem.RWMol(template)
    # The StructureV2 already has bond pairs; only update type ids.
    for i, bond_row in enumerate(bonds):
        a1, a2 = int(bond_row["atom_1"]), int(bond_row["atom_2"])
        bond = rw.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            btype = 1
        elif bt == Chem.BondType.DOUBLE:
            btype = 2
        elif bt == Chem.BondType.TRIPLE:
            btype = 3
        elif bt == Chem.BondType.AROMATIC:
            btype = 4
        else:
            btype = bond_row["type"]
        bonds[i]["type"] = btype
    return pred


def restore_template_connectivity(structure, record: Record):
    """Restore ligand bond orders from the ligand template if available."""
    if not isinstance(structure, StructureV2):
        return structure
    template_dir = Path(record.template_dir) if hasattr(record, "template_dir") else None
    if template_dir is None or not template_dir.exists():
        return structure
    template_mol = _load_template_mol(record, template_dir)
    if template_mol is None:
        return structure
    try:
        structure = _assign_bonds_to_pred(structure, template_mol)
    except Exception:
        return structure
    return structure
