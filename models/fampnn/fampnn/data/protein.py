# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type.
Adapted from original code by alexechu.
"""
import dataclasses
import io
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser, PDBList, PDBParser, Structure
from Bio.PDB.Atom import DisorderedAtom

from fampnn.data import residue_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    #alphabetic IDs of chains contained in the protein example
    chain_ids: list #[num_chains]

    #uncertainty in electron density, or predicted error in atom coordinates
    b_factors: np.ndarray

    # keep track of the offset due to detected insertion codes
    insertion_code_offsets: np.ndarray = None # [num_res]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


def read_pdb(pdb_file: Union[str, Structure.Structure], chain_ids_override: Optional[str] = None, max_conformers: int = 1) -> Tuple[Protein, Dict[str, int]]:
    """Takes a PDB string and constructs a Protein object.
    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.
      Ignores heteroatoms.

    Args:
      pdb_file: The path to the PDB file or a Biopython Structure object
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.
      max_conformers: Handle disordered atoms, max number of altlocs to store

    Returns:
      - A new `Protein` parsed from the pdb contents.
      - A dictionary that maps from chain letter to chain ID
    """
    if isinstance(pdb_file, str):
        if pdb_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure("none", pdb_file)
    else:
        structure = pdb_file

    models = list(structure.get_models())
    if len(models) != 1:
        print(f"Only single model PDBs are supported. Found {len(models)} models, but using first model by default.")

    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    residue_chain_ids = []
    chain_ids = []
    insertion_code_offsets = []

    for chain in model:
        insertion_code_offset = 0
        if (chain_ids_override is not None) and (chain.id not in chain_ids_override):
            continue

        if chain.id not in chain_ids:
            chain_ids.append(chain.id)

        for res in chain:
            if res.id[2] != " ":
                insertion_code_offset +=1
                print(f'Insertion code detected, increased residue index offset to {insertion_code_offset} for pdb {Path(pdb_file).name}')

            if res.id[0] != " ":
                if res.resname in residue_constants.ncaa_mapping.keys(): #allow all ncaas to get classified as 'X'
                    pass
                else:
                    continue

            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((max_conformers, residue_constants.atom_type_num, 3))
            mask = np.zeros((max_conformers, residue_constants.atom_type_num, ))
            res_b_factors = np.zeros((max_conformers, residue_constants.atom_type_num, ))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                if isinstance(atom, DisorderedAtom):
                    for conf_idx, atom_multi_conf in enumerate(atom.disordered_get_list()):
                        if conf_idx >= max_conformers:
                            # skip the rest of the conformers if exceeds max_conformers
                            break
                        pos[conf_idx, residue_constants.atom_order[atom.name]] = atom_multi_conf.coord
                        mask[conf_idx, residue_constants.atom_order[atom.name]] = 1.0
                        res_b_factors[conf_idx, residue_constants.atom_order[atom.name]] = atom_multi_conf.bfactor
                else:
                    pos[0, residue_constants.atom_order[atom.name]] = atom.coord
                    mask[0, residue_constants.atom_order[atom.name]] = 1.0
                    res_b_factors[0, residue_constants.atom_order[atom.name]] = atom.bfactor
            #* handle case where only a subset of atoms are disordered, e.g. sidechain atoms
            #* in this case, copy over the coordinates of the first altloc
            ai_exists = np.nonzero(mask[0])[0]
            for ci in range(1, max_conformers):
                for ai in ai_exists:
                    #* check if some atoms exist in conformer 'ci' and if the current atom doesn't exist
                    if np.sum(mask[ci]) != 0 and mask[ci, ai] == 0:
                        pos[ci, ai] = pos[0, ai] #* fill in coordinates from the first conformer
                        mask[ci, ai] = 1
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            # Squeeze out the conformer dimension if only one conformer
            pos = pos.squeeze(0) if max_conformers == 1 else pos
            mask = mask.squeeze(0) if max_conformers == 1 else mask
            res_b_factors = res_b_factors.squeeze(0) if max_conformers == 1 else res_b_factors

            # Append features;
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1] + insertion_code_offset)
            b_factors.append(res_b_factors)
            residue_chain_ids.append(chain.id)
            insertion_code_offsets.append(insertion_code_offset)

    # If specified, override chain ids with provided override, else use chain ids discovered from parsing PDB
    if chain_ids_override is not None:
        chain_ids = chain_ids_override

    # Chain IDs are usually characters so map these to ints.
    chain_id_mapping = {cid: n for n, cid in enumerate(chain_ids)}
    chain_ids_numeric = [n for _, n in chain_id_mapping.items()]
    chain_index = np.array([chain_id_mapping[cid] for cid in residue_chain_ids])
    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        chain_ids=chain_ids_numeric,
        b_factors=b_factors,
        insertion_code_offsets=insertion_code_offsets,
    ), chain_id_mapping


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def are_atoms_bonded(res3name, atom1_name, atom2_name):
    lookup_table = residue_constants.standard_residue_bonds
    for bond in lookup_table[res3name]:
        if bond.atom1_name == atom1_name and bond.atom2_name == atom2_name:
            return True
        elif bond.atom1_name == atom2_name and bond.atom2_name == atom1_name:
            return True
    return False


def to_pdb(prot: Protein, conect=False, model_idx: int = 1) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model_idx}")
    atom_index = 1
    last_chain_index = chain_index[0]
    conect_lines = []
    c_atom_idx, n_atom_idx = None, None
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        atoms_appended_for_res = []
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.3f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)

            for prev_atom_idx, prev_atom in atoms_appended_for_res:
                if are_atoms_bonded(res_name_3, atom_name, prev_atom):
                    conect_line = f"CONECT{prev_atom_idx:5d}{atom_index:5d}\n"
                    conect_lines.append(conect_line)
            atoms_appended_for_res.append((atom_index, atom_name))
            if atom_name == "N":
                n_atom_idx = atom_index
            if atom_name == "C":
                c_atom_idx = atom_index

            atom_index += 1

        if i > 0 and (n_atom_idx is not None) and (prev_c_atom_idx is not None):
            conect_line = f"CONECT{prev_c_atom_idx:5d}{n_atom_idx:5d}\n"
            conect_lines.append(conect_line)
        prev_c_atom_idx = c_atom_idx

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    # pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    pdb_str = "\n".join(pdb_lines) + "\n"  # Add terminating newline.
    if conect:
        conect_str = "".join(conect_lines) + "\n"
        return pdb_str + conect_str
    return pdb_str


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return Protein(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )
