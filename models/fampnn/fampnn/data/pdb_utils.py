from typing import Dict, List, Optional

import numpy as np
import torch
from einops import rearrange
from torchtyping import TensorType

import fampnn.data.protein as protein
from fampnn.data import residue_constants as rc
from fampnn.data.data import torch_rmsd_weighted
from fampnn.data.protein import PDB_CHAIN_IDS



def write_batched_to_pdb(
    aatype: TensorType["b n"],
    atom_positions: TensorType["b n 37 3"],
    atom_mask: TensorType["b n 37"],
    residue_index: TensorType["b n"],
    chain_index: TensorType["b n"],
    b_factors: Optional[TensorType["b n 37"]],
    filenames: List[str],
    mode: str = "aa",
    conect: bool = False):
    """
    Write batched protein structures to PDB files.

    If `b_factors` is not provided, the B-factors will be set to 0.

    mode: one of ["aa", "bb"]

    """
    B = aatype.shape[0]

    for i in range(B):
        write_to_pdb(
            aatype=aatype[i],
            atom_positions=atom_positions[i],
            atom_mask=atom_mask[i],
            residue_index=residue_index[i],
            chain_index=chain_index[i],
            b_factors=None if b_factors is None else b_factors[i],
            filename=filenames[i],
            mode=mode,
            conect=conect
        )



def write_to_pdb(
    aatype: TensorType["n"],
    atom_positions: TensorType["n 37 3"],
    atom_mask: TensorType["n 37"],
    residue_index: TensorType["n"],
    chain_index: TensorType["n"],
    b_factors: Optional[TensorType["n 37"]],
    filename: str,
    mode: str = "aa",
    conect: bool = False):
    """
    Write protein structures to a PDB file.

    If `b_factors` is not provided, the B-factors will be set to 0.

    mode: one of ["aa", "bb"]
    """
    if b_factors is None:
        b_factors = torch.zeros_like(atom_mask)

    if mode == "bb":
        bb_mask = torch.tensor(rc.restype_atom37_mask[rc.restype_order["G"]])
        atom_mask = atom_mask * bb_mask
    else:
        assert mode == "aa", f"Invalid pdb writing mode: {mode}"

    prot = protein.Protein(
        aatype=aatype.numpy(),
        atom_positions=atom_positions.numpy(),
        atom_mask=atom_mask.numpy(),
        residue_index=residue_index.numpy(),
        chain_index=chain_index.numpy(),
        chain_ids=[PDB_CHAIN_IDS[int(idx)] for idx in torch.sort(torch.unique(chain_index)).values.tolist()],
        b_factors=b_factors.numpy()
    )

    with open(filename, "w") as f:
        f.write(protein.to_pdb(prot, conect=conect))


def write_to_pdb_frames(
    aatype: TensorType["f n", int],
    atom_positions: TensorType["f n 37 3", float],
    atom_mask: TensorType["f n 37", bool],
    residue_index: TensorType["f n", int],
    chain_index: TensorType["f n", int],
    b_factors: Optional[TensorType["f n 37", float]],
    filename: str,
    mode: str = "aa",
    conect: bool = False,
    align_models_to_idx: Optional[int] = None
    ) -> Optional[TensorType["f", float]]:
    """
    Write protein structures to a PDB file with multiple frames.

    If `b_factors` is not provided, the B-factors will be set to 0.

    - align_models_to_idx: if not None, kabsch align all models to the model at the specified index. Returns RMSDs.
    """
    if b_factors is None:
        b_factors = torch.zeros_like(atom_mask)

    if mode == "bb":
        bb_mask = torch.tensor(rc.restype_atom37_mask[rc.restype_order["G"]])
        atom_mask = atom_mask * bb_mask
    else:
        assert mode == "aa", f"Invalid pdb writing mode: {mode}"

    if align_models_to_idx is not None:
        B, N, A, _ = atom_positions.shape
        atom_positions = rearrange(atom_positions, "b n a x -> b (n a) x")
        rmsds, (atom_positions, _) = torch_rmsd_weighted(a=atom_positions,
                                                         b=atom_positions[align_models_to_idx:align_models_to_idx + 1],
                                                         weights=rearrange(atom_mask, "b n a -> b (n a)"),
                                                         return_aligned=True)
        atom_positions = rearrange(atom_positions, "b (n a) x -> b n a x", n=N, a=A)

    prots = [
        protein.Protein(
            aatype=aatype[i].numpy(),
            atom_positions=atom_positions[i].numpy(),
            atom_mask=atom_mask[i].numpy(),
            residue_index=residue_index[i].numpy(),
            chain_index=chain_index[i].numpy(),
            b_factors=b_factors[i].numpy()
        )
        for i in range(aatype.shape[0])
    ]

    with open(filename, "w") as f:
        for i, prot in enumerate(prots):
            f.write(protein.to_pdb(prot, conect=conect, model_idx=i+1))

    if align_models_to_idx is not None:
        return rmsds
