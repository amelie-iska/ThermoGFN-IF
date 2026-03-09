import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType

import fampnn.data.residue_constants as rc


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_decoding_order(seq_mask: TensorType["b n", float]) -> TensorType["b n", int]:
    """
    Get the randomized order in which residues should be decoded, from 0 to N-1.
    Padding tokens are decoded last.
    """
    B, N = seq_mask.shape
    res_decoding_order = torch.where(seq_mask.bool(), torch.rand_like(seq_mask), 1.0e6)  # decode padded positions last
    res_decoding_order = res_decoding_order.argsort(dim=-1)

    return res_decoding_order.long()


def update_mlm_mask(mlm_mask: TensorType["b n", float],
                    aatype_decoding_order: TensorType["b n", int],
                    K: TensorType["b", int],
                    ) -> TensorType["b n", float]:
    """
    Update mlm_mask so that K total residues are unmasked.
    """
    mlm_mask_prev = mlm_mask.clone()

    ## using decoding order to decide positions to unmask
    residues_to_unmask = (~mlm_mask_prev.bool()) & (aatype_decoding_order < K[:,None])
    mlm_mask = residues_to_unmask + mlm_mask_prev
    return mlm_mask


def unmask(curr: TensorType["b n ..."],
           pred: TensorType["b n ..."],
           mlm_mask_prev: TensorType["b n", float],
           mlm_mask: TensorType["b n", float]) -> TensorType["b n ..."]:
    """
    Update curr based on pred and newly unmasked residues.
    """
    residues_to_unmask = mlm_mask - mlm_mask_prev
    assert residues_to_unmask.min() >= 0, "Trying to mask residues that are already unmasked"

    # Expand to data dims
    n_data_dims = len(curr.shape) - 2
    residues_to_unmask = residues_to_unmask.view(residues_to_unmask.shape + (1,) * n_data_dims)

    # Unmask residues
    curr = torch.where(residues_to_unmask.bool(), pred, curr)

    return curr


def get_timesteps_from_schedule(mode: str,
                                num_steps: int,
                                t_start: float,
                                t_end: float
                                ) -> TensorType["S+1", float]:
    """
    Get timesteps from timestep schedule for sampling. Essentially warps the time schedule to be non-linear.
    """
    S = num_steps
    timesteps = torch.linspace(t_start, t_end, S + 1)
    if mode == "linear":
        pass
    elif mode == "square":
        timesteps = timesteps ** 2
    elif mode == "cubic":
        timesteps = timesteps ** 3
    elif mode == "sqrt":
        timesteps = timesteps ** 0.5
    elif mode == "cbrt":
        timesteps = timesteps ** (1.0 / 3.0)
    elif mode == "cosine":
        timesteps = 1 - torch.cos(timesteps * np.pi / 2)
    else:
        raise NotImplementedError(f"timestep schedule mode {mode} not implemented")

    return timesteps


def parse_fixed_positions(fixed_pos_str: str,
                          chain_id_mapping: Dict[str, int],
                          residue_index: TensorType["n", int],
                          chain_index: TensorType["n", int]) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A1", "A10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        List[int]: List of absolute indices to set to 1 in the masks.
    """
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match = re.match(r"([A-Za-z])?(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1) if match.group(1) else " "
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

        if chain_letter not in chain_id_mapping:
            raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

        # For the given chain, create a mask for all residues in the desired range
        chain_i = chain_id_mapping[chain_letter]
        range_mask = (chain_index == chain_i) & (residue_index >= start_residue) & (residue_index <= end_residue)
        matching_indices = torch.where(range_mask)[0]

        # Check that each residue in the requested range; warn if not found
        found_residues = residue_index[matching_indices].tolist()
        found_residues_set = set(found_residues)

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues_set:
                print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

        # Extend our fixed indices with whatever we did find
        fixed_indices.extend(matching_indices.tolist())

    return list(set(fixed_indices))


def get_override_masks(batch: Dict[str, TensorType["b ..."]],
                       pdb_names: List[str],
                       batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
                       fixed_pos_df: pd.DataFrame,
                       mode: str,  # "seq_design" or "packing"
                       verbose: bool = False) -> Tuple[Optional[TensorType["b n", int]],
                                                       Optional[TensorType["b n", int]]]:
    """
    Given a batch of data and the fixed_pos_df, get masks for fixing positions in sequence and sidechain.
    If mode is "seq_design":
    - sequence mask defaults to all 0s (no starting sequence)
    - sidechain mask defaults to all 0s (no starting sidechains)

    If mode is "packing":
    - sequence mask defaults to all 1s (provide whole sequence from input PDB)
    - sidechain mask defaults to all 0s (no starting sidechains)
    """
    if mode == "seq_design":
        aatype_override_mask, scn_override_mask = torch.zeros_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])
    elif mode == "packing":
        aatype_override_mask, scn_override_mask = torch.ones_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])

    for i, pdb_name in enumerate(pdb_names):
        if pdb_name not in fixed_pos_df.index:
            if verbose:
                print(f"No fixed positions found for {pdb_name}")
            continue
        # Get fixed positions from df
        row = fixed_pos_df.loc[pdb_name]
        fixed_pos_seq, fixed_pos_scn = row["fixed_pos_seq"], row["fixed_pos_scn"]

        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        if fixed_pos_seq:
            # sequence override
            if mode == "packing":
                aatype_override_mask[i] = 0  # if fixed_pos_seq is provided in packing, we first start with no provided sequence, and then override

            if verbose:
                print(f"{pdb_name}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_positions(fixed_pos_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            aatype_override_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                fixed_seq_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if aatype_override_mask[i, j] else "-" for j in range(aatype_override_mask.shape[1])])
                print(f"Fixed sequence: {fixed_seq_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sequence positions specified.")

        if fixed_pos_scn:
            # sidechain override
            if verbose:
                print(f"{pdb_name}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_positions(fixed_pos_scn, chain_id_mapping, example["residue_index"], example["chain_index"])
            scn_override_mask[i, abs_fixed_pos_scn] = 1

            # print fixed sidechains
            if verbose:
                fixed_scn_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if scn_override_mask[i, j] else "-" for j in range(scn_override_mask.shape[1])])
                print(f"Fixed sidechains: {fixed_scn_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sidechain positions specified.")

        if verbose:
            print("\n========================\n")

    return aatype_override_mask, scn_override_mask
