import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from scipy.stats import pearsonr, spearmanr
from torchtyping import TensorType

from fampnn.data import residue_constants as rc
from fampnn.model.sd_model import SeqDenoiser


def apply_single_muts(x: TensorType["b n a 3", float],
                   aatype: TensorType["b n", int],
                   seq_mask: TensorType["b n", int],
                   missing_atom_mask: TensorType["b n a", int],
                   mutations: List[List[str]]
                   ) -> Tuple[TensorType["b n a 3", float],
                              TensorType["b n", int],
                              TensorType["b n", int]]:

    b = aatype.shape[0]
    seq_mlm_mask = seq_mask.clone()
    scn_mlm_mask = seq_mask.clone()
    aatype_masked = aatype.clone()

    mut_positions = np.zeros(b)
    mut_res_idxs = np.zeros(b)
    wt_res_idxs = np.zeros(b)
    wt_example_mask = np.full(b, False)

    # mutated structure where we will mask the sidechains
    x_masked = x.clone()

    # Loop through the batch and apply mutations
    for i in range(b):

        #parse mutation info
        mut = mutations[i]

        if mut == 'wt':
            wt_example_mask[i] = True
            continue

        wt_res, pos, mut_res = mut[0], int(mut[1:-1]), mut[-1]

        #save positions and wt residue indices for later evaluation
        mut_positions[i] = pos
        mut_res_idxs[i] = rc.restype_order_with_x[mut_res]
        wt_res_idxs[i] = rc.restype_order_with_x[wt_res]

        #ensure mutations are applied correctly
        assert aatype[i, pos] == rc.restype_order_with_x[wt_res], f'Mutation info and sequence do not match!! {[rc.idx_to_restype_with_x[int(res)] for res in aatype[i,pos-5:pos+5]]} {wt_res}{pos}{mut_res}'

        # mask sequence at mutation position
        seq_mlm_mask[i, pos] = 0
        aatype_masked[i, pos] = rc.restype_order_with_x["X"]

        # Mask the sidechains at the mutated positions in x
        x_masked[i, pos, rc.non_bb_idxs, :] = 0
        missing_atom_mask[i, pos, rc.non_bb_idxs] = 1
        scn_mlm_mask[i, pos] = 0


    return x_masked, scn_mlm_mask, seq_mlm_mask, missing_atom_mask, aatype_masked, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask

def apply_multi_muts(model: SeqDenoiser,
                     x: TensorType["b n a 3", float],
                     aatype: TensorType["b n", int],
                     mutation_list: List[List[str]],
                     seq_mask: TensorType["b n", int],
                     residue_index: TensorType["b n", int],
                     missing_atom_mask: TensorType["b n a", float],
                     chain_index: TensorType["b n", int],
                     scd_inputs: Dict,
                     max_number_muts: int,
                     seq_only: False
                    ) -> Tuple[TensorType["b n a 3", float],
                              TensorType["b n", int]]:
    aatype_mut = aatype.clone()
    x_mut = x.clone()
    b = x.shape[0]
    mut_positions = np.full((b, max_number_muts), 0)
    mut_res_idxs = np.full((b, max_number_muts), 0)
    wt_res_idxs = np.full((b, max_number_muts), 0)
    padded_mutations_mask = torch.full((b, max_number_muts), True)
    wt_example_mask = np.full(b, False)

    aux_inputs = {"scd": scd_inputs,
                  "seq_mlm_mask": seq_mask.clone(),
                  "scn_mlm_mask": seq_mask.clone()}


    #mutation_list is nested list as there may be more than one mutation per sequence
    for i in range(b):
        mutations = mutation_list[i]
        for mut_num, mut in enumerate(mutations.split(':')):

            if mut =='wt':
                wt_example_mask[i] = True
                continue

            wt_res, pos, mut_res = mut[0], int(mut[1:-1]), mut[-1]

            #ensuring mutation information is correct
            context_start = max(0, pos - 5)
            context_end = min(aatype.shape[1], pos + 5)  # Ensure we don't exceed sequence length

            assert aatype[i, pos] == rc.restype_order_with_x[wt_res], (
                f"❌ Mismatch between mutation info and sequence at position {pos}!\n"
                f"⚠️ Ensure that:\n"
                f"   - Positions are **zero-indexed**.\n"
                f"   - Indices are **not reset** for each new chain.\n"
                f"Context (residues {context_start} to {context_end}): "
                f"{''.join([rc.idx_to_restype_with_x[int(res)] for res in aatype[i, context_start:context_end]])}\n"
                f"Expected: {wt_res}, Found: {rc.idx_to_restype_with_x[int(aatype[i, pos])]}\n"
                f"Mutation: {wt_res}{pos}{mut_res}"
            )

            #apply mutation to sequence
            aatype_mut[i, pos] = rc.restype_order_with_x[mut_res]

            #mask out wt sidechain coords at mutated position
            aux_inputs['scn_mlm_mask'][i, pos] = 0
            x_mut[i, pos, rc.non_bb_idxs, :] = 0
            missing_atom_mask[i, pos, rc.non_bb_idxs] = 1

            #store mutation info
            mut_positions[i, mut_num] = pos
            padded_mutations_mask[i, mut_num] = False
            mut_res_idxs[i, mut_num] = rc.restype_order_with_x[mut_res]
            wt_res_idxs[i, mut_num] = rc.restype_order_with_x[wt_res]

    #teacher force mutated sequence
    aux_inputs['scd']["aatype_override"] = aatype_mut
    aux_inputs['scd']["aatype_override_mask"] = seq_mask.clone()

    if not seq_only:
        #run model
        x1_mut_pred, _, _ = model.denoiser(x_mut,
                                           aatype_mut,
                                           residue_index=residue_index,
                                           seq_mask=seq_mask,
                                           chain_encoding=chain_index,
                                           missing_atom_mask=missing_atom_mask,
                                           scn_mlm_mask = aux_inputs['scn_mlm_mask'],
                                           aux_inputs=aux_inputs,
                                           is_sampling=True
                                           )

        #update structure with newly packed residues
        x_mut[torch.arange(b).unsqueeze(1), mut_positions,...] = x1_mut_pred[torch.arange(b).unsqueeze(1), mut_positions,...]

    return aatype_mut, x_mut, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask, padded_mutations_mask


def score_seq(model: SeqDenoiser,
              x: TensorType["b n a 3", float],
              aatype: TensorType["b n", int],
              seq_mask: TensorType["b n", int],
              residue_index: TensorType["b n", int],
              missing_atom_mask: TensorType["b n a", float],
              chain_index: TensorType["b n", int],
              mutations: List[List[str]],
              scd_inputs: Dict,
              method: str,
              seq_only: False
            ) -> TensorType["b n", int]:

    B = x.shape[0]

    if method == 'single':
        x_masked, scn_mlm_mask, seq_mlm_mask, missing_atom_mask, aatype_masked, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask = apply_single_muts(x, aatype, seq_mask, missing_atom_mask, mutations)

        if seq_only:
            scn_mlm_mask = torch.zeros_like(seq_mask)

        #score examples
        logprobs = model.score(
            x_masked,
            aatype_masked,
            missing_atom_mask=missing_atom_mask,
            seq_mask=seq_mask,
            scn_mlm_mask=scn_mlm_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        scores = logprobs[torch.arange(len(logprobs)), mut_positions, mut_res_idxs] - logprobs[torch.arange(len(logprobs)), mut_positions, wt_res_idxs]

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    elif method == 'multiple':

        #we will eval O(max_number_of_muts)  so we can parallelize
        max_number_muts = max([len(muts.split(':')) for muts in mutations])
        scores = torch.zeros((B, max_number_muts))

        aatype_mut, x_mut, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask, padded_mutations_mask = apply_multi_muts(model=model,
                                                                                                        x=x,
                                                                                                        aatype=aatype,
                                                                                                        mutation_list=mutations,
                                                                                                        seq_mask=seq_mask,
                                                                                                        residue_index=residue_index,
                                                                                                        missing_atom_mask=missing_atom_mask,
                                                                                                        chain_index=chain_index,
                                                                                                        scd_inputs=scd_inputs,
                                                                                                        max_number_muts=max_number_muts,
                                                                                                        seq_only=seq_only
        )

        scn_mlm_mask = seq_mask.clone()
        for mut_num in range(max_number_muts):
            #mask seq at position
            scn_mlm_mask = scn_mlm_mask.clone()
            scn_mlm_mask[torch.arange(B), mut_positions[:, mut_num]] = 0
            aatype_masked = aatype_mut.clone()
            aatype_masked[torch.arange(B), mut_positions[:, mut_num]] = rc.restype_order_with_x["X"]

            #mask sidechain at position
            x_masked = x_mut.clone()
            x_masked[torch.arange(B)[:,None], mut_positions[:, mut_num][:,None], rc.non_bb_idxs, :] = 0
            missing_atom_mask[torch.arange(B)[:,None], mut_positions[:, mut_num][:,None], rc.non_bb_idxs] = 1

            if seq_only:
                scn_mlm_mask = torch.zeros_like(seq_mask)

            #score examples
            logprobs = model.score(
                x_masked,
                aatype_masked,
                seq_mask=seq_mask,
                scn_mlm_mask=scn_mlm_mask,
                missing_atom_mask=missing_atom_mask,
                residue_index=residue_index,
                chain_index=chain_index,
            )

            scores[:, mut_num] = logprobs[torch.arange(len(logprobs)), mut_positions[:, mut_num], mut_res_idxs[:, mut_num]] - logprobs[torch.arange(len(logprobs)), mut_positions[:, mut_num], wt_res_idxs[:, mut_num]]

        #ignore dummy mutation position
        scores = torch.where(padded_mutations_mask, 0, scores)
        scores = torch.sum(scores, dim = -1)

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    else:
        raise ValueError(f'Incorrect scoring method given: {method}, choose between: single, multiple')
    return scores


def get_avg_metrics(scores_exp: Dict,
                    labels_exp: Dict
                    ):

    experiments = scores_exp.keys()
    return np.mean([pearsonr(scores_exp[exp], labels_exp[exp]).correlation for exp in experiments]), np.mean([spearmanr(scores_exp[exp], labels_exp[exp]).correlation for exp in experiments])


def update_data_cfg(data_cfg: DictConfig
                    ):

    dataset_path = data_cfg.pdb_path
    with open(os.path.join(dataset_path,'config.yaml'), 'r') as file:
        dataset_config = yaml.safe_load(file)

    for k, v in dataset_config.items():
        data_cfg[k] = v

    return data_cfg

def mask_positions(x: TensorType["b n a 3", float],
                   aatype: TensorType["b n", int],
                   seq_mask: TensorType["b n", int],
                   missing_atom_mask: TensorType["b n a", float],
                   positions: list,
                   ):
    B = x.shape[0]
    non_bb_idxs = torch.tensor(rc.non_bb_idxs, device=x.device)
    x_masked, aatype_masked, missing_atom_mask_masked = x.clone(), aatype.clone(), missing_atom_mask.clone()
    scn_mlm_mask = seq_mask.clone()

    scn_mlm_mask[torch.arange(B), positions] = 0
    aatype_masked[torch.arange(B), positions] = rc.restype_order_with_x["X"]
    x_masked[torch.arange(B)[:,None], positions[:,None], non_bb_idxs[None,:], :] = 0
    missing_atom_mask_masked[torch.arange(B)[:,None], positions[:,None], non_bb_idxs[None,:]] = 1

    return x_masked, aatype_masked, missing_atom_mask_masked, scn_mlm_mask
