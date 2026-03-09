from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from fampnn import sampling_utils
from fampnn.data import residue_constants as rc
from fampnn.data.data import get_rc_tensor
from fampnn.data.pdb_utils import *
from fampnn.interpolants.mar_interpolant import MAR
from fampnn.model.fampnn_denoiser import FAMPNNDenoiser


class SeqDenoiser(nn.Module):
    """
    Sequence denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Data scaling parameters
        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = self.scn_std

        self.denoiser = FAMPNNDenoiser(cfg.denoiser, self.sigma_data)
        self.interpolant = MAR(cfg.interpolant)


    def score(self,
              x,
              aatype,
              seq_mask,
              missing_atom_mask,
              scn_mlm_mask,
              residue_index,
              chain_index
        ) -> TensorType["b n"]:
        """
        batch should contain:
        - x: TensorType["b n a 3", float]
        - residue_index: TensorType["b n", int]
        - seq_mask: TensorType["b n", float]
        """
        atom_mask_noised = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)  # 0 for ghost atoms; X only has backbone atoms
        atom_mask_noised = atom_mask_noised * seq_mask.unsqueeze(-1)  # mask out padding
        atom_mask_noised = atom_mask_noised * (1 - missing_atom_mask)  # mask out missing atoms
        atom_mask_noised[..., rc.non_bb_idxs] = atom_mask_noised[..., rc.non_bb_idxs] * scn_mlm_mask.unsqueeze(-1)  # mask out masked sidechain atoms


        # Run denoiser and get logits
        seq_logits, _ = self.denoiser.seq_design_module(
                                        x,
                                        aatype,
                                        seq_mask,
                                        atom_mask_noised,
                                        residue_index,
                                        chain_index)
        log_probs = F.log_softmax(seq_logits, dim=-1)
        return log_probs


    def sidechain_pack(self,
                       x: TensorType["b n a 3", float],
                       aatype: TensorType["b n", int],
                       seq_mask: TensorType["b n", float],
                       missing_atom_mask: TensorType["b n 37", float],  # 1 where atoms are missing
                       residue_index: TensorType["b n", int],
                       chain_index: TensorType["b n", int],
                       scn_override_mask: Optional[TensorType["b n", int]] = None,
                       aatype_override_mask: Optional[TensorType["b n", int]] = None,
                       scd_inputs: Dict[str, Any] = {}):
        """
        Given backbone and sequence, denoise sidechain atoms (sidechain packing).

        Also supports packing partial sequence with partial sidechains through aatype_override_mask and scn_override_mask.

        scd_inputs should contain the following keys:
        - num_steps: int
        - timesteps: TensorType["b S_scd+1", float]
        - churn_cfg: Dict[str, Any]
        """
        aux, aux_inputs = {}, {}
        B, N, A, _ = x.shape

        # Override aatype with the input aatype during sequence denoising
        if aatype_override_mask is None:
            # if not provided, assume full sequence
            aatype_override_mask = seq_mask.clone()

        # Set sidechain to fully masked
        if scn_override_mask is None:
            # if not provided, assume no sidechains
            scn_override_mask = torch.zeros_like(seq_mask)

        # Set up structure input dependent on structure mask
        x0 = x.clone()
        x0[:,:,rc.non_bb_idxs,:] =  x0[:,:,rc.non_bb_idxs,:] * scn_override_mask[:,:,None,None]

        # Sample aatype prior dependency on aatype mask
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
        aatype_noised = torch.where(aatype_override_mask == 1, aatype, aatype_noised)

        # Override aatype in sidechain diffusion as well
        scd_inputs["aatype_override"] = aatype_noised
        scd_inputs["aatype_override_mask"] = aatype_override_mask

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        seq_mlm_mask = torch.zeros_like(seq_mask).float() + aatype_override_mask # start with all masked tokens, other than partial seq
        scn_mlm_mask = torch.zeros_like(seq_mask).float() + scn_override_mask # start with all masked tokens, other than partial scn
        assert torch.all((seq_mlm_mask - scn_mlm_mask) >= 0), "Unmasked sidechains should be a subset of unmasked sequence"

        # Run denoising steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              missing_atom_mask=missing_atom_mask,
                              chain_encoding=chain_index,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        psce_t = torch.zeros((B, N, len(rc.non_bb_idxs)), device=x.device)

        # Run sequence denoiser to get packed sidechains
        x1_pred, _, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)

        # Unmask sidechains and sidechain confidence to match seq_mlm_mask
        xt = sampling_utils.unmask(xt, x1_pred, scn_mlm_mask, seq_mlm_mask)
        psce_t = sampling_utils.unmask(psce_t, aux_preds["scn_diffusion_aux"]["psce"], scn_mlm_mask, seq_mlm_mask)

        aux["psce"] = psce_t
        aux["seq_mask"] = seq_mask

        return xt, aatype_t, aux


    def sample(self,
               x: TensorType["b n a 3", float],
               aatype: TensorType["b n", int],
               seq_mask: TensorType["b n", float],
               missing_atom_mask: TensorType["b n a 3", float],  # 1 where atoms are missing
               residue_index: TensorType["b n", int],
               chain_index:  TensorType["b n", int],
               timesteps: TensorType["b s+1", float],  # timesteps for t_seq
               temperature: float,  # 0.0 for argmax / greedy sampling
               seq_only: bool = False,  # only sample sequence
               repack_last: bool = False,  # repack last step after sampling the sequence
               psce_threshold: Optional[float] = None,  # during design, only keep sidechains with psce below threshold; None to keep all
               scn_override_mask: Optional[TensorType["b n", int]] = None,
               aatype_override_mask: Optional[TensorType["b n", int]] = None,
               restrict_pos_aatype: Optional[Tuple[TensorType["b n", float],
                                                   TensorType["b n k", int]]] = None,  # restrict aatype sampling at certain positions
               scd_inputs: Dict[str, Any] = {},  # sidechain diffusion inputs
               ):
        """
        scd_inputs should contain the following keys:
        - num_steps: int
        - timesteps: TensorType["b S_scd+1", float]
        - churn_cfg: Dict[str, Any]
        """
        aux, aux_inputs = {}, {}
        S = timesteps.shape[1] - 1
        B, N, A, _ = x.shape

        # Handle default overrides
        if aatype_override_mask is None:
            aatype_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        if scn_override_mask is None:
            scn_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Set up structure input dependent on structure mask
        x0 = x.clone()
        x0[:,:,rc.non_bb_idxs,:] =  x0[:,:,rc.non_bb_idxs,:] * scn_override_mask[:,:,None,None]

        # Sample aatype prior, dependending on aatype mask
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
        aatype_noised = torch.where(aatype_override_mask == 1, aatype, aatype_noised)

        # Override aatype in sidechain diffusion as well
        scd_inputs["aatype_override"] = aatype_noised
        scd_inputs["aatype_override_mask"] = aatype_override_mask

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        # Get residue decoding order
        seq_mlm_mask = torch.zeros_like(seq_mask).float() + aatype_override_mask # start with all masked tokens, other than partial seq
        scn_mlm_mask = torch.zeros_like(seq_mask).float() + scn_override_mask # start with all masked tokens, other than partial scn
        aatype_decoding_order = sampling_utils.get_decoding_order(seq_mask=seq_mask)
        aux_inputs["lengths"] = seq_mask.sum(dim=-1)
        aux_inputs["temperature"] = temperature

        # Initialize trajectories
        xt_traj = []
        aatype_t_traj, aatype_pred_traj = [], []
        psce_t_traj = []
        seq_logits_traj = []

        # Set up function for updating mlm mask
        mask_update_fn = partial(sampling_utils.update_mlm_mask,
                                 aatype_decoding_order=aatype_decoding_order)

        # Run denoising steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              missing_atom_mask=missing_atom_mask,
                              chain_encoding=chain_index,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        seq_probs_t = torch.zeros((B, N, len(rc.restypes_with_x)), device=x.device)  # keep track of unscaled sequence probabilities as we decode
        psce_t = torch.zeros((B, N, len(rc.non_bb_idxs)), device=x.device)  # keep track of sidechain confidence as we decode

        if torch.any((aatype_override_mask - scn_override_mask) < 0):
            raise ValueError('Sidechain cannot be fixed at any positions where sequence is not fixed')

        # Get timesteps based on the number of unmasked residues
        num_partial = seq_mlm_mask.sum(dim=-1).long()
        timesteps_K = torch.ceil(timesteps * (aux_inputs["lengths"][:, None] - num_partial[:,None])).long()  # timestep schedule is defined relative to masked residues
        timesteps_K += num_partial[:,None]

        for i in tqdm(range(S), leave=False, desc="Sampling..."):
            # get next K residues to unmask
            K_next = timesteps_K[:, i + 1]

            # Run sequence denoiser
            x1_pred, aatype_pred, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)

            # Update masks
            seq_mlm_mask_prev, scn_mlm_mask_prev = seq_mlm_mask.clone(), scn_mlm_mask.clone()
            seq_mlm_mask = mask_update_fn(seq_mlm_mask, K=K_next)
            scn_mlm_mask = seq_mlm_mask.clone() if not seq_only else scn_override_mask  # default to user-provided sidechains if seq_only

            # Unmask sequence, sidechains, and sidechain confidence
            aatype_t = sampling_utils.unmask(aatype_t, aatype_pred, seq_mlm_mask_prev, seq_mlm_mask)
            xt = sampling_utils.unmask(xt, x1_pred, scn_mlm_mask_prev, scn_mlm_mask)
            seq_probs_t = sampling_utils.unmask(seq_probs_t, aux_preds["seq_probs"], seq_mlm_mask_prev, seq_mlm_mask)
            psce_t = sampling_utils.unmask(psce_t, aux_preds["scn_diffusion_aux"]["psce"], scn_mlm_mask_prev, scn_mlm_mask)

            if (psce_threshold is not None) and (i != S - 1):
                # Re-mask sidechains with low confidence, but only if we are not at the last step
                # get mask based on per-residue confidence
                atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_t)[..., rc.non_bb_idxs]
                psce_t_per_res = (psce_t * atom_mask_scn).sum(dim=-1) / atom_mask_scn.sum(dim=-1).clamp(min=1)  # average confidence per residue
                scn_mlm_mask = scn_mlm_mask * (psce_t_per_res <= psce_threshold).float()

                # apply mask
                xt[..., rc.non_bb_idxs, :] = xt[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()
                psce_t = psce_t * rearrange(scn_mlm_mask, "b n -> b n 1")

            # Save trajectory outputs
            xt_traj.append(xt.cpu())
            aatype_t_traj.append(aatype_t.cpu())
            psce_t_traj.append(psce_t.cpu())
            aatype_pred_traj.append(aatype_pred.cpu())
            seq_logits_traj.append(aux_preds["seq_logits"].cpu())

        if repack_last:
            # Repack the structure after sampling the sequence (except for the provided sidechains)
            xt, _, aux_preds_pack = self.sidechain_pack(xt, aatype_t, seq_mask, missing_atom_mask, residue_index, chain_index,
                                                        scn_override_mask,  # start from the provided sidechains
                                                        seq_mlm_mask,  # pack to the known sequence
                                                        scd_inputs)
            psce_t = aux_preds_pack["psce"]
            scn_mlm_mask = seq_mlm_mask.clone()

        aux["xt_traj"] = torch.stack(xt_traj, dim=1)
        aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=1)
        aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=1)
        aux["seq_probs"] = seq_probs_t
        aux["psce"] = psce_t
        aux["psce_t_traj"] = torch.stack(psce_t_traj, dim=1)
        aux["seq_logits_traj"] = torch.stack(seq_logits_traj, dim=1)
        aux["seq_mask"] = seq_mask

        return xt, aatype_t, aux


    @staticmethod
    def save_samples_to_pdb(samples: Dict[str, TensorType["b ..."]],
                            filenames: List[str],
                            ) -> None:
        """
        Save samples from the denoiser to PDB files.
        Samples should contain the following keys:
        - x_denoised: Tensor["b n a 3", float]
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        - pred_aatype: Tensor["b n", int]
        - psce: Tensor["b n 33", float]
        """
        final_atom37_positions = samples["x_denoised"]
        residue_index = samples["residue_index"]
        seq_mask = samples["seq_mask"]
        aatype = samples["pred_aatype"]
        chain_index = samples["chain_index"]

        # Create atom mask, including backbone atoms even for unknown aatype
        atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=aatype.device)[aatype] * seq_mask[..., None]
        atom_mask = atom_mask * (1 - samples["missing_atom_mask"])  # mask out missing atoms

        # Set b-factors to predicted Sidechain Error (PSCE)
        b_factors = torch.zeros_like(atom_mask, dtype=torch.float32).cpu()
        b_factors[..., rc.non_bb_idxs] = samples["psce"].cpu()

        feats = {
            "aatype": aatype,
            "atom_positions": final_atom37_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": chain_index,
            "b_factors": b_factors
        }

        feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}  # move to cpu
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")
