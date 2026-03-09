from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from openfold.model.primitives import Linear
from torchtyping import TensorType

import fampnn.data.residue_constants as rc
from fampnn.data import life
from fampnn.data.data import get_rc_tensor, transform_sidechain_frame
from fampnn.interpolants.edm_interpolant import EDM
from fampnn.model.sidechain_diffusion.dit_utils import (DenoisingMLPBlock,
                                                        FinalLayer,
                                                        TimestepEmbedder)
from fampnn.model.sidechain_diffusion.sidechain_confidence import \
    SidechainConfidenceModule


class SidechainDiffusionModule(nn.Module):
    def __init__(self, cfg: DictConfig, scn_sigma_data: TensorType[(), float]):
        """
        Sidechain denoising module. For now, basically just a small DiT.
        """
        super().__init__()
        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.scn_interpolant = EDM(cfg.interpolant, sigma_data=scn_sigma_data)

        # Set up denoising model
        self.scn_denoiser = SidechainMLP(cfg.scn_denoiser, self.scn_interpolant)

        # Confidence module
        self.use_confidence_module = cfg.confidence_module.enabled
        if self.use_confidence_module:
            self.confidence_module_train_p = 1 / cfg.confidence_module.subsample_train_iter_mult
            self.confidence_module = SidechainConfidenceModule(cfg.confidence_module)


    def sidechain_diffusion(self,
                            mpnn_feature_dict: Dict[str, TensorType["b ..."]],
                            aatype: TensorType["b n", int],
                            seq_mask: TensorType["b n", float],
                            residue_index: TensorType["b n", int],
                            chain_index: TensorType["b n", int],
                            aux_inputs: Optional[Dict],
                            is_sampling: bool,
                            ) -> Tuple[TensorType["b n a 3", float],
                                       Dict[str, TensorType["b ...", float]]]:
        h_V = mpnn_feature_dict["h_V"]
        B, N, _ = h_V.shape
        diffusion_aux = defaultdict(lambda: None)

        # === Sampling === #
        # Sample sidechains from prior
        A = len(rc.non_bb_idxs)
        x0_scn_local = self.scn_interpolant.sample_prior((B, N, A, 3), h_V.device)

        # Extract sampling parameters
        scd_aux_inputs = aux_inputs["scd"]
        S_scd = scd_aux_inputs["num_steps"]
        timesteps = scd_aux_inputs["timesteps"]
        churn_cfg = scd_aux_inputs["churn_cfg"]
        step_scale = scd_aux_inputs["step_scale"]
        aatype_override_mask = scd_aux_inputs["aatype_override_mask"]
        aatype = torch.where(aatype_override_mask.bool(), scd_aux_inputs["aatype_override"], aatype)

        denoiser_fn = partial(self.scn_denoiser, aatype=aatype,
                                h_V=h_V, seq_mask=seq_mask)
        # Run integration steps
        # Store trajectory
        xt_scn_traj, x1_scn_traj = [], []

        xt_scn_local = x0_scn_local
        for i in range(S_scd):
            t = timesteps[:, i]
            t_next = timesteps[:, i + 1]

            xt_scn_local, t = self.scn_interpolant.churn(xt_scn_local, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

            xt_scn_local, aux_preds = self.scn_interpolant.euler_step(denoiser_fn,
                                                                        xt_scn_local,
                                                                        t=t, t_next=t_next,
                                                                        step_scale=step_scale,
                                                                        cfg_cfg=None)

            if self.use_self_conditioning:
                # Apply self-conditioning
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux_preds["x1_pred"])

        # Finalize outputs
        # Compute confidence using local scn coordinates
        if self.use_confidence_module:
            _, psce = self.confidence_module(xt_scn_local,
                                                mpnn_feature_dict,
                                                aatype,
                                                seq_mask,
                                                residue_index,
                                                chain_index)
            diffusion_aux["psce"] = psce
        else:
            diffusion_aux["psce"] = torch.zeros((B, N, A), device=xt_scn_local.device)

        # Transform denoised sidechains back to global coordinates
        x_bb = mpnn_feature_dict["X"][..., rc.atom14_bb_idxs, :]
        atom_mask_bb = mpnn_feature_dict["atom14_mask"][..., rc.atom14_bb_idxs]  # 0 denotes missing backbone atoms in input backbone
        atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)[..., rc.non_bb_idxs]  # all atoms are present for the packed aatype, since we sampled them
        x1_scn, _ = transform_sidechain_frame(xt_scn_local, x_bb,
                                                atom_mask_scn, atom_mask_bb, to_local=False)  # if backbone frame doesn't exist, we predict all sidechain atoms at CA
        diffusion_aux["scn_pred"] = x1_scn


        return x1_scn, diffusion_aux


class SidechainMLP(nn.Module):
    def __init__(self, cfg: DictConfig, scn_interpolant: EDM):
        """
        MLP for per-token sidechain diffusion conditioned on MPNN sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.scn_interpolant = scn_interpolant

        # Set up MLP model
        self.use_self_conditioning = cfg.use_self_conditioning
        self.in_channels = len(rc.non_bb_idxs) * 3  # 33 * 3; input sidechain atoms
        self.in_channels += cfg.n_aatype  # concatenate one-hot encoded amino acid type

        self.out_channels = len(rc.non_bb_idxs) * 3  # 33 * 3; output all sidechain atoms
        if self.use_self_conditioning:
            self.in_channels += self.out_channels  # concatenate input with output from previous timestep

        self.n_aatype = cfg.n_aatype

        # Conditioning
        self.timestep_embedder = TimestepEmbedder(cfg.hidden_size)

        # input feature embedder: embed reference positions
        self.f_embedder = Linear(cfg.num_atoms_in * 3, cfg.hidden_size)

        # node embedding conditioning
        self.h_V_embedder = nn.Linear(cfg.c_h_V, cfg.hidden_size)

        # Blocks
        self.x_embedder = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")

        # Blocks
        self.blocks = nn.ModuleList([
            DenoisingMLPBlock(cfg.hidden_size,
                              mlp_dropout=cfg.mlp_dropout,
                              mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.depth)
            ])
        self.final_layer = FinalLayer(cfg.hidden_size, self.out_channels)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self,
                x_scn: TensorType["b n a_scn 3", float],  # noisy sidechain atoms
                aatype: TensorType["b n", float],  # aatype to condition on (predicted during inference; GT during training)
                t: TensorType["b n", float],  # timestep
                h_V: TensorType["b n h", float],  # conditioning latent
                seq_mask: TensorType["b n", float],
                x_scn_self_cond: Optional[TensorType["b n a_scn 3", float]] = None,  # self-conditioning input
                ) -> Tuple[TensorType["b n a 3", float], Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        # Preconditioning
        precondition_in, precondition_out = self.scn_interpolant.setup_preconditioning(x_scn, x_scn_self_cond, t)
        x_scn, x_scn_self_cond, t = precondition_in()  # input preconditioning

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_scn_self_cond is None:
                x_scn_self_cond = torch.zeros_like(x_scn)
            x_scn = torch.cat([x_scn, x_scn_self_cond], dim=-1)

        x = rearrange(x_scn, "b n a x -> b n (a x)")

        # Concatenate one-hot sequence conditioning
        aatype_oh = F.one_hot(aatype, num_classes=self.n_aatype).float()  # aatype is ground truth during training
        x = torch.cat([x, aatype_oh], dim=-1)

        # Begin MLP forward pass
        x = self.x_embedder(x)

        # Embed reference positions
        ref_pos = life.RESTYPE_REF_POS_ATOM37.to(aatype.device)[aatype.long()]
        ref_pos = rearrange(ref_pos, "b n a x -> b n (a x)")
        x = x + self.f_embedder(ref_pos)

        # Conditioning
        # embed timestep
        c = self.timestep_embedder(t).unsqueeze(1)

        # add conditioning from h_V
        h_V = self.h_V_embedder(h_V)
        c = c + h_V
        x = x + h_V

        # MLP blocks
        for block in self.blocks:
            x = block(x, c)

        # Final output
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x_scn = precondition_out(x)  # output preconditioning on sidechains

        return x_scn, aux_preds
