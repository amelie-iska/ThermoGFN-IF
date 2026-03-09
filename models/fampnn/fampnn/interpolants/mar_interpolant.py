from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

from fampnn.data import residue_constants as rc


class MAR():
    def __init__(self, cfg: DictConfig):
        """
        Interpolant for masked autoregressive diffusion on sequence and sidechains.
        """
        super().__init__()
        self.cfg = cfg

        # Training noise distribution
        self.training_noise_schedule = cfg.training_noise_schedule
        assert self.training_noise_schedule in ["uniform_t", "constant_t",
                                                "uniform_squared_t", "uniform_cubed_t",
                                                "uniform_cosine_t", "uniform_sqrt_t", "uniform_cbrt_t"], f"Unknown timestep schedule: {self.timestep_schedule}"

        self.training_noise_cfg = cfg.training_noise_cfg[self.training_noise_schedule]

        # Additional training tasks
        self.full_noise_p = getattr(cfg, "full_noise_p", 0.0)  # randomly set full_noise_p timesteps to full noise
        self.drop_scn_p = getattr(cfg, "drop_scn_p", 0.0)  # randomly set all sidechain atoms to zero


    @torch.compiler.disable
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                x0: Optional[TensorType["b n a 3"]] = None,
                ) -> Dict[str, Any]:
        x1 = batch["x"]

        # Sample time steps if not provided
        if t is None:
            t = self.sample_timestep(x1.shape[0], device=x1.device)

        # Get noisy samples
        xt, aatype_noised, seq_mlm_mask = self.noise_samples(x1, batch["aatype"], t, batch["seq_mask"])

        # During training, randomly drop sidechains from unmasked aatypes
        if self.training:
            xt, scn_mlm_mask = self.drop_sidechains(xt, seq_mlm_mask)
        else:
            scn_mlm_mask = seq_mlm_mask.clone()

        # Construct outputs
        outputs = {}
        outputs["t"] = t  # [b]
        outputs["x_noised"] = xt  # [b n a 3]
        outputs["aatype_noised"] = aatype_noised  # [b n]
        outputs["seq_mlm_mask"] = seq_mlm_mask  # [b n]
        outputs["scn_mlm_mask"] = scn_mlm_mask  # [b n]

        return outputs


    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        """
        Sample a batch of b timesteps from the noise schedule.

        - uniform_t: sample time from uniform distribution
        - constant_t: sample time from constant distribution
        """
        if self.training_noise_schedule == "constant_t":
            t = torch.ones(n, device=device) * self.training_noise_cfg.t
        elif self.training_noise_schedule.startswith("uniform"):
            # sample time from uniform distribution
            t_min, t_max = self.training_noise_cfg.t_min, self.training_noise_cfg.t_max
            t = torch.rand(n, device=device) * (t_max - t_min) + t_min

            # apply transformation to t
            if self.training_noise_schedule == "uniform_t":
                t = t
            elif self.training_noise_schedule == "uniform_squared_t":
                t = t ** 2
            elif self.training_noise_schedule == "uniform_cubed_t":
                t = t ** 3
            elif self.training_noise_schedule == "uniform_cosine_t":
                t = 1 - torch.cos(t * np.pi / 2)
            elif self.training_noise_schedule == "uniform_sqrt_t":
                t = t ** 0.5
            elif self.training_noise_schedule == "uniform_cbrt_t":
                t = t ** (1/3)

        if self.full_noise_p > 0:
            # randomly set full_noise_p timesteps to full noise
            t = torch.where(torch.rand(n, device=device) < self.full_noise_p, torch.zeros(n, device=device), t)

        return t


    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        """
        Sample n samples from the prior.
        """
        return torch.zeros(*shape, device=device)


    def noise_samples(self,
                      x: TensorType["b n a 3"],
                      aatype: TensorType["b n", int],
                      t: TensorType["b"],
                      seq_mask: TensorType["b n"],
                      ) -> Tuple[TensorType["b n a 3", float],
                                 TensorType["b n", int],
                                 TensorType["b n", int]]:
        """
        Add noise to x and aatype. Return x_noised, aatype_noised, and mlm_mask.

        For MAR, we keep each residue with probability t.
        When masking residues, we zero out the coordinates and set aatype to X.
        """
        B, N, _, _ = x.shape
        mlm_mask = torch.rand(B, N, device=x.device) < rearrange(t, "b -> b 1")  # 1 if we keep the residue, 0 if we mask it
        mlm_mask = (mlm_mask * seq_mask).float()  # mask out residues that are not in the sequence
        x_noised = x.clone()

        # Mask sidechains based on mlm_mask
        x_noised[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(mlm_mask, "b n -> b n 1 1").float()

        # Mask sequence based on mlm_mask
        aatype_noised = torch.where(mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])  # TODO: replace with MASK
        aatype_noised = aatype_noised * seq_mask  # set pad residues back to 0
        aatype_noised = aatype_noised.long()

        # Occasionally zero out all sidechain atoms
        scn_mask = torch.ones(B, device=x.device)
        if self.drop_scn_p > 0:
            scn_mask = (torch.rand(B, device=x.device) > self.drop_scn_p).float()
            x_noised[..., rc.non_bb_idxs, :] = x_noised[..., rc.non_bb_idxs, :] * rearrange(scn_mask, "b -> b 1 1 1")

        return x_noised, aatype_noised, mlm_mask


    def drop_sidechains(self,
                        x: TensorType["b n a 3"],
                        seq_mlm_mask: TensorType["b n"],
                        ) -> Tuple[TensorType["b n a 3", float],
                                   TensorType["b n", int]]:
        """
        Randomly drop out sidechains of unmasked aatypes.
        """
        # Sample probability of dropping sidechains (TODO: we can try different schedules for this)
        keep_scn_t = torch.rand(x.shape[0], device=x.device)  # choose probability of keeping from uniform

        # Create sidechain mlm mask
        scn_mlm_mask = torch.rand_like(seq_mlm_mask) < rearrange(keep_scn_t, "b -> b 1")  # [b n]
        scn_mlm_mask = scn_mlm_mask * seq_mlm_mask  # sidechains should be dropped where aatype is masked

        # Noise sidechains
        x_noised = x.clone()
        x_noised[..., rc.non_bb_idxs, :] = x_noised[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()

        return x_noised, scn_mlm_mask
