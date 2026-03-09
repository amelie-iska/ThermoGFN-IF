from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fampnn import sampling_utils, scoring_utils
from fampnn.data.data import load_feats_from_pdb, process_single_pdb
from fampnn.model.sd_model import SeqDenoiser
from fampnn.sampling_utils import seed_everything


@hydra.main(config_path="../../configs", config_name="score", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for scoring the likelihood of mutations given an input protein
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Load in sequence denoiser (in eval mode)
    torch.set_grad_enabled(False)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    model = SeqDenoiser(ckpt["model_cfg"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    # Make output directory and preserve config
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load input PDB
    data = load_feats_from_pdb(cfg.pdb_path)
    batch = process_single_pdb(data)

    # Move inputs to device
    model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
    model_inputs = {k: batch[k].to(device) for k in model_input_keys}

    # Repeat inputs along new batch dimension
    B = cfg.batch_size
    repeat_along_batch_fn = lambda x: x[None, ...].repeat(B, *([1] * len(x.shape)))
    model_inputs = {k: repeat_along_batch_fn(v) for k, v in model_inputs.items()}

    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    # create sidechain diffusion churn config
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                  "timesteps": None,  # filled in based on batch size
                  "step_scale": cfg.scn_diffusion.step_scale,
                  "churn_cfg": churn_cfg,
                  }

    mutations_df = pd.read_csv(cfg.mutations_path, usecols=['mutations'])
    num_mutations = len(mutations_df)

    # Split into batches
    num_batches = num_mutations // B  + (num_mutations % B > 0)
    mutations_list = mutations_df['mutations'].to_list()
    batched_mutations = np.array_split(mutations_list, num_batches)

    ### BEGIN EVAL ###
    scores_all = []

    for mutations in tqdm(batched_mutations, desc='Evaluating mutations', total=num_batches):
        x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = [model_inputs[k][:len(mutations),...].clone() for k in model_input_keys]

        scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)
        scores = scoring_utils.score_seq(model=model,
                                         x=x,
                                         aatype=aatype,
                                         seq_mask=seq_mask,
                                         residue_index=residue_index,
                                         missing_atom_mask=missing_atom_mask,
                                         chain_index=chain_index,
                                         mutations=mutations,
                                         scd_inputs=scd_inputs,
                                         method='multiple',
                                         seq_only=False).cpu().tolist()

        scores_all += scores

    mutations_df['scores'] = scores_all
    mutations_df.to_csv(f"{cfg.out_dir}/score_multi.csv", index=False)


if __name__ == "__main__":
    main()
