import math
from pathlib import Path

import hydra
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fampnn import scoring_utils
from fampnn.data import residue_constants as rc
from fampnn.data.data import load_feats_from_pdb, process_single_pdb
from fampnn.model.sd_model import SeqDenoiser
from fampnn.sampling_utils import seed_everything


@hydra.main(config_path="../../configs", config_name="score", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for scoring every possible mutation for an input pdb
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

    # Compute valid positions
    num_positions = int(torch.sum(model_inputs['seq_mask'][0]).item())
    total_positions = torch.arange(num_positions)  # Create tensor of positions

    # Split into batches
    num_batches = math.ceil(num_positions / B)
    batched_positions = total_positions.split(B)  # Create batches

    # Iterate over batches
    scores_dict = {}
    for positions in tqdm(batched_positions, total=num_batches):
        x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = [model_inputs[k][:len(positions),...].clone() for k in model_input_keys]

        #mask positions we are evaluating
        x_masked, aatype_masked, missing_atom_mask_masked, scn_mlm_mask = scoring_utils.mask_positions(x, aatype, seq_mask, missing_atom_mask, positions)

        #score examples
        logprobs = model.score(
            x=x_masked,
            aatype=aatype_masked,
            missing_atom_mask=missing_atom_mask_masked,
            seq_mask=seq_mask,
            scn_mlm_mask=scn_mlm_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        #get scores by normalizing with wild-type
        wt_aatype = aatype[torch.arange(len(positions)), positions]
        wt_logprobs = logprobs[torch.arange(len(positions)), positions, wt_aatype][:,None]
        mut_logprobs = logprobs[torch.arange(len(positions)), positions, :]
        scores = mut_logprobs - wt_logprobs

        #add scores to dictionary
        for score, position, wt_res in zip(scores, positions, wt_aatype):
            scores_dict[f'{position.item() + 1}{rc.idx_to_restype_with_x[wt_res.item()]}'] = {rc.restypes_with_x[res_num]: score[res_num].item() for res_num in range(rc.restype_num)}

    #Output scores into csv
    out_df = pd.DataFrame(scores_dict).T
    out_df.to_csv(f"{cfg.out_dir}/all_scores.csv")


if __name__ == "__main__":
    main()