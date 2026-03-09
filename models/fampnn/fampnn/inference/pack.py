import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fampnn import sampling_utils
from fampnn.data.data import (load_feats_from_pdb, pad_to_max_len,
                              process_single_pdb)
from fampnn.model.sd_model import SeqDenoiser
from fampnn.sampling_utils import seed_everything


@hydra.main(config_path="../../configs", config_name="pack", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    seed_everything(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
        torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Load in sequence denoiser (in eval mode)
    torch.set_grad_enabled(False)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    model = SeqDenoiser(ckpt["model_cfg"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    # Create output directories and preserve config
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "samples")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    ### Load in PDB files ###
    if cfg.pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}" for key in pdb_keys]
    else:
        # Get all PDBs with .pdb extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*.pdb")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

    ### Read in fixed positions ###
    if cfg.fixed_pos_csv is not None:
        fixed_pos_df = pd.read_csv(cfg.fixed_pos_csv, names=["fixed_pos_seq", "fixed_pos_scn"], index_col=0)
        fixed_pos_df = fixed_pos_df.fillna("")
        fixed_pos_df.index = fixed_pos_df.index.str.replace(".pdb", "").str.replace(".cif", "")  # remove extension from index if present
    else:
        fixed_pos_df = pd.DataFrame(columns=["fixed_pos_seq", "fixed_pos_scn"])


    ### Sample ###
    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,  # will be filled per batch
        "step_scale": cfg.scn_diffusion.step_scale,
        "churn_cfg": churn_cfg,
    }

    pdb_files_repeated = np.repeat(pdb_files, cfg.num_samples_per_pdb)
    pbar = tqdm(total=len(pdb_files_repeated))
    for i in range(0, len(pdb_files_repeated), cfg.batch_size):
        pdb_batch_files = pdb_files_repeated[i:i+cfg.batch_size]
        B = len(pdb_batch_files)

        # Load and process all PDBs in this batch
        batch_list = []
        batch_chain_id_mapping = []
        for pdb_file in pdb_batch_files:
            data = load_feats_from_pdb(pdb_file)
            single = process_single_pdb(data)
            batch_list.append(single)

            # store chain ID mapping for parsing fixed positions
            batch_chain_id_mapping.append(data["chain_id_mapping"])
        pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

        # Create a batch dictionary from batch_list by stacking
        model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
        max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
        batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
        batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

        # Move to device
        batch = {k: batch[k].to(device) for k in model_input_keys}

        # Prepare scd_inputs for this batch
        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

        # Read in fixed positions for this batch
        aatype_override_mask, scn_override_mask = sampling_utils.get_override_masks(batch, pdb_names, batch_chain_id_mapping, fixed_pos_df, verbose=cfg.fixed_pos_verbose, mode="packing")

        # Pack sidechains
        x_denoised, aatype_denoised, aux = model.sidechain_pack(
            batch["x"],
            batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            scd_inputs=scd_inputs,
        )

        samples = {"x_denoised": x_denoised,
                   "seq_mask": batch["seq_mask"],
                   "missing_atom_mask": torch.zeros_like(batch["missing_atom_mask"]),
                   "residue_index": batch["residue_index"],
                   "chain_index": batch["chain_index"],
                   "pred_aatype": aatype_denoised,
                   "psce": aux["psce"],
                   }

        # Save samples to PDB
        pdb_keys = [f"{pdb_name}_sample{(i+j) % cfg.num_samples_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
        pdbs = [f"{sample_out_dir}/{pdb_key}.pdb" for pdb_key in pdb_keys]
        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        pbar.update(B)

    pbar.close()


if __name__ == "__main__":
    main()
