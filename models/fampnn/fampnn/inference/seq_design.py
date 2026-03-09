import glob
import pickle
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fampnn import sampling_utils
from fampnn.data import residue_constants as rc
from fampnn.data.data import (load_feats_from_pdb, pad_to_max_len,
                              process_single_pdb)
from fampnn.model.sd_model import SeqDenoiser
from fampnn.sampling_utils import seed_everything


@hydra.main(config_path="../../configs", config_name="seq_design", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for all PDBs in a directory.
    For each batch of PDBs, we produce one designed sequence per PDB.
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

    # Make output directories
    out_dir = cfg.out_dir  # base output directory
    sample_out_dir = f"{out_dir}/samples"  # directory for designed PDBs
    fasta_out_dir = f"{out_dir}/fastas"  # directory for sequences in FASTA format
    sample_pkl_dir = f"{out_dir}/sample_pkls"  # directory for pkls containing helpful info about each sample

    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(fasta_out_dir).mkdir(parents=True, exist_ok=True)
    Path(sample_pkl_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    ### Read in fixed positions ###
    if cfg.fixed_pos_csv is not None:
        fixed_pos_df = pd.read_csv(cfg.fixed_pos_csv, names=["fixed_pos_seq", "fixed_pos_scn"], index_col=0)
        fixed_pos_df = fixed_pos_df.fillna("")
        fixed_pos_df.index = fixed_pos_df.index.str.replace(".pdb", "").str.replace(".cif", "")  # remove extension from index if present
    else:
        fixed_pos_df = pd.DataFrame(columns=["fixed_pos_seq", "fixed_pos_scn"])

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

    # If specified, pre-sort by length for efficiency (descending)
    if cfg.presort_by_length:
        # determine lengths
        results = Parallel(n_jobs=-1)(delayed(get_length)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        # sort by length, longest first
        pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)

    ### SAMPLING ###
    print(f"Sampling with num denoising steps S={cfg.timestep_schedule.num_steps} on {len(pdb_files)} PDBs")

    # Set up sequence design timesteps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,  # will be filled per batch
        "step_scale": cfg.scn_diffusion.step_scale,
        "churn_cfg": churn_cfg,
    }

    # Process PDBs in batches of size B
    pdb_files_repeated = np.repeat(pdb_files, cfg.num_seqs_per_pdb)

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

        # Prepare sampling timesteps
        timesteps = t_seq[None].expand(B, -1).to(device)

        aatype_override_mask, scn_override_mask = sampling_utils.get_override_masks(batch, pdb_names, batch_chain_id_mapping, fixed_pos_df, verbose=cfg.fixed_pos_verbose, mode="seq_design")

        # Run sampling
        x_denoised, aatype_denoised, aux = model.sample(
            batch["x"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            timesteps=timesteps,
            seq_only=cfg.seq_only,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            scd_inputs=scd_inputs,
        )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
            # save other useful info
            "original_aatype": batch["aatype"],
            "interface_residue_mask": batch["interface_residue_mask"],
            "aatype_override_mask": aatype_override_mask,
            "scn_override_mask": scn_override_mask,
        }

        # Save outputs
        # Save to PDB
        pdb_keys = [f"{pdb_name}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
        pdbs = [f"{sample_out_dir}/{pdb_key}.pdb" for pdb_key in pdb_keys]
        fastas = [f"{fasta_out_dir}/{pdb_key}.fasta" for pdb_key in pdb_keys]
        pred_seqs = []
        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        for j, pdb_file in enumerate(pdb_batch_files):
            # Extract the sequence
            seq_mask_i = samples["seq_mask"][j].cpu()
            pred_aatype_i = samples["pred_aatype"][j].cpu()
            pred_aatype_i = pred_aatype_i[seq_mask_i.bool()]
            pred_seq_i = "".join(rc.restypes_with_x[a] for a in pred_aatype_i)
            pred_seqs.append(pred_seq_i)

            # Save fasta
            fasta_out = fastas[j]
            with open(fasta_out, "w") as f:
                f.write(f">{pdb_keys[j]}\n{pred_seq_i}\n")

        # Save samples as pkl
        for j in range(B):
            sample_j = {k: v[j].cpu().numpy() for k, v in samples.items()}
            # crop to the actual sequence length
            seq_mask_j = sample_j["seq_mask"]
            sample_j = {k: v[seq_mask_j.astype(bool)] for k, v in sample_j.items()}
            with open(f"{sample_pkl_dir}/{pdb_keys[j]}.pkl", "wb") as f:
                pickle.dump(sample_j, f)

        pbar.update(B)
    pbar.close()


def get_length(pdb_file: str) -> Tuple[str, int]:
    data = load_feats_from_pdb(pdb_file)
    return pdb_file, len(data["aatype"])


if __name__ == "__main__":
    main()
