#!/usr/bin/env python
"""
test/inverse_folding.py

End-to-end example for ADFLIP all-atom inverse protein folding.
Loads a pretrained checkpoint, processes a PDB file, runs sampling,
and reports overall and interacting-residue recovery rates.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ema_pytorch import EMA
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN
from data import all_atom_parse as aap

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    

def propagate_mask_vectorized(mask: np.ndarray, index: np.ndarray) -> torch.BoolTensor:
    """
    Given a boolean mask per-atom and a residue_index array,
    returns a per-atom mask where each residue is True if ANY
    of its atoms was True in the original mask.
    """
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    idx_t  = torch.as_tensor(index, dtype=torch.long)

    unique_idx, inv_idx = torch.unique(idx_t, return_inverse=True)
    any_true = torch.zeros_like(unique_idx, dtype=torch.bool)
    any_true.scatter_reduce_(0, inv_idx, mask_t, reduce='amax')
    return any_true[inv_idx]


def interact_residue(data) -> object:
    """
    Annotate `data` with four new boolean arrays:
       .interact_non_protein_res
       .interact_ion_res
       .interact_nucleotide_res
       .interact_molecule_res

    Each is True for residues whose any atom lies within 5Å of
    the respective non-protein/ion/nucleotide/other atoms.
    """
    pos = data.position
    is_p = data.is_protein
    idx = data.residue_index

    # gather per-group positions
    prot_pos = pos[is_p][:, None, :]
    others = {
        'non_protein': pos[~is_p][None, :, :],
        'ion':         pos[data.is_ion][None, :, :],
        'nucleotide':  pos[data.is_nucleotide][None, :, :],
        'molecule':    pos[~is_p & ~data.is_ion & ~data.is_nucleotide][None, :, :]
    }

    for name, group_pos in others.items():
        dists = np.linalg.norm(prot_pos - group_pos, axis=-1)
        mask_atoms = np.any(dists < 5.0, axis=1)         # per-protein-atom
        mask_res   = propagate_mask_vectorized(mask_atoms, idx[is_p])
        setattr(data, f"interact_{name}_res", mask_res.numpy())

    return data


def pdb2data(pdb_file: str, device: torch.device) -> dict:
    """
    Parse mmCIF/PDB to torch-friendly dict, add interaction masks.
    """
    data = aap.parse_mmcif_to_structure_data(pdb_file)
    data = interact_residue(data)

    tensor_data = {}
    for key, val in data.__dict__.items():
        if isinstance(val, np.ndarray):
            tensor = torch.from_numpy(val)
            tensor_data[key] = tensor.to(device)
        else:
            tensor_data[key] = torch.tensor([val], device=device)

    # add batch index
    tensor_data['batch_index'] = torch.zeros_like(tensor_data['residue_index'])
    # ensure floats are float32
    for k, v in tensor_data.items():
        if v.dtype == torch.float64:
            tensor_data[k] = v.float()

    return tensor_data


def build_denoiser(config, device: torch.device):
    cfg = config.zoidberg_denoiser
    model = Zoidberg_GNN(
        hidden_dim=cfg.hidden_dim,
        encoder_hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_layers,
        num_heads=cfg.num_heads,
        k=cfg.k_neighbors,
        num_positional_embeddings=cfg.num_positional_embeddings,
        num_rbf=cfg.num_rbf,
        augment_eps=cfg.augment_eps,
        backbone_diheral=cfg.backbone_diheral,
        dropout=cfg.dropout,
        denoiser=True,
        update_atom=cfg.update_atom,
        num_decoder_blocks=cfg.num_decoder_blocks,
        num_tfmr_heads=cfg.num_tfmr_heads,
        num_tfmr_layers=cfg.num_tfmr_layers,
        number_ligand_atom=cfg.number_ligand_atom,
        mpnn_cutoff=cfg.mpnn_cutoff,
    )
    return model.to(device)


def load_flow_model(ckpt_path: str, device: torch.device):
    """
    Load DiscreteFlow_AA model + EMA wrapper from checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    denoiser = build_denoiser(config, device)
    flow = DiscreteFlow_AA(config, denoiser,
                           min_t=0.0,
                           sidechain_packing=True,
                           sample_save_path='results/new_samples/')
    flow.load_state_dict(ckpt["model"])

    ema_flow = EMA(flow,
                   beta=config.training.ema_beta,
                   update_every=config.training.ema_update_every)
    ema_flow.load_state_dict(ckpt["ema"])
    return ema_flow.to(device), config


def main():
    parser = argparse.ArgumentParser(description="Run ADFLIP inverse folding")
    parser.add_argument("--pdb",          help="Path to input PDB/mmCIF file" , default="dataset/1g7g.pdb")
    parser.add_argument("--ckpt",       default="results/weights/ADFLIP_ICML_camera_ready.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device",     default="cuda:0", help="Torch device")
    parser.add_argument("--method",     choices=["fixed","adaptive"], default="adaptive",
                        help="Sampling scheme")
    parser.add_argument("--dt",         type=float, default=0.2,
                        help="Fixed-step dt if using fixed scheme")
    parser.add_argument("--steps",      type=int,   default=8,
                        help="Max steps for adaptive sampling")
    parser.add_argument("--threshold",  type=float, default=0.9,
                        help="Confidence threshold for adaptive sampling")
    args = parser.parse_args()

    device = torch.device(args.device)
    ema_flow, config = load_flow_model(args.ckpt, device)
    data = pdb2data(args.pdb, device)

    mask_center = data["is_center"] & data["is_protein"]
    true_tokens =  data["residue_token"][data["is_center"] & data["is_protein"]]

    # Run sampling
    if args.method == "fixed":
        samples, logits = ema_flow.ema_model.sample(args.pdb, dt=args.dt, argmax_final=True)
    else:
        samples, logits = ema_flow.ema_model.adaptive_sample(
            args.pdb,
            num_step=args.steps,
            threshold=args.threshold,
            argmax_final=False
        )
    samples = samples.squeeze()

    # Compute recovery rates
    rr = (samples == true_tokens).float().mean().item()
    interact_mol = data["interact_molecule_res"][data["is_center"][data["is_protein"]]].cpu()
    rr_mol = (samples[interact_mol] == true_tokens[interact_mol]).float().mean().item()

    print(f"Overall residue recovery rate:           {rr:.4f}")
    print(f"Interacting-residue recovery rate:        {rr_mol:.4f}")
    
    seq = ''.join(aap.restype_3to1[aap.index_to_token[i]] for i in samples.cpu().numpy())
    print(f"Generated sequence: {seq}")

if __name__ == "__main__":
    main()
