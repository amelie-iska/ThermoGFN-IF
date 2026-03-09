import torch
import glob
import os
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
import random
import pickle


from pathlib import Path
from typing import Dict, List, Literal

from data.all_atom_parse import residue_tokens
from data import all_atom_parse as aap
from data.all_atom_parse import StructureData
from data.utils import TimeSortedCacheRBT


Partition = Literal["train", "valid", "test"]

def propagate_mask_vectorized(mask, index):
    # Ensure inputs are tensors
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)
    
    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
    
    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)
    
    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce='amax')
    
    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]




class AllAtomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
    ):

        self.max_num_residues = cfg.max_num_residues
        self.max_num_atoms = cfg.max_num_atoms
        self.dataset_type   = dataset_type
        self.cut_position_type = cfg.cut_position_type
        self.random_gen = np.random.default_rng(cfg.random_seed)

        paths = cfg.data_path

        self.structs = (
            glob.glob(f"{paths}/*.pdb")
            + glob.glob(f"{paths}/*.cif")
            + glob.glob(f"{paths}/*.npz")
            + glob.glob(f"{paths}/*.cif.gz")
        )
        if len(self.structs) == 0:
            self.structs = (
                glob.glob(f"{paths}/**/*.pdb")
                + glob.glob(f"{paths}/**/*.cif")
                + glob.glob(f"{paths}/**/*.npz")
                + glob.glob(f"{paths}/**/*.cif.gz")
            )
        self.test_time = test_time
        self.use_cache = use_cache

        if self.use_cache:
            self.cache_data = TimeSortedCacheRBT(
                capacity=cache_length, parse_fn=aap.parse_or_load_mmcif
            )

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx, mask_chain_id = None) -> aap.StructureData:
        try:
            if self.use_cache:
                data = self.cache_data.get(self.structs[idx])
            else:
                data = aap.parse_or_load_mmcif(self.structs[idx])
            if isinstance(data, tuple):
                data = data[1]
            num_residues = data.num_residues()
        except Exception as e:
            print(f"Failed to parse {self.structs[idx]}")
            return None

        if num_residues > self.max_num_residues:
            # Here can do all kinds of complicated logic about finding ligands
            # that are closest to protein residues etc
            # For now, just pick a random protein residue
            if self.cut_position_type == "protein":
                random_position = self.random_gen.choice(
                    data.position[data.is_protein], axis=0
                )
            elif self.cut_position_type == "ligand":
                if data.is_protein.sum() == data.is_protein.shape[0]: # no ligand
                    random_position = self.random_gen.choice(
                        data.position, axis=0
                    )
                    # print("No ligand found, path", self.structs[idx])
                else:
                    random_position = self.random_gen.choice(
                    data.position[~data.is_protein], axis=0
                )

            data = aap.get_closest_n_residues(
                data, random_position, n=self.max_num_residues
            )

        if len(data) > self.max_num_atoms:
            if data.is_protein.sum() == 0:
                print("No protein found, path", self.structs[idx])
                return None
            else:
                random_position = self.random_gen.choice(
                    data.position[data.is_protein], axis=0
                )
                data = aap.get_closest_n_atoms(
                    data,
                    random_position,
                    n=self.max_num_atoms,
                )

        if data.is_protein.sum() == 0:
            print("No protein found, path", self.structs[idx])
            return None
        data = self.interact_residue(data)

        if self.dataset_type == "train":
            return self.corrupt(data,mask_chain_id=mask_chain_id)
        else:
            return self.corrupt(data, time=np.array([self.test_time]),mask_chain_id=mask_chain_id)

    def corrupt(self, data: aap.StructureData, time=None,mask_chain_id = None) -> aap.StructureData:
        # Corrupt the data in some way
        batch_dict = data.__dict__

        mask_chain = np.zeros_like(batch_dict['chain_id'],dtype=bool) # True -> mask chain, False -> visible chain

        if mask_chain_id is not None:

            # make sure mask_chain_id is an array (works whether it’s a scalar or a list)
            mask_chain_id = np.atleast_1d(mask_chain_id)

            # True for positions where the chain_id is 0 or 1 (in your example)
            to_mask = np.isin(batch_dict['chain_id'], mask_chain_id)

            mask_chain[to_mask] = True # True -> mask chain, False -> visible chain
            
        num_residue_tokens = batch_dict["residue_index"].max() + 1
        if time is not None:
            time_step = time
        else:
            time_step = np.random.uniform(0, 1, 1)
        u = np.random.uniform(0, 1, num_residue_tokens)
        target_mask = u < (1.0 - time_step)  # True -> mask
        target_mask[~mask_chain[batch_dict['is_center']]] = False #unmask visble chain

        noisy_residue_token = np.copy(batch_dict["residue_token"])
        noisy_residue_token[
            target_mask[batch_dict["residue_index"]] & batch_dict["is_protein"]
        ] = residue_tokens["<MASK>"] #only change protein
        batch_dict["noisy_residue_token"] = noisy_residue_token


        #check visible chain and mask chain
        print("mask_chain_id",mask_chain_id,'time_step',time_step)
        print('mask ratio in visble chain', target_mask[~mask_chain[batch_dict['is_center']]].sum() / target_mask[~mask_chain[batch_dict['is_center']]].shape[0])
        print('mask ratio in mask chain', target_mask[mask_chain[batch_dict['is_center']]].sum() / target_mask[mask_chain[batch_dict['is_center']]].shape[0])

        mask_sidechain = np.ones_like(batch_dict["residue_token"]).astype(bool)
        mask_sidechain[~batch_dict["is_backbone"] & batch_dict["is_protein"]] = (
            ~target_mask[
                batch_dict["residue_index"][
                    ~batch_dict["is_backbone"] & batch_dict["is_protein"]
                ]
            ]
        )

        for name, data in batch_dict.items():
            if name == "noisy_residue_token":
                batch_dict[name] = noisy_residue_token[mask_sidechain]
            elif name == "time_step":
                batch_dict[name] = time_step
            else:
                batch_dict[name] = data[mask_sidechain]

        return StructureData(**batch_dict)

    def interact_residue(self,data):
        protein_pos = np.expand_dims(data.position[data.is_protein],axis=1)
        non_protein_pos = np.expand_dims(data.position[~data.is_protein],axis=0)
        ion_pos = np.expand_dims(data.position[data.is_ion],axis=0)
        nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide],axis=0)
        molecule_pos = np.expand_dims(data.position[~data.is_protein&~data.is_ion&~data.is_nucleotide],axis=0)

        dist_non_protein = np.sqrt((np.sum((protein_pos-non_protein_pos)**2,axis=-1)))
        interact_non_protein = np.any(dist_non_protein<5,axis=1)
        dist_ion = np.sqrt((np.sum((protein_pos-ion_pos)**2,axis=-1)))
        interact_ion = np.any(dist_ion<5,axis=1)
        dist_nucleotide = np.sqrt((np.sum((protein_pos-nucleotide_pos)**2,axis=-1)))
        interact_nucleotide = np.any(dist_nucleotide<5,axis=1)
        dist_molecule = np.sqrt((np.sum((protein_pos-molecule_pos)**2,axis=-1)))
        interact_molecule = np.any(dist_molecule<5,axis=1)


        interact_non_protein_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_non_protein_res[data.is_protein] = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
        interact_ion_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_ion_res[data.is_protein] = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
        interact_nucleotide_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_nucleotide_res[data.is_protein] = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
        interact_molecule_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_molecule_res[data.is_protein] = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


        data.interact_non_protein_res = interact_non_protein_res
        data.interact_ion_res = interact_ion_res
        data.interact_nucleotide_res = interact_nucleotide_res
        data.interact_molecule_res = interact_molecule_res

        return data


def _load_cluster_ids(path: Path | str) -> List[int]:
    """Return a list of integer IDs, one per line, from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f]

class Cluster_AllAtomDataset(AllAtomDataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
    ):
        super().__init__(cfg, dataset_type, use_cache, cache_length, test_time)
        self.cfg = cfg
        assert dataset_type in ["train", "valid", "test"], f"Invalid dataset type: {dataset_type}"
        with open(cfg.cluster_path, "rb") as f:
            self.cluster_dict = pickle.load(f)

        id_lists: Dict[Partition, List[int]] = {
            "train": _load_cluster_ids(cfg.train_cluster),
            "valid": _load_cluster_ids(cfg.validation_cluster),
            "test":  _load_cluster_ids(cfg.test_cluster),
        }        
        self.sel_keys = id_lists[dataset_type]
        self.select_cluster = {
            idx: value
            for idx, (key, value) in enumerate(self.cluster_dict.items())
            if key in self.sel_keys
        }

        self.all_pdb = []
        for _, v in self.select_cluster.items():
            #v  17: ['5jog_0', '5joh_0', '5m5q_0', '4f7o_1', '4f7o_0'], 18: ['6jo8_2', '6ort_0', '6nk3_1', '6jo7_0', '6nk3_0']
            v_ = list(set([x[:4] for x in v]))
            self.all_pdb.extend(v_)
        if self.use_cache:
            self.structs = [
                os.path.join(
                    cfg.data_path,
                    x[1:3],
                    x + ".npz"
                ) for x in self.all_pdb
            ]
        else:
            self.structs = [
                os.path.join(
                    cfg.data_path,
                    x+ ".cif.gz"
                ) for x in self.all_pdb
            ]
        print(f"Dataset {dataset_type} contains {len(self.structs)} structures, {len(set(self.structs))} unique PDB IDs")   
    def __len__(self) -> int:
        return len(self.select_cluster.keys())

    def __getitem__(self, idx) -> StructureData:
        cluster = self.select_cluster[self.sel_keys[idx]]
        sample = random.choice(cluster) 
        # check if sample in this cluster has same pdb id
        sel_pdb = sample[:4]
        cluster_pdb = [ x[:4] for x in cluster]
        if cluster_pdb.count(sel_pdb) > 1: # 
            sample_list = [x for x in cluster if x[:4] == sel_pdb]
            mask_chain_id = [int(x[5]) for x in sample_list]
        else:
            mask_chain_id = [int(sample[5])]
        if self.use_cache:
            sample_pdb_path = os.path.join(self.cfg.data_path,sample[1:3],sel_pdb + ".npz")
        else:
            sample_pdb_path = os.path.join(''.join(self.structs[0].split('/')[:-2]),sample[1:3],sel_pdb + ".cif.gz")
        index = self.structs.index(sample_pdb_path)
        return super().__getitem__(index,mask_chain_id=mask_chain_id)


if __name__ == "__main__":

    # class Config:
    #     data_path = "/ceph/scheres_grp/kyi/Documents/dataset/ligand_pdb/train_parsed"
    #     max_num_atoms = 4096
    #     max_num_residues = 512
    #     random_seed = 42
    #     cut_position_type = "ligand"
    #     use_cathe = True

    # dataset = AllAtomDataset(Config())


    class Config:
        data_path = "/ssd/dataset/pdb/parser_data/"
        cluster_path = '/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/cluster/cluster_to_pdb_chain.pkl'
        train_cluster = '/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/cluster/train_clusters.txt'
        validation_cluster = '/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/cluster/valid_clusters.txt'
        test_cluster = '/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/cluster/test_clusters.txt'
        max_num_atoms = 4096
        max_num_residues = 512
        random_seed = 42
        cut_position_type = "ligand"
        use_cathe = True

    # dataset = Cluster_AllAtomDataset(Config(),dataset_type = 'train')
    # print(len(dataset))
    # dataset = Cluster_AllAtomDataset(Config(),dataset_type = 'valid')
    # print(len(dataset))
    dataset = Cluster_AllAtomDataset(Config(),dataset_type = 'train')
    # print(len(dataset))
    for i in range(10):
        print(dataset[i])
    # print(dataset[0])


    # for i in range(10):
    #     print(dataset[i])


