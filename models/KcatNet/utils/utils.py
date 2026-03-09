import os
import numpy as np
import sys
from scipy import stats

# Check if the code is running in a Jupyter notebook

from tqdm import tqdm

    
from itertools import repeat
import pandas as pd 


import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from utils.metrics import evaluate_reg

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def create_custom_loader(type='epoch'):
    if type == 'epoch':
        return DataLoader
    elif type =='infinite':
        return InfiniteDataLoader
    else:
        raise Exception('Not Implemented')
        
class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def sampler_from_weights(weights):
    sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler 
def create_custom_sampler(class_list, specified_weight={}):
    assert isinstance(specified_weight,dict)
    class_list = np.array(class_list)
    class_weight = {
         t: 1./len(np.where(class_list == t)[0]) for t in np.unique(class_list)
    }
    
    samples_weight = np.array([class_weight[t] for t in class_list])
    
    if specified_weight:
        specified_weight = np.array([specified_weight[i] for i in class_list])
        samples_weight *= specified_weight
        
    sampler = CustomWeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler 

def compute_pna_degrees(train_loader):
    prot_max_degree = -1
    for data in tqdm(train_loader):
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_esm.shape[0], dtype=torch.long)
        prot_max_degree = max(prot_max_degree, int(prot_d.max()))

    # Compute the in-degree histogram tensor
    prot_deg = torch.zeros(prot_max_degree + 1, dtype=torch.long)

    for data in tqdm(train_loader):
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_esm.shape[0], dtype=torch.long)
        prot_deg += torch.bincount(prot_d, minlength=prot_deg.numel())
    return  prot_deg


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_nodes(data_tensor, index_tensor):
    """
    Unbatch a data tensor based on an index tensor.

    Args:
    data_tensor (torch.Tensor): The tensor to be unbatched.
    index_tensor (torch.Tensor): A tensor of the same length as data_tensor's first dimension, 
                                 indicating the batch index for each element in data_tensor.

    Returns:
    list[torch.Tensor]: A list of tensors, where each tensor corresponds to a separate batch.
    """
    return [data_tensor[index_tensor == i] for i in index_tensor.unique()]


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def printline(line):
    sys.stdout.write(line + "\x1b[K\r")
    sys.stdout.flush()


def protein_degree_from_dict(protein_dict):
    protein_max_degree = -1 
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_max_degree = max(protein_max_degree, protein_degree.max())

    protein_deg = torch.zeros(protein_max_degree + 1, dtype=torch.long)
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_deg += torch.bincount(protein_degree, minlength=protein_deg.numel())   

    return protein_deg


def minmax_norm(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def percentile_rank(arr):
    return np.argsort(np.argsort(arr)) / (len(arr)-1) 

from rdkit import Chem
from rdkit.Chem import PropertyPickleOptions
import pickle
        
def store_ligand_score(ligand_smiles, atom_types, atom_scores, ligand_path):
    # Create a molecule from a SMILES string
    mol = Chem.MolFromSmiles(ligand_smiles)
    # Add an atom-level property to the first atom
    for i, atom in enumerate(mol.GetAtoms()):
        
        if atom_types[i] == atom.GetSymbol():
            atom.SetProp("PSICHIC_Atom_Score", str(atom_scores[i]))
        else:
            return False
    # Configure RDKit to pickle all properties
    Chem.SetDefaultPickleProperties(PropertyPickleOptions.AllProps)

    # Serialize molecule to a pickle file
    with open(ligand_path, 'wb') as f:
        pickle.dump(mol, f)

    return True
    
def store_result(df, attention_dict, interaction_keys,  ligand_dict, 
                 reg_pred=None, cls_pred=None, mcls_pred=None, 
                 result_path='', save_interpret=True):
    if save_interpret:
        unbatched_residue_score = unbatch(attention_dict['residue_final_score'],attention_dict['protein_residue_index'])
        unbatched_atom_score = unbatch(attention_dict['atom_final_score'], attention_dict['drug_atom_index'])
        unbatched_residue_layer_score = unbatch(attention_dict['residue_layer_scores'],attention_dict['protein_residue_index'])
        unbatched_clique_layer_score = unbatch(attention_dict['clique_layer_scores'], attention_dict['drug_clique_index'])
    
    for idx, key in enumerate(interaction_keys):
        matching_row = (df['Protein'] == key[0]) & (df['Ligand'] == key[1])
        if reg_pred is not None:
            if 'predicted_binding_affinity' in df.columns:
                df.loc[matching_row, 'predicted_binding_affinity'] = reg_pred[idx]
            else:
                df['predicted_binding_affinity'] = None
                df.loc[matching_row, 'predicted_binding_affinity'] = reg_pred[idx]
        if cls_pred is not None:
            if 'predicted_binary_interaction' in df.columns:
                df.loc[matching_row, 'predicted_binary_interaction'] = cls_pred[idx]
            else:
                df['predicted_binary_interaction'] = None
                df.loc[matching_row, 'predicted_binary_interaction'] = cls_pred[idx]

        if mcls_pred is not None:
            if 'predicted_antagonist' in df.columns and 'predicted_nonbinder' in df.columns and 'predicted_agonist' in df.columns:
                df.loc[matching_row, ['predicted_antagonist','predicted_nonbinder','predicted_agonist']] = mcls_pred[idx].tolist()
            else:
                df['predicted_antagonist'] = None
                df['predicted_nonbinder'] = None
                df['predicted_agonist'] = None
                df.loc[matching_row, ['predicted_antagonist','predicted_nonbinder','predicted_agonist']] = mcls_pred[idx].tolist()
        

        if all([idx in attention_dict['cluster_s'] for idx in range(3)]):
            unbatched_cluster_s0 = unbatch_nodes(attention_dict['cluster_s'][0].softmax(dim=-1), attention_dict['protein_residue_index'])
            unbatched_cluster_s1 = unbatch_nodes(attention_dict['cluster_s'][1].softmax(dim=-1), attention_dict['protein_residue_index'])
            unbatched_cluster_s2 = unbatch_nodes(attention_dict['cluster_s'][2].softmax(dim=-1), attention_dict['protein_residue_index'])

        if save_interpret:
            for pair_id in df[matching_row]['ID']:
                pair_path = os.path.join(result_path,pair_id)
                if not os.path.exists(pair_path):
                    os.makedirs(pair_path)
                ## STORE Protein Interpretation
                protein_interpret = pd.DataFrame({
                    'Residue_Type':list(key[0]),
                    'PSICHIC_Residue_Score':minmax_norm(unbatched_residue_score[idx].cpu().flatten().numpy())
                    })

                protein_interpret['Residue_ID'] = protein_interpret.index + 1
                protein_interpret['PSICHIC_Residue_Percentile'] = percentile_rank(protein_interpret['PSICHIC_Residue_Score'])
                protein_interpret = protein_interpret[['Residue_ID','Residue_Type','PSICHIC_Residue_Score','PSICHIC_Residue_Percentile']]

                if all([id_ in attention_dict['cluster_s'] for id_ in range(3)]):
                    for ci in range(5):
                        protein_interpret['Layer0_Cluster'+str(ci)] = unbatched_cluster_s0[idx][:,ci].cpu().flatten().numpy()

                    for ci in range(10):
                        protein_interpret['Layer1_Cluster'+str(ci)] = unbatched_cluster_s1[idx][:,ci].cpu().flatten().numpy()
                    
                    for ci in range(20):
                        protein_interpret['Layer2_Cluster'+str(ci)] = unbatched_cluster_s2[idx][:,ci].cpu().flatten().numpy()

                protein_interpret.to_csv(os.path.join(pair_path,'protein.csv'),index=False)

                ## STORE Ligand Interpretation
                ligand_path = os.path.join(pair_path,'ligand.pkl')

                successful_ligand = store_ligand_score(key[1], ligand_dict[key[1]]['atom_types'].split('|'), 
                                                       minmax_norm(unbatched_atom_score[idx].cpu().flatten().numpy()),
                                                       ligand_path)
                if not successful_ligand:
                    print('Ligand Intepretation for {} failed due to not matching atom order.'.format(pair_id))
                ## STORE Fingerprint
                np.save(os.path.join(pair_path,'fingerprint.npy'),
                        attention_dict['interaction_fingerprint'][idx].detach().cpu().numpy()
                        )
    return df

def virtual_screening( model, data_loader, device='cpu'):
    """if "ID" in screen_df.columns:
        # Iterate through the DataFrame check any empty pairs
        for i, row in screen_df.iterrows():
            if pd.isna(row['ID']):
                screen_df.at[i, 'ID'] = f"PAIR_{i}"
    else:
        screen_df['ID'] = 'PAIR_'
        screen_df['ID'] += screen_df.index.astype(str)"""
    reg_preds = []
    reg_truths = []
    model.eval()
    count=0
    with torch.no_grad():
        for data in tqdm(data_loader):
            count+=1
            data = data.to(device)
            reg_pred, cl_loss = model(
                    # Molecule
                    mol_x=data.mol_x, mol_x_feat=data.mol_x_feat,mol_total_fea=data.mol_total_fea,
                    # Enz
                residue_esm=data.prot_node_esm, residue_prot5=data.prot_node_prot5,
                residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                # Mol-Enzyme Interaction batch
                mol_batch=data.mol_x_batch, prot_batch=data.prot_node_esm_batch)


            reg_pred = reg_pred.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_preds += reg_pred
            reg_y = data.reg_y.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_truths += reg_y

        if len(reg_truths) > 0:
            eval_reg_result = evaluate_reg(np.array(reg_truths), np.array(reg_preds))

    return reg_preds, reg_truths, eval_reg_result

