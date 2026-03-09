import torch.utils.data
from torch_geometric.data import Data, Dataset
import torch
import pandas as pd
import pickle
from copy import deepcopy
import numpy as np


class EnzMolDataset(Dataset):
    def __init__(self, reaction_data, mol_obj, prot_obj):
        super(EnzMolDataset, self).__init__()
        self.pairs = reaction_data
        self.mols = mol_obj
        self.prots = prot_obj

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Extract data
        mol_key = self.pairs.loc[idx, 'Smile']
        prot_key = self.pairs.loc[idx, 'Pro_seq']
        try:
            reg_y = self.pairs.loc[idx, 'label']
            reg_y = torch.tensor(reg_y).float()
        except:
            reg_y =None

        # MOL
        mol = self.mols[mol_key]
        mol_x = mol['atom_idx'].long().view(-1, 1)
        mol_x_feat = mol['atom_feature'].float()
        mol_total_fea = mol["total_fea"].float()

        ## Enz
        prot = self.prots[prot_key]
        prot_seq = prot['seq']
        prot_node_prot5 = prot['token_representation']
        prot_node_esm = prot['token_representation_esm'].float()
        prot_num_nodes = len(prot['seq'])
        prot_node_pos = torch.arange(len(prot['seq'])).reshape(-1,1)
        prot_edge_index = prot['edge_index']
        prot_edge_weight = prot['edge_weight'].float()

        out = MultiGraphData(
                ## MOLECULE
                mol_x=mol_x, mol_x_feat=mol_x_feat, mol_total_fea=mol_total_fea,
                ## Enz
                prot_node_esm=prot_node_esm, prot_node_prot5=prot_node_prot5,prot_node_pos=prot_node_pos, prot_seq=prot_seq,
                prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,prot_num_nodes=prot_num_nodes,
                ## Y output,keys
                reg_y=reg_y, mol_key = mol_key, prot_key = prot_key)
        return out

def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here, 
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'prot_edge_index':
            return self.prot_node_esm.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_esm.size(0)
        elif key == 'm2p_edge_index':
             return torch.tensor([[self.mol_x.size(0)], [self.prot_node_esm.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

