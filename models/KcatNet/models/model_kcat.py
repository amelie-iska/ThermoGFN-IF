import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool
from torch.nn import Embedding, Linear
import torch.nn.functional as F
from torch_geometric.utils import  to_dense_adj, to_dense_batch, softmax
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from models.layers import  Protein_PNAConv, InterConv, PosLinear, GCNCluster
from models.Mol_pool import MotifPool
from models.protein_pool import dense_mincut_pool

from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv, APPNP, SGConv

EPS = 1e-15
class KcatNet(torch.nn.Module):
    def __init__(self, prot_deg,
                 # MOLECULE
                 mol_in_channels=43, prot_in_channels=40, prot_evo_channels=1280,
                 hidden_channels=200, pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,
                 K=[10, 15, 20],
                 t=1,
                 # training
                 heads=5,
                 dropout=0,
                 dropout_attn_score=0.2,
                 drop_atom=0,
                 device='cuda:0'): #drop_residue=0,dropout_cluster_edge=0,gaussian_noise=0,

        super(KcatNet, self).__init__()

        self.hidden_channels = hidden_channels
        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, hidden_channels)

        ### MOLECULE and PROTEIN
        self.prot_convs = torch.nn.ModuleList()
        self.atom_update = torch.nn.ModuleList()
        self.inter_convs = torch.nn.ModuleList()
        self.num_cluster = K
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.res_update = torch.nn.ModuleList()
        self.mol_update = torch.nn.ModuleList()
        self.atom_embed_total = torch.nn.ModuleList()
        self.atom_embed_total2 = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.prot_convs.append(Protein_PNAConv(prot_deg, hidden_channels, edge_channels=hidden_channels, pre_layers=pre_layers, post_layers=post_layers,
                                                   aggregators=aggregators, scalers=scalers, num_towers=heads, dropout=dropout))
            self.cluster.append(GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]]))

            self.atom_update.append(Linear(hidden_channels, hidden_channels))
            self.mol_pools.append(MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom))
            self.atom_embed_total.append(Linear(hidden_channels * 2, hidden_channels))
            self.atom_embed_total2.append(Linear(hidden_channels, hidden_channels))

            self.inter_convs.append(InterConv(atom_channels=hidden_channels, residue_channels=hidden_channels, heads=heads, t=t,dropout_attn_score=dropout_attn_score))

            self.res_update.append(Linear(hidden_channels, hidden_channels))
            self.mol_update.append(Linear(hidden_channels, hidden_channels))



        self.dropout = dropout
        self.device = device

        self.seq_embed_esm = torch.nn.Linear(prot_in_channels, hidden_channels*2)
        self.seq_embed_prot5 = torch.nn.Linear(prot_evo_channels, hidden_channels*2)
        self.seq_embed = torch.nn.Linear(hidden_channels*4, hidden_channels)
        self.seq_embed_evo2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)

        self.atom_feat_embed = Linear(mol_in_channels, hidden_channels)
        self.atom_type_embed = Embedding(20, hidden_channels)
        self.atom_type_embed2 = Linear(hidden_channels // 2, hidden_channels // 2)
        self.atom_feat_embed2 = Linear(hidden_channels//2, hidden_channels)
        self.mol_embed= torch.nn.Linear(1024,hidden_channels)
        self.mol_embed2 = torch.nn.Linear(hidden_channels, hidden_channels)


        self.norm = torch.nn.LayerNorm(hidden_channels) #
        self.GN = GraphNorm(hidden_channels)


        self.inter_attn_lin = PosLinear(heads, 1, bias=False,init_value=1 / heads)  # (heads * total_layer)) PositiveLinear(heads, 1, bias=False)#
        self.inter_attn_lin2 = PosLinear(heads, 1, bias=False,init_value=1 / heads) #PositiveLinear(heads, 1, bias=False)#

        self.mol_fea_update = Linear(hidden_channels * total_layer, hidden_channels)
        self.res_fea_update = Linear(hidden_channels * total_layer, hidden_channels)
        self.res_fea_update2 = Linear(hidden_channels * total_layer, hidden_channels)
        self.cluster_fea_update2 = Linear(hidden_channels * total_layer, hidden_channels)

        self.classifier = nn.Linear(hidden_channels * 4, 512)  # 1024
        self.classifier1 = nn.Linear(512, 128)
        self.classifier2 = nn.Linear(128, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)


    def forward(self, mol_x, mol_x_feat,mol_total_fea,
                residue_esm, residue_prot5, residue_edge_index, residue_edge_weight,
                mol_batch=None, prot_batch=None):

        cluster_loss = torch.tensor(0.).to(self.device)
        residue_scores = []

        residue_ini = torch.cat([F.relu(self.seq_embed_prot5 (residue_prot5)), F.relu(self.seq_embed_esm(residue_esm))], dim=-1)
        residue_x = F.relu(self.seq_embed(residue_ini))
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)

        atom_x = self.atom_type_embed(mol_x.squeeze())+ F.relu(self.atom_feat_embed(mol_x_feat))
        mol_total_fea = F.relu(self.mol_embed(mol_total_fea))
        mol_total_fea = self.norm(self.mol_embed2(mol_total_fea))

        res_feas = []
        res_feas2 = []
        mol_feas = []
        cluster_feas = []

        for idx in range(self.total_layer):
            residue_x = self.GN(residue_x, prot_batch)
            atom_x = self.GN(atom_x, mol_batch)

            # cluster residues
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            residue_max = global_max_pool(residue_x, prot_batch) #(32,200)
            res_feas.append(residue_max)

            s = self.cluster[idx](residue_x, residue_edge_index)  # (12329,cluster_num) GCN聚类
            s, _ = to_dense_batch(s, prot_batch)  # (32,903,cluster_num)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)  ##(32,903,200)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)  # (32,903,903)
            s, cluster_x, residue_adj, cl_loss, _ = dense_mincut_pool(residue_hx, residue_adj, s, residue_mask, None)
            cluster_x = self.norm(cluster_x)
            cluster_loss += cl_loss


            atom_x = F.relu(self.atom_update[idx](atom_x))#F.leaky_relu()
            mol_x, _ = self.mol_pools[idx](atom_x, mol_batch) #(32,200)
            mol_x = self.norm(mol_x)

            mol_x = torch.cat([mol_total_fea, mol_x], dim=-1)
            mol_x = F.relu(self.atom_embed_total[idx](mol_x))
            mol_x = self.atom_embed_total2[idx](mol_x)  # )
            mol_x = self.norm(mol_x)


            # connect molecule and protein cluster
            batch_size = s.size(0)
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)  # cluster_x(160,200)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device) #(0001112)
            p2m_edge_index = torch.stack([torch.arange(batch_size * self.num_cluster[idx]),torch.arange(batch_size).repeat_interleave(self.num_cluster[idx])]).to(self.device)

            ## model interative relationship
            mol_x, cluster_x, inter_attn = self.inter_convs[idx](mol_x, cluster_x, p2m_edge_index)  # mol_x(32,200), cluster_x(192,200), inter_attn((2,B*5),(B*5,5)) #mol_x太小而cluster太大
            mol_feas.append(mol_x)
            inter_attn = inter_attn[1] #(B*cluster_num,heads_num) #pan0715 inter_attn已经不对了
            #print("33333",inter_attn)


            atom_x = atom_x + F.relu(self.mol_update[idx](mol_x)[mol_batch])
            cluster_score = softmax(self.inter_attn_lin(inter_attn), cluster_residue_batch)
            pool_cluster = self.norm(global_max_pool(cluster_x * cluster_score, cluster_residue_batch))

            cluster_feas.append(pool_cluster)

            cluster_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)  # (32,3,200)
            inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)  # (B,3,head5)

            residue_x = residue_x + F.relu((self.res_update[idx]((s @ cluster_hx)[residue_mask]))) #self.norm() # cluster -> residue (429,200)

            residue_score = self.inter_attn_lin2( (s @ inter_attn)[residue_mask])
            residue_score = softmax(residue_score, prot_batch) #
            residue_scores.append(residue_score)

            pool_enz = self.norm(global_max_pool(residue_x * residue_score, prot_batch))
            res_feas2.append(pool_enz)

        mol_feas = torch.cat(mol_feas, dim=-1)
        res_feas = torch.cat(res_feas, dim=-1)
        res_feas2 = torch.cat(res_feas2, dim=-1)
        clu_fea = torch.cat(cluster_feas,dim=-1)

        mol_x = F.relu(self.mol_fea_update(mol_feas))#F.leaky_relu()
        res_feas = F.relu(self.res_fea_update(res_feas))
        res_feas2 = F.relu(self.res_fea_update2(res_feas2))  # F.leaky_relu()
        clu_fea = F.relu(self.cluster_fea_update2(clu_fea))  # F.leaky_relu()
        mol_prot_feat = torch.cat([res_feas, res_feas2, clu_fea,mol_x], dim=-1)  #clu_fea

        reg_pred = F.relu((self.classifier(mol_prot_feat)))
        reg_pred = F.relu((self.classifier1(reg_pred)))
        reg_pred = self.classifier2(reg_pred)

        return reg_pred, cluster_loss








def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    Returns an 径向基函数 Radial Basis Function - RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device))
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


