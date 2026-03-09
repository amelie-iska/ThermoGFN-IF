# Contents of this file are from the open source code for
#
#   Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2020).
#   Learning from Protein Structure with Geometric Vector Perceptrons. In
#   International Conference on Learning Representations.
#
# MIT License
#
# Copyright (c) 2020 Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael Townshend, Ron Dror
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing

from fampnn.data import residue_constants as rc
from fampnn.data.data import (dihedrals, dist, get_rc_tensor,
                              get_rotation_frames, nan_to_num, normalize,
                              orientations, positional_embeddings, rbf, rotate,
                              sidechains)
from fampnn.model.gvp.gvp_utils import (_merge, _norm_no_nan, _split,
                                        flatten_graph, tuple_cat, tuple_index,
                                        tuple_sum)


class GVPEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_graph = GVPGraphEmbedding(cfg.graph_embedding)
        self.rotate_out = getattr(cfg, "rotate_out", False)
        self.node_hidden_dim_vector = cfg.node_hidden_dim_vector
        self.node_hidden_dim_scalar = cfg.node_hidden_dim_scalar

        node_hidden_dim = (cfg.node_hidden_dim_scalar,
                cfg.node_hidden_dim_vector)
        edge_hidden_dim = (cfg.edge_hidden_dim_scalar,
                cfg.edge_hidden_dim_vector)

        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=cfg.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                )
            for i in range(cfg.num_encoder_layers)
        )

        if not self.rotate_out:
            self.W_out = nn.Sequential(
                LayerNorm(node_hidden_dim),
                GVP(node_hidden_dim, (cfg.out_dim, 0)))
        else:
            flattened_hidden_dim = self.node_hidden_dim_scalar + 3 * self.node_hidden_dim_vector
            self.W_out = nn.Linear(flattened_hidden_dim, cfg.out_dim)


    def forward(self, coords, seq, E_idx, h_V, h_E, padding_mask, atom14_mask):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(coords, seq, E_idx, h_V, h_E, padding_mask, atom14_mask)

        for _, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        B, N  = coords.shape[0], coords.shape[1]

        if self.rotate_out:
            gvp_out_scalars, gvp_out_vectors = node_embeddings
            R = get_rotation_frames(coords)
            gvp_out_vectors = rotate(gvp_out_vectors.reshape(B, N, -1, 3), R.transpose(-2, -1))
            gvp_out_vectors = gvp_out_vectors.reshape(B * N, self.node_hidden_dim_vector, 3).flatten(-2, -1)
            node_embeddings = torch.cat([
                gvp_out_scalars,
                gvp_out_vectors,
            ], dim=-1)

        h_V = self.W_out(node_embeddings)
        return h_V

class GVPGraphEmbedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.top_k_neighbors = cfg.top_k_neighbors
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.remove_edges_without_coords = cfg.remove_edges_without_coords
        mpnn_n_channel = getattr(cfg, "mpnn_n_channel", cfg.node_hidden_dim_scalar)  # backwards compatibility
        self.model_type = getattr(cfg, "model_type", "sidechain")  # backwards compatibility
        node_input_dim = (6, 15) if self.model_type == 'sidechain' else (6, 3)
        edge_input_dim = (32, 56) if self.model_type == 'sidechain' else (32, 1)
        node_hidden_dim = (cfg.node_hidden_dim_scalar,
                cfg.node_hidden_dim_vector)
        edge_hidden_dim = (cfg.edge_hidden_dim_scalar,
                cfg.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )


        self.embed_mpnn_node = nn.Linear(mpnn_n_channel, cfg.node_hidden_dim_scalar)
        self.embed_mpnn_edge = nn.Linear(mpnn_n_channel * 4 if self.model_type == 'sidechain' else mpnn_n_channel * 3, cfg.edge_hidden_dim_scalar)
        self.zero_ghost_atoms = False

    def forward(self, coords, seq, mpnn_E_idx, mpnn_node_embedding, mpnn_edge_embedding, padding_mask, atom14_mask):
        with torch.no_grad():
            node_features = self.get_node_features(coords, padding_mask, atom14_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, padding_mask, mpnn_E_idx, atom14_mask)

        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        node_embeddings = (
            node_embeddings_scalar + self.embed_mpnn_node(mpnn_node_embedding),
            node_embeddings_vector
        )

        edge_embeddings = (
            edge_embeddings[0] + self.embed_mpnn_edge(mpnn_edge_embedding.flatten(1,2)),
            edge_embeddings[1]
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)

        return node_embeddings, edge_embeddings, edge_index

    def get_atom_type(self, seq, atom14_mask):
        atom_indices = get_rc_tensor(rc.RESTYPE_TO_ATOM37_IDX, seq)
        atom_indices = torch.where(atom_indices == -1, 0, atom_indices) #temporaily set ghost atom idx to 0
        atom_indices_one_hot = F.one_hot(atom_indices, num_classes=rc.atom_type_num).float()
        atom_indices_one_hot *= atom14_mask[..., None].expand_as(atom_indices_one_hot)
        atom_types_summed = torch.sum(atom_indices_one_hot, dim = -2)
        return atom_types_summed

    def get_node_features(self, coords, padding_mask, atom14_mask):
        # scalar features
        node_scalar_features = dihedrals(coords)

        # vector features
        X_ca = coords[:, :, 1]
        ca_orientations = orientations(X_ca)
        fa_orientations = self.intra_residue_orientations(coords, atom14_mask)

        #for residues w/out CB, overwrite with pseudo CB
        cb_orientations = sidechains(coords)
        no_cb_mask = atom14_mask[:, :, 4] == 0 #use atom14 mask to find positions with no cb
        no_cb_mask = torch.where(padding_mask, False, no_cb_mask) #exclude padded positions from getting pseudo cb
        fa_orientations[:,:,3][no_cb_mask, :] = cb_orientations[no_cb_mask]

        if self.model_type == 'baseline':
            node_vector_features = torch.cat([ca_orientations, cb_orientations.unsqueeze(-2)], dim=-2)
        else:
            node_vector_features = torch.cat([ca_orientations, fa_orientations], dim=-2)

        return node_scalar_features, node_vector_features

    def intra_residue_orientations(self, coords, atom14_mask):
        X_ca = coords[:, :, 1]
        vectors = []
        atom_positions = [0,2,3,4,5,6,7,8,9,10,11,12,13]

        for atom_pos in atom_positions:
            intra_residue_vector = normalize(X_ca - coords[:, :, atom_pos])
            vectors.append(intra_residue_vector)

        return torch.stack(vectors, dim=2)

    def get_edge_features(self, coords, padding_mask, E_idx, atom14_mask):
        X_ca = coords[:, :, 1]

        # Get distances to the top k neighbors, using E_idx from ProteinMPNN
        E_dist, E_residue_mask = dist(X_ca, E_idx, padding_mask)

        # Flatten the graph to be batch size 1 for torch_geometric package
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF
        pos_embeddings = positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)

        # Calculate relative orientation
        if self.model_type == 'baseline':
            X_ca = X_ca.reshape(-1, 3)
            E_vectors = (X_ca[edge_index[0]] - X_ca[edge_index[1]]).unsqueeze(2)
        else:
            E_vectors = self.get_edge_vectors(coords, E_idx, edge_index, B, L, k, atom14_mask)

        # Normalize and remove nans
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        edge_index[:, ~E_residue_mask] = -1

        return (edge_s, edge_v), edge_index.transpose(0, 1)

    def get_edge_vectors(self, coords, E_idx, edge_index, B, L, k, atom14_mask):
        max_atoms = coords.shape[-2] #14
        X_n = coords[:, :, 0]
        X_ca = coords[:, :, 1]
        X_c = coords[:, :, 2]
        X_o = coords[:, :, 3]

        vectors = []

        for bb_atom in [X_ca, X_n, X_c, X_o]:
            for atom_pos in range(max_atoms):
                relative_orientation_vector = normalize(self.get_relative_orientation(bb_atom, coords[:,:,atom_pos], edge_index, B, L, k))
                vectors.append(relative_orientation_vector)

        return torch.stack(vectors, dim=2)

    def get_relative_orientation(self, X, Y, edge_index, B, L, k):
        X_src = X.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            Y,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )

        return X_src - X_dest

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        if x is None:
            return None
        device = x.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims, tuple_io=True, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.tuple_io = tuple_io
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.eps = eps

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            if self.tuple_io:
                return self.scalar_norm(x[0]), None
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = (vn > 2 * self.eps)
        vn = torch.sum(vn * nonzero_mask, dim=-2, keepdim=True
            ) / (self.eps + torch.sum(nonzero_mask, dim=-2, keepdim=True))
        vn = torch.sqrt(vn + self.eps)
        v = nonzero_mask * (v / vn)
        return self.scalar_norm(s), v

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture
    '''
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3,
            vector_gate=False, module_list=None, aggr="mean", eps=1e-8,
            activations=(F.relu, torch.sigmoid)):
        super(GVPConv, self).__init__(aggr=aggr)
        self.eps = eps
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP((2*self.si + self.se, 2*self.vi + self.ve),
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP((2*self.si + self.se, 2*self.vi + self.ve), out_dims,
                        vector_gate=vector_gate, activations=activations)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP(out_dims, out_dims,
                        vector_gate=vector_gate))
                module_list.append(GVP(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    '''
    def __init__(self, node_dims, edge_dims, vector_gate=False,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, attention_heads=0,
                 conv_activations=(F.relu, torch.sigmoid),
                 n_edge_gvps=0, layernorm=True, eps=1e-8):

        super(GVPConvLayer, self).__init__()
        if attention_heads == 0:
            self.conv = GVPConv(
                    node_dims, node_dims, edge_dims, n_layers=n_message,
                    vector_gate=vector_gate,
                    aggr="add" if autoregressive else "mean",
                    activations=conv_activations,
                    eps=eps,
            )
        else:
            raise NotImplementedError
        if layernorm:
            self.norm = nn.ModuleList([LayerNorm(node_dims, eps=eps) for _ in range(2)])
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP(node_dims, hid_dims, vector_gate=vector_gate))
            for i in range(n_feedforward-2):
                ff_func.append(GVP(hid_dims, hid_dims, vector_gate=vector_gate))
            ff_func.append(GVP(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

        self.edge_message_func = None
        if n_edge_gvps > 0:
            si, vi = node_dims
            se, ve = edge_dims
            module_list = [
                GVP((2*si + se, 2*vi + ve), edge_dims, vector_gate=vector_gate)
            ]
            for i in range(n_edge_gvps - 2):
                module_list.append(GVP(edge_dims, edge_dims,
                    vector_gate=vector_gate))
            if n_edge_gvps > 1:
                module_list.append(GVP(edge_dims, edge_dims,
                    activations=(None, None)))
            self.edge_message_func = nn.Sequential(*module_list)
            if layernorm:
                self.edge_norm = LayerNorm(edge_dims, eps=eps)
            else:
                self.edge_norm = nn.Identity()
            self.edge_dropout = Dropout(drop_rate)

    def forward(self, x, edge_index, edge_attr,
                    autoregressive_x=None, node_mask=None):
            '''
            :param x: tuple (s, V) of `torch.Tensor`
            :param edge_index: array of shape [2, n_edges]
            :param edge_attr: tuple (s, V) of `torch.Tensor`
            :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                    If not `None`, will be used as srcqq node embeddings
                    for forming messages where src >= dst. The corrent node
                    embeddings `x` will still be the base of the update and the
                    pointwise feedforward.
            :param node_mask: array of type `bool` to index into the first
                    dim of node embeddings (s, V). If not `None`, only
                    these nodes will be updated.
            '''
            if self.edge_message_func:
                src, dst = edge_index
                if autoregressive_x is None:
                    x_src = x[0][src], x[1][src]
                else:
                    mask = (src < dst).unsqueeze(-1)
                    x_src = (
                        torch.where(mask, x[0][src], autoregressive_x[0][src]),
                        torch.where(mask.unsqueeze(-1), x[1][src],
                            autoregressive_x[1][src])
                    )
                x_dst = x[0][dst], x[1][dst]
                x_edge = (
                    torch.cat([x_src[0], edge_attr[0], x_dst[0]], dim=-1),
                    torch.cat([x_src[1], edge_attr[1], x_dst[1]], dim=-2)
                )
                edge_attr_dh = self.edge_message_func(x_edge)
                edge_attr = self.edge_norm(tuple_sum(edge_attr,
                    self.edge_dropout(edge_attr_dh)))

            dh = self.conv(x, edge_index, edge_attr)

            if node_mask is not None:
                x_ = x
                x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

            x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

            if node_mask is not None:
                x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
                x = x_

            return x, edge_attr