# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Portions of this file were adapted from the open source code for the following
# two papers:
#
#   Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative
#   models for graph-based protein design. Advances in Neural Information
#   Processing Systems, 32.
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
# 
# ================================================================
# The below license applies to the portions of the code (parts of 
# src/datasets.py and src/models.py) adapted from Ingraham, et al.
# ================================================================
# 
# MIT License
# 
# Copyright (c) 2019 John Ingraham, Vikas Garg, Regina Barzilay, Tommi Jaakkola
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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features


##UTILS

def flatten_graph(node_embeddings, edge_embeddings, edge_index):
    """
    Flattens the graph into a batch size one (with disconnected subgraphs for
    each example) to be compatible with pytorch-geometric package.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch size x edges x edge_embed_dim
                - vector: shape batch size x edges x edge_embed_dim x 3
        edge_index: shape batch_size x 2 (source node and target node) x edges
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch total_edges x edge_embed_dim
                - vector: shape batch total_edges x edge_embed_dim x 3
        edge_index: shape 2 x total_edges
    """
    x_s, x_v = node_embeddings
    e_s, e_v = edge_embeddings
    batch_size, N = x_s.shape[0], x_s.shape[1]
    node_embeddings = (torch.flatten(x_s, 0, 1), torch.flatten(x_v, 0, 1))
    edge_embeddings = (torch.flatten(e_s, 0, 1), torch.flatten(e_v, 0, 1))

    edge_mask = torch.any(edge_index != -1, dim=1)
    # Re-number the nodes by adding batch_idx * N to each batch
    edge_index = edge_index + (torch.arange(batch_size, device=edge_index.device) *
            N).unsqueeze(-1).unsqueeze(-1)
    edge_index = edge_index.permute(1, 0, 2).flatten(1, 2)
    edge_mask = edge_mask.flatten()
    edge_index = edge_index[:, edge_mask] 
    edge_embeddings = (
        edge_embeddings[0][edge_mask, :],
        edge_embeddings[1][edge_mask, :]
    )
    return node_embeddings, edge_embeddings, edge_index 


def unflatten_graph(node_embeddings, batch_size):
    """
    Unflattens node embeddings.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        batch_size: int
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
    """
    x_s, x_v = node_embeddings
    x_s = x_s.reshape(batch_size, -1, x_s.shape[1])
    x_v = x_v.reshape(batch_size, -1, x_v.shape[1], x_v.shape[2])
    return (x_s, x_v)

def tuple_size(tp):
    return tuple([0 if a is None else a.size() for a in tp])

def tuple_sum(tp1, tp2):
    s1, v1 = tp1
    s2, v2 = tp2
    if v2 is None and v2 is None:
        return (s1 + s2, None)
    return (s1 + s2, v1 + v2)

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    # clamp is slow
    # out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    out = torch.sum(torch.square(x), axis, keepdims) + eps
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)