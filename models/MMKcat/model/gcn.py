#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-03-09 9:18
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : gcn.py
# @Software: PyCharm


import torch
from torch import Tensor
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super().__init__()

        self.device = device

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long).to(self.device))
        return x
