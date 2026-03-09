#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-01-20 23:56
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : feature_transformer.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4):
        super(PositionalEncoding, self).__init__()

        self.position = nn.Parameter(torch.randn(max_len, 1, d_model)) # 4 * 1 * 1280
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.position)

    def forward(self, x):
        length = x.size(0)
        out = x + self.position[:length, :]
        return out


class FeatureTransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, nout, dropout=0.1):
        super(FeatureTransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, nout)

        self.init_weights()

    # No mask in fact
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


# This class has the same function as 'FeatureTransformerModel' without position embedding process
# class AllFeatureTransformerModel(nn.Module):
#     def __init__(self, ninp, nhead, nhid, nlayers, nout, dropout=0.5):
#         super(AllFeatureTransformerModel, self).__init__()
#
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         # self.pos_encoder = PositionalEncoding(ninp)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.decoder = nn.Linear(ninp, nout)
#
#         self.init_weights()
#
#     # No mask in fact
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src):
#         # src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return output


import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dim_key, dim_query):
        super(CrossAttention, self).__init__()
        self.key = nn.Linear(feature_dim, dim_key, bias=False)
        self.query = nn.Linear(feature_dim, dim_query, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, A, B, C):
        QA = self.query(A.transpose(0, 1))
        QB = self.query(B.transpose(0, 1))
        QC = self.query(C.transpose(0, 1))

        KA = self.key(A.transpose(0, 1))
        KB = self.key(B.transpose(0, 1))
        KC = self.key(C.transpose(0, 1))

        VA = self.value(A.transpose(0, 1))
        VB = self.value(B.transpose(0, 1))
        VC = self.value(C.transpose(0, 1))

        attention_weights_AB = F.softmax(QA.bmm(KB.transpose(-2, -1)), dim=-1)
        attention_weights_BC = F.softmax(QB.bmm(KC.transpose(-2, -1)), dim=-1)
        attention_weights_CA = F.softmax(QC.bmm(KA.transpose(-2, -1)), dim=-1)

        output_A = attention_weights_AB.bmm(VB).transpose(0, 1)
        output_B = attention_weights_BC.bmm(VC).transpose(0, 1)
        output_C = attention_weights_CA.bmm(VA).transpose(0, 1)

        return output_A, output_B, output_C


class FusionCrossAttention(nn.Module):
    def __init__(self, feature_dim, dim_key, dim_query, fusion_dim):
        super(FusionCrossAttention, self).__init__()
        self.cross_attention = CrossAttention(feature_dim, dim_key, dim_query)

        self.fusion = nn.Linear(feature_dim * 3, fusion_dim)

    def forward(self, A, B, C):
        output_A, output_B, output_C = self.cross_attention(A, B, C)

        # 将三者在第一维度进行拼接
        concat_output = torch.cat([output_A, output_B, output_C], 0)

        # 通过全连接层进行特征整合
        fusion_output = self.fusion(concat_output)

        return fusion_output


if __name__ == '__main__':
    model = FeatureTransformerModel(ninp=512, nhead=8, nhid=2048, nlayers=6, nout=10)

    # (Batch_size, Sequence_length, Latent_dimension)
    input = torch.rand(32, 50, 512)

    input = input.permute(1, 0, -1)
    print(input.shape)

    output = model(input)
    output = output.permute(1, 0, -1)

    assert output.shape == (32, 50, 10), "Output shape doesn't match expected shape!"
