#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-01-19 17:04
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : basic_model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from feature_transformer import *
# from smiles_transformer.smiles_transformer.build_vocab import *
from pretrain_trfm import *
from gcn import *

import pickle

mask_array = np.array([[True, True, True, True],
                       [True, True, True, False],
                       [True, True, False, True],
                       [True, True, False, False]])


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        batch, channel, dimension = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(batch, channel, dimension)
        return x


class mmKcatPrediction(nn.Module):
    def __init__(self, device, batch_size, nhead=8, nhid=2048, nlayers=6, gcn_hidden=128, dropout=0.1, test_mask=None,
                 mode='train', lambda_1=0.1, lambda_2=0.1, lambda_3=0.1, lambda_4=0.1, lambda_5=0.1):
        super(mmKcatPrediction, self).__init__()

        # Model parameters
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.gcn_hidden = gcn_hidden
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.mode = mode
        self.test_mask = test_mask
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.lambda_5 = lambda_5

        # Mask mechanism
        self.masker = MaskModal()

        # SMILES transformer parameters
        self.vocab = WordVocab.load_vocab('vocab.pkl')
        assert len(self.vocab) == 45
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

        # Substrates channel
        # Substrates sequence sub-channel (Must available)
        self.sub_seq_channel = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        self.sub_seq_channel.load_state_dict(torch.load('trfm_12_23000.pkl'))

        # Enzymes / Protein channel
        # Protein sequence sub-channel (Must available)
        # self.pro_seq_channel = FeatureTransformerModel(ninp=1280, nhead=self.nhead, nhid=self.nhid,
        #                                                nlayers=self.nlayers, nout=1280)
        # Protein graph sub-channel (Maybe absent)
        self.enzyme_gcn = GCN(in_channels=26, hidden_channels=self.gcn_hidden, out_channels=1024, device=self.device)

        # Products channel
        # Products sequence channel (Maybe absent)
        self.prod_seq_channel = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        self.prod_seq_channel.load_state_dict(torch.load('trfm_12_23000.pkl'))

        # Freeze SMILES transformers
        for param in self.sub_seq_channel.parameters():
            param.requires_grad = False
        for param in self.prod_seq_channel.parameters():
            param.requires_grad = False

        # Multimodel representation adaptor
        self.substrates_adaptor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=1280)
        )

        self.enzymes_seq_adaptor = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=1280)
        )

        self.enzymes_graph_adaptor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=1280)
        )

        self.products_adaptor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=1280)
        )

        # Multi-Modal Fusion Transformer
        self.all_rep_channel = FeatureTransformerModel(ninp=1280, nhead=self.nhead, nhid=self.nhid,
                                                       nlayers=self.nlayers, nout=1280)

        # self.prediction_head = nn.Linear(in_features=3840, out_features=1)
        self.prediction_head = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512),
            # nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        # Auxiliary Regularizer
        self.aux_prediction_head = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512),
            # nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.loss = nn.MSELoss(reduction='mean')

        self.init_weight()

    def init_weight(self):
        for m in self.substrates_adaptor.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
        for m in self.enzymes_seq_adaptor.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
        for m in self.enzymes_graph_adaptor.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
        for m in self.products_adaptor.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
        for m in self.prediction_head.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
        for m in self.aux_prediction_head.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)

    def forward(self, data):

        # Keep SMILES transformers in 'eval' mode to avoid dropout and normalization
        self.sub_seq_channel.eval()
        self.prod_seq_channel.eval()

        # Simulate the interaction among the substrates, the enzymes and the products
        # Divide data into corresponding channel
        substrates_seq = data[0]
        enzymes_seq_rep = data[1]
        enzymes_graph = data[2]
        products_seq = data[3]
        kcats_list = data[4]

        if self.mode == 'train' or self.mode == 'random_mask':
            cur_batch_size = len(substrates_seq)
            cur_mask_index = np.random.choice(len(mask_array), cur_batch_size)
            cur_mask = torch.from_numpy(mask_array[cur_mask_index])
        elif self.mode == 'validation' or self.mode == 'test':
            cur_batch_size = len(substrates_seq)
            cur_mask = torch.from_numpy(self.test_mask).repeat((cur_batch_size, 1))

        # Get substrates modality-specific representation ('Must' available)
        substrates_rep_list = []
        for substrates_sub_set in substrates_seq:
            sub_split = [self.split(sm) for sm in substrates_sub_set]
            xid, xseg = self.get_array(sub_split)
            cur_substrate_rep = self.sub_seq_channel.encode(torch.t(xid).to(self.device))
            cur_substrate_rep = torch.from_numpy(cur_substrate_rep).mean(dim=0).unsqueeze(dim=0)
            substrates_rep_list.append(cur_substrate_rep)
        substrates_seq_rep = torch.stack(substrates_rep_list, dim=0).to(self.device)
        substrates_seq_rep = self.substrates_adaptor(substrates_seq_rep)  # batch_size * 1 * dimension

        # Get enzymes modality-specific representation
        # Enzyme sequence representation ('Must' available)
        enzymes_seq_rep = torch.stack(enzymes_seq_rep, dim=0).to(self.device)
        enzymes_seq_rep = self.enzymes_seq_adaptor(enzymes_seq_rep)  # batch_size * 1 * dimension
        # Enzyme graph representation ('Maybe' missing)
        enzymes_graph_rep_list = []
        for graph_item in enzymes_graph:
            x = graph_item[0].to(self.device)
            edge_index = graph_item[1].to(self.device)
            cur_graph_rep = self.enzyme_gcn(x, edge_index)
            enzymes_graph_rep_list.append(cur_graph_rep)
        enzymes_graph_rep = torch.stack(enzymes_graph_rep_list, dim=0)
        enzymes_graph_rep = self.enzymes_graph_adaptor(enzymes_graph_rep)  # batch * 1 * dimension

        # Get products modality-specific representation (Missing in the most cases)
        products_rep_list = []
        for products_sub_set in products_seq:
            if None not in products_sub_set:
                prod_split = [self.split(sm) for sm in products_sub_set]
                xid, xseg = self.get_array(prod_split)
                cur_product_rep = self.prod_seq_channel.encode(torch.t(xid).to(self.device))
                cur_product_rep = torch.from_numpy(cur_product_rep).mean(dim=0).unsqueeze(dim=0)
                products_rep_list.append(cur_product_rep)
            else:
                cur_product_rep = torch.zeros(1, 1024)
                products_rep_list.append(cur_product_rep)
        products_seq_rep = torch.stack(products_rep_list, dim=0).to(self.device)
        products_seq_rep = self.products_adaptor(products_seq_rep)  # batch_size * 1 * dimension

        # Get kcat modality-specific representation
        kcats_list = torch.stack(kcats_list, dim=0).to(self.device)

        # Stage one prediction
        # Auxiliary regularization 1: all unmasked features before
        aux_mask_1 = torch.from_numpy(np.array([True, True, True, True])).repeat((cur_batch_size, 1))
        rep_for_reg_1 = self.masker(
            torch.cat([substrates_seq_rep, enzymes_seq_rep, enzymes_graph_rep, products_seq_rep], dim=1),
            aux_mask_1)
        # rep_for_reg_1 = rep_for_reg_1.permute(1, 0, -1)
        # rep_for_reg_1 = self.all_rep_channel(rep_for_reg_1).permute(1, 0, -1)
        rep_for_reg_1_for_pre = self.pool(rep_for_reg_1.permute(0, 2, 1)).permute(0, 2, 1).squeeze(dim=1)
        predicted_kcats_x1 = self.aux_prediction_head(rep_for_reg_1_for_pre)
        loss1 = self.loss(predicted_kcats_x1, kcats_list)

        # Auxiliary regularization 2: enzyme graph features are missing
        aux_mask_2 = torch.from_numpy(np.array([True, True, False, True])).repeat((cur_batch_size, 1))
        rep_for_reg_2 = self.masker(
            torch.cat([substrates_seq_rep, enzymes_seq_rep, enzymes_graph_rep, products_seq_rep], dim=1),
            aux_mask_2)
        # rep_for_reg_2 = rep_for_reg_2.permute(1, 0, -1)
        # rep_for_reg_2 = self.all_rep_channel(rep_for_reg_2).permute(1, 0, -1)
        rep_for_reg_2_for_pre = self.pool(rep_for_reg_2.permute(0, 2, 1)).permute(0, 2, 1).squeeze(dim=1)
        predicted_kcats_x2 = self.aux_prediction_head(rep_for_reg_2_for_pre)
        loss2 = self.loss(predicted_kcats_x2, kcats_list)

        # Auxiliary regularization 3: product features are missing
        aux_mask_3 = torch.from_numpy(np.array([True, True, True, False])).repeat((cur_batch_size, 1))
        rep_for_reg_3 = self.masker(
            torch.cat([substrates_seq_rep, enzymes_seq_rep, enzymes_graph_rep, products_seq_rep], dim=1),
            aux_mask_3)
        # rep_for_reg_3 = rep_for_reg_3.permute(1, 0, -1)
        # rep_for_reg_3 = self.all_rep_channel(rep_for_reg_3).permute(1, 0, -1)
        rep_for_reg_3_for_pre = self.pool(rep_for_reg_3.permute(0, 2, 1)).permute(0, 2, 1).squeeze(dim=1)
        predicted_kcats_x3 = self.aux_prediction_head(rep_for_reg_3_for_pre)
        loss3 = self.loss(predicted_kcats_x3, kcats_list)

        # Auxiliary regularization 3: both graph and product features are missing
        aux_mask_4 = torch.from_numpy(np.array([True, True, False, False])).repeat((cur_batch_size, 1))
        rep_for_reg_4 = self.masker(
            torch.cat([substrates_seq_rep, enzymes_seq_rep, enzymes_graph_rep, products_seq_rep], dim=1),
            aux_mask_4)
        # rep_for_reg_4 = rep_for_reg_4.permute(1, 0, -1)
        # rep_for_reg_4 = self.all_rep_channel(rep_for_reg_4).permute(1, 0, -1)
        rep_for_reg_4_for_pre = self.pool(rep_for_reg_4.permute(0, 2, 1)).permute(0, 2, 1).squeeze(dim=1)
        predicted_kcats_x4 = self.aux_prediction_head(rep_for_reg_4_for_pre)
        loss4 = self.loss(predicted_kcats_x4, kcats_list)

        # Stage two prediction
        rep_in_all = torch.cat([substrates_seq_rep, enzymes_seq_rep, enzymes_graph_rep, products_seq_rep], dim=1)
        rep_in_all = self.masker(rep_in_all, cur_mask)
        rep_in_all = rep_in_all.permute(1, 0, -1)
        rep_in_all = self.all_rep_channel(rep_in_all).permute(1, 0, -1)
        rep_in_all_for_pre = self.pool(rep_in_all.permute(0, 2, 1)).permute(0, 2, 1).squeeze(dim=1)
        predicted_kcats_x5 = self.prediction_head(rep_in_all_for_pre)
        loss5 = self.loss(predicted_kcats_x5, kcats_list)

        loss1 = loss1 * self.lambda_1
        loss2 = loss2 * self.lambda_2
        loss3 = loss3 * self.lambda_3
        loss4 = loss4 * self.lambda_4
        loss5 = loss5 * self.lambda_5

        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return (loss, loss1, loss2, loss3, loss4, loss5,
                predicted_kcats_x1, predicted_kcats_x2, predicted_kcats_x3, predicted_kcats_x4, predicted_kcats_x5)

    def split(self, sm):
        '''
        function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
        input: A SMILES
        output: A string with space between words
        '''
        arr = []
        i = 0
        while i < len(sm) - 1:
            if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                             'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
                arr.append(sm[i])
                i += 1
            elif sm[i] == '%':
                arr.append(sm[i:i + 3])
                i += 3
            elif sm[i] == 'C' and sm[i + 1] == 'l':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'C' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'C' and sm[i + 1] == 'u':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'B' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'S' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'N' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'N' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'R' and sm[i + 1] == 'b':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'R' and sm[i + 1] == 'a':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'X' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'L' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'l':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 's':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'g':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'A' and sm[i + 1] == 'u':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'M' and sm[i + 1] == 'g':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'M' and sm[i + 1] == 'n':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'T' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'Z' and sm[i + 1] == 'n':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 's' and sm[i + 1] == 'i':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 's' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 't' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'H' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '2':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '3':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '+' and sm[i + 1] == '4':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '2':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '3':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == '-' and sm[i + 1] == '4':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'K' and sm[i + 1] == 'r':
                arr.append(sm[i:i + 2])
                i += 2
            elif sm[i] == 'F' and sm[i + 1] == 'e':
                arr.append(sm[i:i + 2])
                i += 2
            else:
                arr.append(sm[i])
                i += 1
        if i == len(sm) - 1:
            arr.append(sm[i])
        return ' '.join(arr)

    def get_array(self, smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = self.get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    def get_inputs(self, sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109] + sm[-109:]
        ids = [self.vocab.stoi.get(token, self.unk_index) for token in sm]
        ids = [self.sos_index] + ids + [self.eos_index]
        seg = [1] * len(ids)
        padding = [self.pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
