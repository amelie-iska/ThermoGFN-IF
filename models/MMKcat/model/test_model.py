#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-01-26 10:57
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : train.py
# @Software: PyCharm
import os
import time

import pandas as pd
import psutil
from scipy.stats import spearmanr

import pickle

from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset

import json

from basic_model_mm import mmKcatPrediction
from pretrain_trfm import *
from early_stop import *

mask_array = np.array([[True, True, True, True],
                       [True, True, True, False],
                       [True, True, False, True],
                       [True, True, False, False],  # Normal
                       [False, True, True, True],
                       [False, True, True, False],
                       [False, True, False, True],
                       [False, True, False, False],
                       [True, False, True, True],
                       [True, False, True, False],
                       [True, False, False, True],
                       [True, False, False, False],
                       [False, False, True, True],
                       [False, False, True, False],
                       [False, False, False, True],
                       [False, False, False, False]])  # Abnormal


class CustomizedDataset(Dataset):
    def __init__(self, seq_path, graph_x_path, graph_edge_index_path):
        with open(seq_path, 'r') as file:
            data_dict = json.load(file)
        file.close()

        # Prepare sequence-relevant data
        print('Preparing sequence data...')
        chosen_seq_data = {k: v for k, v in data_dict.items() if
                           len(v['Substrate_Smiles']) != 0 and v['Sequence_Rep'] is not None}
        self.seq_data = list(chosen_seq_data.values())
        assert len(data_dict) == len(self.seq_data)

        # Prepare graph-relevant data
        print('Preparing graph data...')
        chosen_graph_data = []
        with open(graph_x_path, 'rb') as file:
            graph_x = pickle.load(file)
        file.close()
        with open(graph_edge_index_path, 'rb') as file:
            graph_edge_index = pickle.load(file)
        file.close()
        for index, cur_graph in enumerate(zip(graph_x, graph_edge_index)):
            chosen_graph_data.append(cur_graph)
        self.graph_data = chosen_graph_data

        assert len(self.seq_data) == len(self.graph_data)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq_item, graph_item = self.seq_data[idx], self.graph_data[idx]
        return seq_item, graph_item


def costomized_collate_fn(batch):
    substrates_list = []
    enzymes_rep_list = []
    enzymes_graph_list = []
    products_list = []
    kcats_list = []
    seq_data = []
    graph_data = []
    for data_item in batch:
        seq_data.append(data_item[0])
        graph_data.append(data_item[1])
    for _data in seq_data:
        substrates_list.append(_data['Substrate_Smiles'])
        enzymes_rep_list.append(torch.tensor(_data['Sequence_Rep']).unsqueeze(dim=0))
        products_list.append(_data['Product_Smiles'])
        kcats_list.append(torch.tensor(float(_data['Value'])).unsqueeze(-1))
    for _data in graph_data:
        enzymes_graph_list.append(_data)
    return [substrates_list, enzymes_rep_list, enzymes_graph_list, products_list, kcats_list]


# Set random seed
torch.manual_seed(131)
np.random.seed(131)

dataset = CustomizedDataset('../data/concat_test_dataset_final_latest.json',
                            '../data/concat_test_graph_x_latest.pkl',
                            '../data/concat_test_graph_edge_index_latest.pkl')

device = torch.device('cuda:1')
batch_size = 32
save_path = f'../test_records/'
os.makedirs(save_path, exist_ok=True)

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                         collate_fn=costomized_collate_fn)

print(f'{len(dataset)} test data in total')

# lr=3e-5
model = mmKcatPrediction(device=device, batch_size=batch_size, nhead=4, nhid=1024, nlayers=4, gcn_hidden=512,
                         dropout=0.2, lambda_1=0.8, lambda_2=0.2, mode='test', ).to(device)
model.load_state_dict(torch.load(f'../ckpt/concat_best_checkpoint.pth'))
model.eval()  # Eval!!!!

df = pd.DataFrame(columns=['Mask', 'RMSE', 'R2', 'SRCC'])

with torch.no_grad():
    for mask in mask_array:
        print(f'Testing {mask}....')
        ground_truths = []
        predicted_values = []

        # Set mask
        model.test_mask = mask

        for batch_idx, data in enumerate(test_loader):
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss
            (loss, loss_x1, loss_x2, loss_x3, loss_x4, loss_x5,
             predicted_x1, predicted_x2, predicted_x3, predicted_x4, predicted_x5) = model(data)
            predicted_list = [predicted_kcat[0] for predicted_kcat in predicted_x5.tolist()]
            kcat_list = [kcat.item() for kcat in data[4]]
            ground_truths += (kcat_list)
            predicted_values += (predicted_list)

        RMSE = np.sqrt(mean_squared_error(ground_truths, predicted_values))
        R2 = r2_score(ground_truths, predicted_values)
        correlation, _ = spearmanr(ground_truths, predicted_values)
        print(f'Mask: {mask}\n'
              f'RMSE: {RMSE}\n'
              f'R2: {R2}\n'
              f'SRCC: {correlation}\n')

        new_data = {'Mask': mask, 'RMSE': RMSE, 'R2': R2, 'SRCC': correlation}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(f'{save_path}/testing_results.csv')
