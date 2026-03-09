#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-02-04 16:36
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : train_model.py
# @Software: PyCharm
import os
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_squared_error, r2_score
import pickle
import random
import numpy as np
import json
from colorama import Fore, Style, init

init(autoreset=True)

from basic_model_mm import mmKcatPrediction
from pretrain_trfm import *
from early_stop import *


def set_seed(seed):
    # seeding
    print(Fore.RED + f'Current seed is set as {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("CUDA Available: ", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())


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


def train(args, seed):
    # Set global seed
    set_seed(seed)

    train_dataset = CustomizedDataset('../data/concat_train_dataset_final_latest.json',
                                      '../data/concat_train_graph_x_latest.pkl',
                                      '../data/concat_train_graph_edge_index_latest.pkl')
    test_dataset = CustomizedDataset('../data/concat_test_dataset_final_latest.json',
                                     '../data/concat_test_graph_x_latest.pkl',
                                     '../data/concat_test_graph_edge_index_latest.pkl')
    train_ratio = 0.9
    train_size = int(train_ratio * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                 collate_fn=costomized_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              collate_fn=costomized_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             collate_fn=costomized_collate_fn)

    device = torch.device('cuda:0')
    print(Fore.RED + 'basic_model_mm is training!!!!')
    save_path = f'../ckpt/'
    os.makedirs(save_path, exist_ok=True)

    model = mmKcatPrediction(device=device, batch_size=args.batch_size, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers,
                             gcn_hidden=args.gcn_hidden, dropout=args.dropout, lambda_1=args.lambda_1,
                             lambda_2=args.lambda_1, lambda_3=args.lambda_1, lambda_4=args.lambda_1,
                             lambda_5=args.lambda_2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=300, eta_min=0)
    early_stop = EarlyStopping(patience=20, verbose=True, dataset_name='concat')
    writer = SummaryWriter('/root/tf-logs')
    epoch = 1000

    for e in range(1, epoch + 1):
        # Train model
        model.mode = 'train'
        model.train()
        model.sub_seq_channel.eval()
        model.prod_seq_channel.eval()
        train_loss_total = 0.0
        for batch_idx, data in tqdm(enumerate(training_loader), total=len(training_loader), desc="Training"):
            loss, _, _, _, _, _, _, _, _, _, _ = model(data)
            train_loss_total += loss.item() * args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('train_loss', train_loss_total / len(train_dataset), e)
        print(f'epoch: {e}, train_loss: {train_loss_total / len(train_dataset):.4f}')

        # Valid model
        model.mode = 'validation'
        model.test_mask = np.asarray([True, True, True, True])
        model.eval()
        valid_loss_total = 0.0
        with (torch.no_grad()):
            for batch_idx, data in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation"):
                loss, _, _, _, _, _, _, _, _, _, _ = model(data)
                valid_loss_total += loss.item() * args.batch_size
            writer.add_scalar('validation_loss', valid_loss_total / len(valid_dataset), e)
            print(f'epoch: {e}, validation_loss: {valid_loss_total / len(valid_dataset):.4f}')

            # Decide whether to early stop
            early_stop(val_loss=valid_loss_total / len(valid_dataset), model=model, path=save_path)
            if early_stop.early_stop:
                print(Fore.RED + 'Early Stop!!!!')
                break

        scheduler.step()

    # Test model
    model.mode = 'test'
    print(Fore.RED + 'Load Best Model!!!!')
    model.load_state_dict(torch.load(f'{save_path}/concat_best_checkpoint.pth'))
    model.test_mask = np.asarray([True, True, True, True])
    model.eval()
    ground_truths = []
    predicted_values = []
    with (torch.no_grad()):
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            loss, _, _, _, _, _, _, _, _, _, predicted_kcats = model(data)
            predicted_list = [predicted_kcat[0] for predicted_kcat in predicted_kcats.tolist()]
            kcat_list = [kcat.item() for kcat in data[4]]
            ground_truths += (kcat_list)
            predicted_values += (predicted_list)
    RMSE = np.sqrt(mean_squared_error(ground_truths, predicted_values))

    # Save final model
    print(f'Training and Testing finished. RMSE: {RMSE:.4f}')
    return RMSE


if __name__ == '__main__':

    # Define hyper-parameters for training mmKcat
    parser = argparse.ArgumentParser(description='Run mmKcat for Prediction.')
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nhid', type=int, default=1024)
    parser.add_argument('--nlayers', type=int, default=4)
    parser.add_argument('--gcn_hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lambda_1', type=float, default=0.8)
    parser.add_argument('--lambda_2', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    args = parser.parse_args()

    seeds = [131]
    for seed in seeds:
        train(args, seed)
