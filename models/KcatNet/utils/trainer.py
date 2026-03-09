import torch
from utils.metrics import evaluate_cls, evaluate_mcls, evaluate_reg
import json
from utils import unbatch
import math
import os
import sys
import pandas as pd
import numpy as np
import csv
from utils.utils import  virtual_screening

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, lrate, min_lrate, clip, steps_per_epoch, num_epochs,
                 total_iters,warmup_iters=0, lr_decay_iters=None, schedule_lr=True,  evaluate_metric='rmse',
                 result_path='', runid=0, device='cuda', skip_test_during_train=False):

        self.model = model
        self.model.to(device)
        # #
        self.optimizer = self.model.optimizer
        self.clip = clip
        self.regression_loss = torch.nn.MSELoss(reduction='mean')
        self.num_epochs = num_epochs

        self.result_path = result_path
        self.runid = runid
        self.device = device
        self.evaluate_metric = evaluate_metric
        self.skip_test_during_train = skip_test_during_train
        self.schedule_lr = schedule_lr
        if total_iters:
            self.total_iters = total_iters
        else:
            self.total_iters = num_epochs * steps_per_epoch

        self.lrate = lrate
        self.min_lrate = min_lrate
        self.warmup_iters = warmup_iters
        if lr_decay_iters is None:
            self.lr_decay_iters = self.total_iters
        else:
            self.lr_decay_iters = lr_decay_iters

    def train_epoch(self, train_loader, val_loader):#
        train_epoch_losses = []
        val_epoch_losses = []
        best_result = float('inf')
        iter_num = 0

        for epoch in range(1, self.num_epochs + 1):
            print("epoch:", epoch)
            running_reg_loss = 0
            running_cluster_loss = 0
            self.model.train()

            for data in train_loader:
                if self.schedule_lr:
                    curr_lr_rate = self.get_lr(iter_num)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = curr_lr_rate

                self.optimizer.zero_grad()

                data = data.to(self.device)
                reg_pred,cl_loss = self.model(
                    # Molecule
                    mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, mol_total_fea=data.mol_total_fea,
                    # Enzyme
                    residue_esm=data.prot_node_esm, residue_prot5=data.prot_node_prot5,
                    residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                    # Mol-Enzyme Interaction batch
                    mol_batch=data.mol_x_batch, prot_batch=data.prot_node_esm_batch)
                ## Loss compute
                loss_val = torch.tensor(0.).to(self.device)
                loss_val += cl_loss * 0.1
                cl_loss = cl_loss.item()


                reg_pred = reg_pred.squeeze()
                reg_y = data.reg_y.squeeze()
                reg_loss = self.regression_loss(reg_pred, reg_y)
                loss_val += reg_loss* 0.9
                reg_loss = reg_loss.item()
                loss_val.backward() #retain_graph=True

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                running_reg_loss += reg_loss
                running_cluster_loss += cl_loss
                # pbar.update(1)
                iter_num += 1

            train_reg_loss = running_reg_loss / len(train_loader)
            print("train rmse",np.sqrt(train_reg_loss))
            train_epoch_losses.append(np.sqrt(train_reg_loss))

            print("starting to evaluate -------------------------------------------")
            val_result = self.eval(val_loader)
            print("validation rmse",val_result["rmse"])
            val_epoch_losses.append(val_result["rmse"])

            if val_result[self.evaluate_metric] < best_result:
                print(f'Validation rmse decreased ({best_result:.4f} --> {val_result["rmse"]:.4f})', 'best_pearson:', val_result["pearson"], 'R2:', val_result["r2_score"], 'Saving model ...')
                best_result = val_result[self.evaluate_metric]
                torch.save(self.model.state_dict(), os.path.join(self.result_path,'model.pt'))
            else:
                print('current mse: ', val_result["rmse"], ' No improvement since best_mse', best_result)



    def eval(self, data_loader):
        reg_preds = []
        reg_truths = []

        running_reg_loss = 0
        running_cluster_loss = 0

        self.model.eval()
        eval_result = {}
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                reg_pred,  cl_loss = self.model(
                    # Molecule
                    mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, mol_total_fea=data.mol_total_fea,
                    # Enzyme
                    residue_esm=data.prot_node_esm, residue_prot5=data.prot_node_prot5,
                    residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                    # Mol-Enzyme Interaction batch
                    mol_batch=data.mol_x_batch, prot_batch=data.prot_node_esm_batch)
                ## Loss compute
                loss_val = 0
                loss_val += cl_loss
                cl_loss = cl_loss.item()

                reg_pred = reg_pred.squeeze().reshape(-1)
                reg_y = data.reg_y.squeeze().reshape(-1)
                reg_loss = self.regression_loss(reg_pred, reg_y) #* self.regression_weight
                loss_val += reg_loss
                reg_loss = reg_loss.item()
                reg_preds.append(reg_pred)
                reg_truths.append(reg_y)
                running_reg_loss += reg_loss
                running_cluster_loss += cl_loss

            eval_reg_loss = running_reg_loss / len(data_loader)
            eval_cluster_loss = running_cluster_loss / len(data_loader)
            eval_result['regression_loss'] = eval_reg_loss
            eval_result['cluster_loss'] = eval_cluster_loss

        if len(reg_truths) > 0:
            reg_preds = torch.cat(reg_preds).detach().cpu().numpy()
            reg_truths = torch.cat(reg_truths).detach().cpu().numpy()
            eval_reg_result = evaluate_reg(reg_truths, reg_preds)
            eval_result.update(eval_reg_result)
        return eval_result



    def get_lr(self, iter):
        # 1) linear warmup for warmup_iters steps
        if iter < self.warmup_iters:
            return self.lrate * iter / self.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > self.lr_decay_iters:
            return self.min_lrate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return self.min_lrate + coeff * (self.lrate - self.min_lrate)


def pred(model, data_loader,device):
    reg_preds = []
    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            reg_pred,  _ = model(
                # Molecule
                mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, mol_total_fea=data.mol_total_fea,
                # Enzyme
                residue_esm=data.prot_node_esm, residue_prot5=data.prot_node_prot5,
                residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                # Mol-Enzyme Interaction batch
                mol_batch=data.mol_x_batch, prot_batch=data.prot_node_esm_batch)

            reg_pred = reg_pred.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_preds += reg_pred

    return reg_preds

