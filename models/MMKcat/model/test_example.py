#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-01-26 10:57
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : test_example.py
# @Software: PyCharm
import math
import os
import sys

import torch

os.environ['TORCH_HOME'] = '/remote-home/sunxin/'
# Solve the issue of dssp
sys.path.append('../util')
current_path = os.environ.get('PATH', '')
new_path = '/root/anaconda3/envs/Torch/bin'
os.environ['PATH'] = f"{new_path}:{current_path}"
import esm
from colorama import Fore, Style, init

init(autoreset=True)

from basic_model_mm import mmKcatPrediction
from pretrain_trfm import *
from early_stop import *
from util.generate_graph import *

mask_array = np.array([[True, True, True, True],
                       [True, True, True, False],
                       [True, True, False, True],
                       [True, True, False, False]])

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print(Fore.RED + 'Using GPU for testing!')
else:
    device = torch.device('cpu')
    print(Fore.RED + 'Using CPU for testing!')

# Load MMKcat
model = mmKcatPrediction(device=device, batch_size=1, nhead=4, nhid=1024, nlayers=4, gcn_hidden=512, dropout=0.2,
                         lambda_1=0.8, lambda_2=0.2, mode='test', ).to(device)
model.load_state_dict(torch.load(f'../ckpt/concat_best_checkpoint.pth'))
model = model.eval()  # disables dropout for deterministic results
print(Fore.BLUE + 'Loading MMKcat finished.')

# Load ESM2_t33_650M_UR50D
ESM2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
ESM2 = ESM2.eval()  # disables dropout for deterministic results
print(Fore.BLUE + 'Loading ESM2_t33_650M_UR50D finished.')

# Load ESMFold
ESMFold = esm.pretrained.esmfold_v1()
ESMFold = ESMFold.eval().cuda()  # disables dropout for deterministic results
print(Fore.BLUE + 'Loading ESMFold finished.')


def get_protein_sequence_rep(sequence):
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = ESM2(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    # print(sequence_representations[0])
    return sequence_representations[0]


def get_protein_graph(sequence):
    output = ESMFold.infer_pdb(sequence)
    with open("./protein.pdb", "w") as file:
        file.write(output)
    file.close()
    graph = pdb2graph('./protein.pdb', '../util/mean_attr.pt')
    return graph


def predict_kcat(substrate_smiles, protein_sequence, product_smiles):
    if len(substrate_smiles) == 0:
        raise ValueError('Substrate SMILES is necessary for kcat prediction, please check and try again.')
    if len(protein_sequence) == 0:
        raise ValueError('Protein sequence is necessary for kcat prediction, please check and try again.')

    # Prepare representations for protein sequence and protein graph
    protein_sequence_rep = get_protein_sequence_rep(protein_sequence).reshape(1, -1)
    protein_graph = get_protein_graph(protein_sequence)
    if protein_graph is None:
        raise Exception('Failed in generating the protein graph, please try again.')

    # Prepare input data for MMKcat
    # The last parameter '0.0' is just for taking this position
    data = [[substrate_smiles], [protein_sequence_rep], [(protein_graph.x, protein_graph.edge_index)], [product_smiles], [torch.tensor(0.0)]]
    

    with torch.no_grad():
        for mask in mask_array:
            print(f'Testing {mask}....')
            # Set mask
            model.test_mask = mask

            (loss, loss_x1, loss_x2, loss_x3, loss_x4, loss_x5,
             predicted_x1, predicted_x2, predicted_x3, predicted_x4, predicted_x5) = model(data)
            predicted_list = [predicted_kcat[0] for predicted_kcat in predicted_x5.tolist()]



if __name__ == '__main__':
    # Replace the values of these variables to your chemical reactions
    substrate_smiles = ["CSCC[C@H](N)C(O)=NCC(=O)O"]
    protein_sequence = "MFLLPLPAAARVAVRHLSVKRLWAPGPAAADMTKGLVLGIYSKEKEEDEPQFTSAGENFNKLVSGKLREILNISGPPLKAGKTRTFYGLHEDFPSVVVVGLGKKTAGIDEQENWHEGKENIRAAVAAGCRQIQDLEIPSVEVDPCGDAQAAAEGAVLGLYEYDDLKQKRKVVVSAKLHGSEDQEAWQRGVLFASGQNLARRLMETPANEMTPTKFAEIVEENLKSASIKTDVFIRPKSWIEEQEMGSFLSVAKGSEEPPVFLEIHYKGSPNASEPPLVFVGKGITFDSGGISIKAAANMDLMRADMGGAATICSAIVSAAKLDLPINIVGLAPLCENMPSGKANKPGDVVRARNGKTIQVDNTDAEGRLILADALCYAHTFNPKVIINAATLTGAMDIALGSGATGVFTNSSWLWNKLFEASIETGDRVWRMPLFEHYTRQVIDCQLADVNNIGKYRSAGACTAAAFLKEFVTHPKWAHLDIAGVMTNKDEVPYLRKGMAGRPTRTLIEFLFRFSQDSA"
    # If products are unknown, please set this variable as: product_smiles = [[None]]
    product_smiles = ["NCC(=O)O", "CSCC[C@H](N)C(=O)O"]

    kcat = predict_kcat(substrate_smiles=substrate_smiles, protein_sequence=protein_sequence,
                        product_smiles=product_smiles)

    # Report the results in both log10 and transformed
    print(kcat, math.pow(10, kcat))
