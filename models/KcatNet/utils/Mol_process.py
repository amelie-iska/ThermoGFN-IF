from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from transformers import AutoTokenizer, AutoModelForMaskedLM

from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import numpy as np
import json
import torch
import tqdm
import pickle


def inchi_to_smiles(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return "Invalid InChI"

def getMeanRepr(smiles_data, tokenizer, model):
    # mean_repr = np.zeros((smiles_data.shape[0], 767))
    mean_repr = {}
    for sequence in smiles_data:
        if len(sequence)>512:
            seq = sequence[:512]
        else:
            seq = sequence
        inputs = tokenizer.encode(seq, return_tensors="pt")

        try:
            output_repr = model(inputs)
            # mean_repr[i] = output_repr.logits[0].mean(dim=0).detach().numpy()
            mean_repr[sequence] = output_repr.logits[0].mean(dim=0).detach().numpy()
        except:
            print(sequence)
    return mean_repr

###pan1210

if __name__ == "__main__":

    with open("/home/tongpan/Kcat_predictor/data/kcat_data/splits/train_df_kcat.pkl", 'rb') as f:
        datasets = pickle.load(f)
    Inchi = [list(datasets["substrates"][ind])[0] for ind in datasets.index]
    Smiles0 = [inchi_to_smiles(Inchi_data) for Inchi_data in Inchi]

    with open("/home/tongpan/Kcat_predictor/data/kcat_data/splits/test_df_kcat.pkl", 'rb') as f:
        datasets = pickle.load(f)
    Inchi = [list(datasets["substrates"][ind])[0] for ind in datasets.index]
    Smiles1 = [inchi_to_smiles(Inchi_data) for Inchi_data in Inchi]

    Smiles = Smiles0 + Smiles1

    with open("/home/tongpan/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json") as infile:
        Kcat_data = json.load(infile)
    for data in Kcat_data:
        if data['Type'] == 'mutant' and float(data['Value']) > 0:
            smile = data['Smiles']
            Smiles.append(smile)

    Smiles.append("C1=CC(=CC=C1[N+](=O)[O-])OC2C(C(C(C(O2)CO)O)O)O",'CC(C)C[C@H](N=C(O)C[C@H](N)C(=O)O)C(=O)O')

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    all_smiles = np.array(Smiles)
    mean_repr = getMeanRepr(all_smiles, tokenizer, model)
    with open("/Workspace10/tongpan/Turnover/Kcat_mol.pkl", "wb") as file:
        pickle.dump(mean_repr, file)
