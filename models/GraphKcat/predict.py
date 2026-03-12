
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from model_enz import bottle_view_graph
from dataset_graphkcat_chai1 import GraphDataset, PLIDataLoader
import numpy as np
from model_utils import *
from config.config_dict import Config
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from pathlib import Path
# from tmtools import tm_align
import esm
from unimol_tools import UniMolRepr
from preprocessing_inference import extract_pocket_and_ligand, get_pocket_by_sdf, \
      transfer_conformation, extract_sequence_from_pdb, get_unimol2_embedding, get_esm2_embeddings, three_to_one

SUPPORTED_LIGAND_ATOMS = {"B", "Br", "C", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"}


def _load_pdb_mol_robust(pdb_path):
    attempts = [
        dict(sanitize=True, removeHs=True),
        dict(sanitize=False, removeHs=True, proximityBonding=False),
        dict(sanitize=False, removeHs=True, proximityBonding=True),
    ]
    for kwargs in attempts:
        mol = Chem.MolFromPDBFile(pdb_path, **kwargs)
        if mol is None:
            continue
        if mol.GetNumAtoms() == 0:
            continue
        if mol.GetNumConformers() == 0:
            continue
        return mol
    return None


def _load_ligand_sdf_robust(sdf_path):
    for sanitize in (True, False):
        mol = Chem.MolFromMolFile(sdf_path, sanitize=sanitize, removeHs=True)
        if mol is None:
            continue
        if mol.GetNumAtoms() == 0:
            continue
        if mol.GetNumConformers() == 0:
            continue
        return mol
    return None


def _validate_graphkcat_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid substrate smiles")
    unsupported = sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - SUPPORTED_LIGAND_ATOMS)
    if unsupported:
        raise ValueError(
            f"unsupported ligand atom types for GraphKcat: {','.join(unsupported)}"
        )
    if mol.GetNumBonds() == 0:
        raise ValueError("GraphKcat requires a bonded substrate molecule")
    return mol


def preprocessing(df, clf, esm_model,alphabet, batch_converter, cutoff=8):
    valid_rows = []
    for i, row in df.iterrows():
        cid = row["id"]
        try:
            has_complex = 'complex' in row.index and row['complex'] and not pd.isna(row['complex'])
            if has_complex:
                data_dir = os.path.dirname(row["complex"])
                complex = row["complex"]
                complex_path = complex
                if not os.path.exists(complex_path):
                    raise FileNotFoundError(f"complex does not exist: {complex_path}")
                extract_pocket_and_ligand(complex_path, cutoff=cutoff)
                protein_path = os.path.join(data_dir, f"{cid}_protein.pdb")
                pocket_path = os.path.join(data_dir, f"Pocket_{cutoff}A.pdb")
            else:
                data_dir = os.path.dirname(row["ligand"])
                ligand_path = row["ligand"]
                protein_path = row["protein"]
                if not os.path.exists(ligand_path):
                    raise FileNotFoundError(f"ligand does not exist: {ligand_path}")
                if not os.path.exists(protein_path):
                    raise FileNotFoundError(f"protein does not exist: {protein_path}")
                pocket_path = get_pocket_by_sdf(ligand_path, protein_path, distance=cutoff)

            smiles = row["Smiles"]
            _validate_graphkcat_smiles(smiles)

            ligand_sdf = os.path.join(data_dir, f"{cid}_ligand.sdf")
            if not os.path.exists(ligand_sdf):
                transfer_conformation(
                    os.path.join(data_dir, f"{cid}_ligand.pdb"),
                    smiles,
                    ligand_sdf,
                )

            if _load_ligand_sdf_robust(ligand_sdf) is None:
                raise ValueError(f"failed to parse generated ligand SDF: {ligand_sdf}")
            if _load_pdb_mol_robust(pocket_path) is None:
                raise ValueError(f"failed to parse pocket PDB with RDKit: {pocket_path}")

            seq = extract_sequence_from_pdb(protein_path)
            if not seq:
                raise ValueError(f"empty protein sequence extracted from {protein_path}")

            print(f"Processing {cid} with sequence length {len(seq)}")
            if os.path.exists(os.path.join(data_dir, f"{cid}_unimol_1b.pt")) and \
               os.path.exists(os.path.join(data_dir, f"{cid}_esm2_3b.pt")):
                print(f"Embeddings for {cid} already exist, skipping...")
                valid_rows.append(row)
                continue

            mol_embedding = get_unimol2_embedding(clf, smiles, embedding_type="atomic_reprs")
            esm_embedding = get_esm2_embeddings(esm_model, alphabet, batch_converter, seq, mean=False)

            torch.save(mol_embedding, os.path.join(data_dir, f"{cid}_unimol_1b.pt"))
            torch.save(esm_embedding, os.path.join(data_dir, f"{cid}_esm2_3b.pt"))
            valid_rows.append(row)
        except Exception as exc:
            print(f"Error: {cid} preprocessing failed, skipping... {exc}")
    return pd.DataFrame(valid_rows).reset_index(drop=True)


def compute_km_loss_and_pcc(pred_km, y_km):

    if isinstance(pred_km, np.ndarray):
        pred_km = torch.from_numpy(pred_km).float()
    if isinstance(y_km, list):
        y_km = torch.tensor([float('nan') if x is None else x for x in y_km], dtype=torch.float32)
    elif isinstance(y_km, np.ndarray):
        y_km = torch.from_numpy(y_km).float()
    
    # 生成有效样本的掩码
    mask_km = ~torch.isnan(y_km)
    
    loss_km, pcc_km = 0.0, None
    
    if mask_km.any():
        # 提取有效样本并确保一维形状
        valid_pred_km = pred_km[mask_km].flatten()
        valid_y_km = y_km[mask_km].flatten()
        
        # 计算 MSE 损失
        loss_km = F.mse_loss(valid_pred_km, valid_y_km)
        
        # 计算 Pearson 相关系数
        if valid_pred_km.size(0) > 1:
            mean_pred = valid_pred_km.mean()
            mean_y = valid_y_km.mean()
            diff_pred = valid_pred_km - mean_pred
            diff_y = valid_y_km - mean_y
            
            numerator = torch.sum(diff_pred * diff_y)
            denominator = torch.sqrt(torch.sum(diff_pred ** 2) * torch.sum(diff_y ** 2))
            
            if denominator == 0:
                # 处理分母为零的情况
                if torch.all(diff_pred == 0) and torch.all(diff_y == 0):
                    pcc_km = 1.0  # 两者均为常数，视为完全相关
                else:
                    pcc_km = None  # 无法计算 PCC
            else:
                pcc_km = (numerator / denominator).item()

    
            # np.save("valid_pred_kcat.npy", valid_pred_km.detach().cpu().numpy())
            # np.save("valid_y_kcat.npy", valid_y_km.detach().cpu().numpy())
    return loss_km, pcc_km, pred_km
def _predict_once(model, dataloader, device):
    pred_kcat_list = []
    pred_km_list = []
    for data in dataloader:
        with torch.no_grad():
            # GraphKcat mutates node features in-place during the forward pass.
            # Clone each graph so repeated inference passes (for MC-dropout or
            # multiple epochs over the same dataset) do not corrupt the cached
            # raw input features stored in the dataset.
            data = [data[i].clone().to(device) for i in range(len(data))]
            pred_kcat, pred_km,_,_,_,_,_,_,_,_,_,_ = model(data)
            pred_kcat_list.append(pred_kcat.detach().cpu().numpy())
            pred_km_list.append(pred_km.detach().cpu().numpy())

    pred_kcat = np.concatenate(pred_kcat_list, axis=0)
    pred_km = np.concatenate(pred_km_list, axis=0)
    log_pred_kcat = pred_kcat
    log_pred_km = pred_km
    log_pred_kcat_km = log_pred_kcat - log_pred_km
    return log_pred_kcat, log_pred_km, log_pred_kcat_km


def _enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def val(model, dataloader, device, mc_dropout_samples=1, mc_dropout_seed=13):
    mc_samples = max(1, int(mc_dropout_samples))
    if mc_samples == 1:
        model.eval()
        log_pred_kcat, log_pred_km, log_pred_kcat_km = _predict_once(model, dataloader, device)
        zeros = np.zeros_like(log_pred_kcat, dtype=np.float32)
        return log_pred_kcat, zeros, log_pred_km, zeros, log_pred_kcat_km, zeros

    pred_kcat_samples = []
    pred_km_samples = []
    pred_kcat_km_samples = []
    for sample_idx in range(mc_samples):
        sample_seed = int(mc_dropout_seed) + sample_idx
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        _enable_mc_dropout(model)
        log_pred_kcat, log_pred_km, log_pred_kcat_km = _predict_once(model, dataloader, device)
        pred_kcat_samples.append(log_pred_kcat)
        pred_km_samples.append(log_pred_km)
        pred_kcat_km_samples.append(log_pred_kcat_km)

    kcat_stack = np.stack(pred_kcat_samples, axis=0)
    km_stack = np.stack(pred_km_samples, axis=0)
    kcat_km_stack = np.stack(pred_kcat_km_samples, axis=0)
    return (
        np.mean(kcat_stack, axis=0),
        np.std(kcat_stack, axis=0, ddof=1 if mc_samples > 1 else 0),
        np.mean(km_stack, axis=0),
        np.std(km_stack, axis=0, ddof=1 if mc_samples > 1 else 0),
        np.mean(kcat_km_stack, axis=0),
        np.std(kcat_km_stack, axis=0, ddof=1 if mc_samples > 1 else 0),
    )


def get_ca_coords_and_sequence(pdb_file, chain_id):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # 只考虑第一个模型
    model = structure[0]
    
    # 尝试获取指定链
    try:
        chain = model[chain_id]
    except KeyError:
        available_chains = list(model.child_dict.keys())
        raise ValueError(f"链 {chain_id} 不存在。可用链: {available_chains}")
    

    coords = []
    seq = []
    for residue in chain:

        if residue.id[0].strip() != "":
            continue
            
        resname = residue.resname.strip()
        

        if is_aa(residue) and residue.has_id('CA'):
            ca = residue['CA']
            coords.append(ca.coord)
            

            try:
                aa_code = three_to_one(resname)
                seq.append(aa_code)
            except:

                seq.append('X')
    
    return np.array(coords, dtype=np.float32), ''.join(seq)

def parse_args():
    parser = argparse.ArgumentParser(description="GraphKcat Prediction")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--cpkt_path', type=str, default='./checkpoint/paper.pt', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (e.g., cuda:0, cpu)')
    parser.add_argument('--cfg', type=str, default='TrainConfig_kcat_enz', help='Configuration file name')
    parser.add_argument('--organism_set_path', type=str, default='./sub_utils/all_organism_set.npy', help='Path to the organism set file')
    parser.add_argument('--temp_set_path', type=str, default='./sub_utils/temp_set.npy', help='Path to the temporary set file')
    parser.add_argument('--mc_dropout_samples', type=int, default=8, help='Number of MC-dropout samples for predictive uncertainty')
    parser.add_argument('--mc_dropout_seed', type=int, default=13, help='Base seed for MC-dropout inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = args.cfg
    config = Config(cfg)
    config = config.get_config()   
    batch_size = config.get("batch_size")
    hidden_dim = config.get("hidden_dim")
    pooling = config.get("pooling")
    vocab_size = config.get("vocab_size")
    num_layers = config.get("num_layers")
    dropout = config.get("dropout")
    ligand_nn_embedding = config.get("ligand_nn_embedding")
    HeteroGNN_layers = config.get("HeteroGNN_layers")
    num_fc_layers = config.get("num_fc_layers")
    fc_hidden_dim = config.get("fc_hidden_dim")  # hidden_dim *  fc_hidden_dim
    share_fc = config.get("share_fc")

    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    clf = UniMolRepr(data_type='molecule',
                    remove_hs=False,
                    model_name='unimolv2',  # avaliable: unimolv1, unimolv2
                    model_size='1.1B',  # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                    )
    model_esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()
    test_df = pd.read_csv(args.csv_file)
    organism_set = args.organism_set_path
    temp_set = args.temp_set_path
    test_df = preprocessing(test_df, clf, model_esm, alphabet, batch_converter, cutoff=8)
    test2016_set = GraphDataset(test_df, organism_set, temp_set, dis_threshold=8)
    test_df = test2016_set.data_df.reset_index(drop=True)
    if len(test2016_set) == 0:
        test_df["pred_log_kcat_graphkcat"] = []
        test_df["pred_log_kcat_graphkcat_std"] = []
        test_df["pred_log_km_graphkcat"] = []
        test_df["pred_log_km_graphkcat_std"] = []
        test_df["pred_log_kcat_km_graphkcat"] = []
        test_df["pred_log_kcat_km_graphkcat_std"] = []
        test_df.to_csv(output_dir / "inference_results.csv", index=False)
        print(f"Results saved to {output_dir / 'inference_results.csv'}")
        return
    test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = bottle_view_graph(node_dim=35,
                              hidden_dim=hidden_dim,
                              HeteroGNN_layers=HeteroGNN_layers,
                              pooling=pooling,
                              vocab_size=vocab_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              ligand_nn_embedding=ligand_nn_embedding,
                              num_fc_layers=num_fc_layers,
                              fc_hidden_dim=fc_hidden_dim,
                              share_fc=share_fc
                              )
    load_model_dict(model, args.cpkt_path)
    model = model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    (
        log_pred_kcat,
        std_pred_kcat,
        log_pred_km,
        std_pred_km,
        log_pred_kcat_km,
        std_pred_kcat_km,
    ) = val(
        model,
        test2016_loader,
        device,
        mc_dropout_samples=args.mc_dropout_samples,
        mc_dropout_seed=args.mc_dropout_seed,
    )
    test_df["pred_log_kcat_graphkcat"] = log_pred_kcat
    test_df["pred_log_kcat_graphkcat_std"] = std_pred_kcat
    test_df["pred_log_km_graphkcat"] = log_pred_km
    test_df["pred_log_km_graphkcat_std"] = std_pred_km
    test_df["pred_log_kcat_km_graphkcat"] = log_pred_kcat_km
    test_df["pred_log_kcat_km_graphkcat_std"] = std_pred_kcat_km
    test_df.to_csv(output_dir / "inference_results.csv", index=False)
    print(f"Results saved to {output_dir / 'inference_results.csv'}")

if __name__ == '__main__':
    main()
    
