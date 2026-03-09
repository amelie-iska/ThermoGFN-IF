import argparse
import ast
import random
from utils.build_vocab import WordVocab

from utils.utils import compute_pna_degrees,virtual_screening
from torch_geometric.loader import DataLoader
from utils.Kcat_Dataset import *  # data
from utils.protein_init import *
from utils.ligand_init import *
from utils.trainer import Trainer

# Model
from models.model_kcat import KcatNet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--result_path', type=str,default="./RESULT",help='path to save trained model')
parser.add_argument('--epochs', type=int, default=80, help='')
parser.add_argument('--batch_size',type=int,default=16)
args = parser.parse_args()

# seed initialize
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

with open('config_KcatNet.json','r') as f:
    config = json.load(f)
device = torch.device(args.device)

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

## import files
with open('./Dataset/KcatNet_traindf.pkl', "rb") as pkl_file:
    train_df = pickle.load(pkl_file)
with open('./Dataset/KcatNet_validdf.pkl', "rb") as pkl_file:
    valid_df = pickle.load(pkl_file)


df = pd.concat([train_df, valid_df], ignore_index=True)
protein_seqs = list(set(df['Pro_seq'].tolist()))
ligand_smiles = list(set(df['Smile'].tolist()))

protein_dict = protein_init(protein_seqs)
torch.save(protein_dict,'./Dataset/protein.pt' )
#protein_dict = torch.load('./Dataset/protein.pt')

ligand_dict = ligand_init(ligand_smiles)
torch.save(ligand_dict,'./Dataset/ligand.pt' )

torch.cuda.empty_cache()
train_dataset = EnzMolDataset(train_df, ligand_dict, protein_dict)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, follow_batch=['mol_x',  'prot_node_esm'])#follow_batch描述节点信息 用于确保 mini-batch 中的节点特征和目标的顺序与原始图中的节点顺序匹配

valid_dataset = EnzMolDataset(valid_df, ligand_dict, protein_dict)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['mol_x', 'prot_node_esm'])

print('Computing training data degrees for PNA')
prot_deg = compute_pna_degrees(train_dataset)
degree_dict = {'protein_deg':prot_deg}
torch.save(degree_dict, './Dataset/degree.pt')
"""degree_dict = torch.load('./Dataset/degree.pt')
prot_deg = degree_dict['protein_deg']"""

model = KcatNet(prot_deg,mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'],
            prot_evo_channels=config['params']['prot_evo_channels'], hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
            post_layers=config['params']['post_layers'],aggregators=config['params']['aggregators'],scalers=config['params']['scalers'],total_layer=config['params']['total_layer'],
            K = config['params']['K'],heads=config['params']['heads'], dropout=config['params']['dropout'],dropout_attn_score=config['params']['dropout_attn_score'],
            device=device).to(device)

print('start training model'+'-'*50)
engine = Trainer(model=model, lrate=config['optimizer']['lrate'], min_lrate=config['optimizer']['min_lrate'], clip=config['optimizer']['clip'],
                 steps_per_epoch=len(train_loader),total_iters=None,num_epochs=args.epochs,  evaluate_metric= 'rmse',result_path=args.result_path, runid=args.seed, device=device)

engine.train_epoch(train_loader, valid_loader)
print('finished training model')




"""print('loading best checkpoint and predicting test data'+'-'*50)
model.load_state_dict(torch.load(os.path.join(args.result_path,'model_KcatNet.pt')))

with open('./Dataset/KcatNet_testdf.pkl', "rb") as pkl_file:
    test_df = pickle.load(pkl_file)

protein_seqs = list(set(test_df['Pro_seq'].tolist()))
ligand_smiles = list(set(test_df['Smile'].tolist()))

protein_dict = protein_init(protein_seqs)
ligand_dict = ligand_init(ligand_smiles)

test_dataset = EnzMolDataset(test_df, ligand_dict, protein_dict)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['mol_x',  'prot_node_esm'])

reg_preds, reg_truths, eval_reg_result= virtual_screening(model, test_loader, device=args.device)
print('rmse:', eval_reg_result["rmse"], 'pearson:', eval_reg_result["pearson"], "spearman:",eval_reg_result["spearman"], "r2_score:", eval_reg_result["r2_score"],"MAE:",eval_reg_result['mae'])
"""