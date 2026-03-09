import random
from utils.build_vocab import WordVocab
from torch_geometric.loader import DataLoader
from utils.Kcat_Dataset import *  
from utils.protein_init import *
from utils.ligand_init import *
from utils.trainer import *
# Model
from models.model_kcat import KcatNet



parser = argparse.ArgumentParser()
parser.add_argument('--file_path',type=str, default='./examples/example.xlsx')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--batch_size',type=int,default=1)
args = parser.parse_args()

with open('config_KcatNet.json','r') as f:
    config = json.load(f)
device = torch.device(args.device)


df = pd.read_excel(args.file_path)
protein_seqs = list(set(df['Pro_seq'].tolist()))
ligand_smiles = list(set(df['Smile'].tolist()))
protein_dict = protein_init(protein_seqs)
ligand_dict = ligand_init(ligand_smiles)

torch.cuda.empty_cache()
dataset = EnzMolDataset(df, ligand_dict, protein_dict)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['mol_x', 'prot_node_esm'])

print('Computing training data degrees for PNA')
degree_dict = torch.load('./Dataset/degree.pt')
prot_deg = degree_dict['protein_deg']

model = KcatNet(prot_deg,mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'],
            prot_evo_channels=config['params']['prot_evo_channels'], hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
            post_layers=config['params']['post_layers'],aggregators=config['params']['aggregators'],scalers=config['params']['scalers'],total_layer=config['params']['total_layer'],
            K = config['params']['K'],heads=config['params']['heads'], dropout=config['params']['dropout'],dropout_attn_score=config['params']['dropout_attn_score'],
            device=device).to(device)


print('loading best checkpoint and predicting test data'+'-'*50)
model.load_state_dict(torch.load('./RESULT/model_KcatNet.pt'))

reg_preds= pred(model, data_loader, device=args.device)
df['Predicted Kcats'] = [math.pow(10, Kcat_log_value) for Kcat_log_value in reg_preds]
df.to_excel(args.file_path, index=False)
