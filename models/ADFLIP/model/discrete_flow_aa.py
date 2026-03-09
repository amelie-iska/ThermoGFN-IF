import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from data.all_atom_parse import residue_tokens, num_protein_tokens, pdb2data,restype_1to3,make_merged_pdb
import os
from PIPPack.model.modules import PIPPackFineTune
from PIPPack.inference import replace_protein_sequence, pdbs_from_prediction
from PIPPack.data.protein import from_pdb_file
from PIPPack.data.top2018_dataset import transform_structure, collate_fn
from PIPPack.ensembled_inference import sample_epoch
import pickle
import time
import prody
from typing import List

def _is_abnormal_pdb_line(line: str) -> bool:
    """Return *True* if the ATOM / HETATM / ANISOU record is shifted left."""

    if not line.startswith(("ATOM  ", "HETATM", "ANISOU")):
        return False

    # 1. residue name length > 3 (columns 18‑20 plus possible spill‑over)
    res_field = line[17:21]  # cols 18‑21 (0‑based slice 17:21)
    if len(res_field.strip()) > 3:
        return True

    # 2. column 21 (index 20) must be blank in a valid PDB
    if line[20] != " ":
        return True

    # 3. column 23 (index 22) is the first of the 4‑char residue number field
    #    If it contains a minus sign, the record is shifted left.
    if line[22] == "-":
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: clean PDB file in place
# ---------------------------------------------------------------------------

def _clean_pdb_file(pdb_file: str) -> None:
    with open(pdb_file, "r", encoding="utf‑8") as fh:
        good_lines: List[str] = [ln for ln in fh if not _is_abnormal_pdb_line(ln)]
    with open(pdb_file, "w", encoding="utf‑8") as fh:
        fh.writelines(good_lines)


class DiscreteFlow_AA(nn.Module):

    def __init__(self, config, model, min_t=0.0,sidechain_packing=False,sample_save_path='results/sample_seq/',num_sc_models = 1, **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self.min_t = min_t
        self.sample_save_path = sample_save_path
        self.label_smoothing = config.training.label_smoothing
        if sidechain_packing:
            self.sc,self.infer_cfg = self.load_sc_model(device = next(self.model.parameters()).device,num_models = num_sc_models)


    def load_sc_model(self,device,num_models=3):
        model_names = ["pippack_model_1", "pippack_model_2", "pippack_model_3"]
        models = []
        inference_cfg = 'PIPPack/model_weights/inference.pickle'
        with open(inference_cfg, 'rb') as f:
            infer_cfg = pickle.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model_name in model_names[:num_models]:
            cfg_file = f'PIPPack/model_weights/{model_name}_config.pickle'
            ckpt_file = f'PIPPack/model_weights/{model_name}_ckpt.pt'

            with open(cfg_file, 'rb') as f:
                cfg = pickle.load(f)
            
            model = PIPPackFineTune(
                node_features = cfg.model.node_features,
                edge_features = cfg.model.edge_features,
                hidden_dim = cfg.model.hidden_dim,
                num_mpnn_layers = cfg.model.num_mpnn_layers,
                k_neighbors = cfg.model.k_neighbors,
                augment_eps = cfg.model.augment_eps,
                use_ipmp = cfg.model.use_ipmp,
                use_ipmp_ipa = cfg.model.use_ipmp_ipa,
                n_points = cfg.model.n_points,
                dropout = cfg.model.dropout,
                act = cfg.model.act,
                predict_bin_chi = cfg.model.predict_bin_chi,
                n_chi_bins = cfg.model.n_chi_bins,
                predict_offset = cfg.model.predict_offset,
                position_scale = cfg.model.position_scale,
                recycle_strategy = cfg.model.recycle_strategy,
                recycle_SC_D_sc = cfg.model.recycle_SC_D_sc,
                recycle_SC_D_probs = cfg.model.recycle_SC_D_probs,
                recycle_X = cfg.model.recycle_X,
                mask_distances = cfg.model.mask_distances,
                loss = cfg.model.loss,
            )
            state_dicts = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state_dicts['model_state_dict'])
            model = model.to(device)
            models.append(model)
        return models,infer_cfg

    def compute_loss(self, logits, data):
        if self.label_smoothing:
            target = data["residue_token"][data["is_center"] & data["is_protein"]]
            S_onehot = F.one_hot(target, num_classes=logits.size(-1)).float()
            S_onehot = S_onehot + 0.1 / float(S_onehot.size(-1))
            S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(S_onehot * log_probs).sum(-1)
            loss_av = torch.sum(loss) / 2000.0 #fixed 
            return loss_av
        else:
            target = data["residue_token"][data["is_center"] & data["is_protein"]]
            loss = F.cross_entropy(logits, target)
            return loss

    def generate_maskx1(self, data):
        x1_mask = self.alphabet.get_idx("<mask>") * torch.ones_like(
            data["esm_batch_tokens"]
        )  # this keeps x1 as the same dtype as idx
        cls_mask = data["esm_batch_tokens"] == self.alphabet.get_idx("<cls>")
        eos_mask = data["esm_batch_tokens"] == self.alphabet.get_idx("<eos>")
        padding_mask = data["esm_batch_tokens"] == self.alphabet.get_idx("<pad>")
        x1_mask[cls_mask] = self.alphabet.get_idx("<cls>")
        x1_mask[eos_mask] = self.alphabet.get_idx("<eos>")
        x1_mask[padding_mask] = self.alphabet.get_idx("<pad>")
        return x1_mask

    def test(self, dataloader, time=0.0):
        device = next(self.model.parameters()).device
        error = 0
        total_flatten_logits = torch.tensor([])
        total_flatten_true = torch.tensor([])
        total_flatten_interact_non_protein_logits = torch.tensor([])
        total_flatten_interact_non_protein_true = torch.tensor([])

        self.model.eval()
        for data in dataloader:
            try:
                data = {k: v.to(device) for k, v in data.__dict__.items()}
                data = {
                    k: v.float() if v.dtype == torch.float64 else v
                    for k, v in data.items()
                }
                noisy_data = {k: v.clone() for k, v in data.items()}
                noisy_data["residue_token"] = data["noisy_residue_token"]
                times = data["time_step"]
                flatten_logits, _ = self.model(noisy_data, times)

                flatten_true = data["residue_token"][
                    data["is_center"] & data["is_protein"]
                ]

                interact_non_protein = data["interact_non_protein_res"][
                    data["is_center"] & data["is_protein"]
                ].cpu()

                flatten_logits = flatten_logits.detach().cpu()
                flatten_true = flatten_true.detach().cpu()
                total_flatten_logits = torch.cat((total_flatten_logits, flatten_logits))
                total_flatten_true = torch.cat((total_flatten_true, flatten_true))

                total_flatten_interact_non_protein_logits = torch.cat((total_flatten_interact_non_protein_logits, flatten_logits[interact_non_protein]))
                total_flatten_interact_non_protein_true = torch.cat((total_flatten_interact_non_protein_true, flatten_true[interact_non_protein]))
                del flatten_logits, flatten_true



            except Exception as e:
                error +=1
                print(f"Error in test in {e}")
                
        print(f' fail to predict {error}/{len(dataloader)} samples')
        loss = F.cross_entropy(total_flatten_logits, total_flatten_true.long())
        accuracy = (
            (total_flatten_logits.argmax(dim=-1) == total_flatten_true).float().mean()
        )
        perplexity = torch.exp(loss)

        interact_non_protein_loss = F.cross_entropy(total_flatten_interact_non_protein_logits, total_flatten_interact_non_protein_true.long())
        interact_non_protein_accuracy = (
            (total_flatten_interact_non_protein_logits.argmax(dim=-1) == total_flatten_interact_non_protein_true).float().mean()
        )
        interact_non_protein_perplexity = torch.exp(interact_non_protein_loss)


        del total_flatten_logits, total_flatten_true,total_flatten_interact_non_protein_logits, total_flatten_interact_non_protein_true
        return loss, accuracy, perplexity,interact_non_protein_accuracy, interact_non_protein_perplexity

    def corrupt_data_by_sample(self, data, time, sample):
        """
        corrupt data for sampling
        """

        device = data["residue_token"].device
        noisy_data = {k: v.squeeze().clone() for k, v in data.items()}
        noisy_data["residue_token"][
            noisy_data["is_center"] & noisy_data["is_protein"]
        ] = sample
        time = torch.ones((1, 1), device=device) * time

        target = noisy_data["residue_token"][noisy_data["is_center"]]
        target_mask = target == residue_tokens["<MASK>"]

        noisy_residue_token = noisy_data["residue_token"][noisy_data["is_protein"]]

        assert target.shape[0]  > noisy_data["residue_index"][noisy_data["is_protein"]].max().item()

        noisy_residue_token[
            target_mask[noisy_data["residue_index"][noisy_data["is_protein"]]]
        ] = residue_tokens["<MASK>"]
        noisy_data["residue_token"][noisy_data["is_protein"]] = noisy_residue_token

        mask_sidechain = torch.ones_like(noisy_data["residue_token"]).bool()#trigger cuda bug
        target_mask_sidechain = torch.ones_like(target_mask).bool() 
        mask_sidechain[~noisy_data["is_backbone"] & noisy_data["is_protein"]] = (
            ~target_mask[
                noisy_data["residue_index"][
                    ~noisy_data["is_backbone"] & noisy_data["is_protein"]
                ]
            ]
        )

        for name, item in noisy_data.items():
            if name == "time_step":
                noisy_data[name] = time.view(-1, 1)
            elif name == "noisy_residue_token" or name == 'interact_non_protein_res' or name == 'interact_ion_res' or name == 'interact_nucleotide_res' or name == 'interact_molecule_res':
                pass
            else:
                noisy_data[name] = item[mask_sidechain].unsqueeze(0)



        return data, noisy_data

    def sample(self, pdb_path, dt=0.1, argmax_final=True,temp = 0.1,noise=1,mask_interact=False,interact_noise=1):
        self.model.eval()
        device = next(self.model.parameters()).device
        data = pdb2data(pdb_path,device)
        if 'cif' in pdb_path:
            import prody
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.cif','')
            os.makedirs(sample_save_folder,exist_ok=True)
            structure = prody.parseMMCIF(pdb_path)
            atom = structure.select("not water and not hydrogen")
            for chain in atom.getHierView():
                # Split the chain identifier and keep only the letter part
                chain_id = chain.getChid().split('.')[-1]
                chain.setChids(chain_id)
            new_filename = os.path.join(sample_save_folder,pdb_path.split('/')[-1].replace('.cif','.pdb'))
            
            prody.proteins.pdbfile.writePDB(new_filename,atom)
            pdb_path = new_filename
        else:
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.pdb','')
            os.makedirs(sample_save_folder,exist_ok=True)
            os.system(f'cp {pdb_path} {sample_save_folder}')
        true_tokens = data["residue_token"][data["is_center"] & data["is_protein"]]
        t = 0.0
        samples = (
            torch.ones_like(
                data["residue_token"][data["is_center"] & data["is_protein"]],
                device=device,
            ).unsqueeze(0)
            * residue_tokens["<MASK>"]
        )
        B, T = samples.size()
        interact_res_index = data['residue_index'][data['is_protein']][data['interact_non_protein_res']].unique()

        # samples_time = 0
        # fig,axis = plt.subplots(10,2,figsize=(5, 10))
        while t <= 1.0:
            data, noisy_data = self.corrupt_data_by_sample(data, t, samples)
            logits, _ = self.model(noisy_data, torch.tensor([[t]], device=device))

            rr = (logits.argmax(dim=1) == true_tokens).sum()/true_tokens.shape[0]
            print('time:', round(t,3), round(rr.item(),4),'context_num',noisy_data['residue_token'].shape[1])
            


            # print('logits:')
            # print(t,F.softmax(logits,dim=-1).max(1)[0][:100])
            # interact_non_protein_res = data['residue_index'][data['is_protein']][data['interact_non_protein_res']].unique()
            # print('interact_non_protein_res logits')
            # print(F.softmax(logits,dim=-1)[interact_non_protein_res].max(1)[0])



            if round(t,3) >= 1.0 or dt >=1.0:
                if argmax_final:
                    samples_final = logits.argmax(dim=-1).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                    # plt.savefig(f'{sample_save_folder}/'+pdb_path.split('/')[-1].replace('.pdb','')+f'_final_{samples_time}.png',dpi=300,bbox_inches='tight')
                
                
                else:
                    samples_final = torch.multinomial(
                        F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
                    ).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
                return samples,logits

            pt_x1_probs = F.softmax(
                logits, dim=-1
            )  # (B, T, V_size)   self.config.diffusion.x1_temp

            '''
            visualize the distribution of the logits
            '''
            # print(pt_x1_probs.max(1))
            
            # axis[samples_time,0].imshow(pt_x1_probs.detach().cpu().numpy()[:,:23].T,vmin=0,vmax=1,aspect='auto')
            # # axis[samples_time,0].set_title(f'time:{round(t,3)}')
            # copy_x1 = pt_x1_probs.detach().cpu().numpy().copy()
            # non_interact_index = data['residue_index'][data['is_protein']][~data['interact_non_protein_res']].unique().cpu().numpy()
            # copy_x1[non_interact_index] = 0
            # axis[samples_time,1].imshow(copy_x1[:,:23].T,vmin=0,vmax=1,aspect='auto')
            # samples_time += 1
            


            sample_is_mask = (samples == residue_tokens["<MASK>"]).view(B, T, 1).float()

            # print('mask ratio:',sample_is_mask.sum()/sample_is_mask.shape[1])
            # if t != 0:
            #     print('unmask rr:',(samples[~sample_is_mask[:,:,0].bool()] == true_tokens[~sample_is_mask.squeeze().bool()]).sum().item()/true_tokens[~sample_is_mask.squeeze().bool()].shape[0])


            step_probs = (
                dt * pt_x1_probs * ((1 + self.config.diffusion.noise * t) / ((1 - t)))
            )  # (B, T, V_size)

            step_probs = step_probs * sample_is_mask


            # average number of dimensions that get re-masked each timestep
            step_probs += (
                dt
                * (1 - sample_is_mask)
                * F.one_hot(
                    torch.tensor(residue_tokens["<MASK>"]),
                    num_classes=num_protein_tokens,
                )
                .view(1, 1, -1)
                .to(device)
                * noise
            )

            step_probs += (
                dt
                * (1 - sample_is_mask)
                * F.one_hot(
                    torch.tensor(residue_tokens["<MASK>"]),
                    num_classes=num_protein_tokens,
                )
                .view(1, 1, -1)
                .to(device)
                * noise
            )




            # sample_is_interact = torch.zeros_like(sample_is_mask)
            # sample_is_interact[:,interact_res_index.squeeze(),:] =1

            # # print('before adding noise',step_probs[:,sample_is_interact.squeeze().bool(),1])
            # step_probs += (
            #     dt
            #     * sample_is_interact
            #     * F.one_hot(
            #         torch.tensor(residue_tokens["<MASK>"]),
            #         num_classes=num_protein_tokens,
            #     )
            #     .view(1, 1, -1)
            #     .to(device)
            #     * interact_noise
            # )

            # print('after adding noise',step_probs[:,sample_is_interact.squeeze().bool(),1])



            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = 0.0
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = (
                1.0 - torch.sum(step_probs, dim=-1).flatten()
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            no_mask_prob = step_probs.clone()
            no_mask_prob[0,:,residue_tokens['<MASK>']] = 0  

            if argmax_final:
                samples = no_mask_prob.argmax(dim=-1).view(B, T)
            else:
                samples = torch.multinomial(
                    no_mask_prob.view(-1, num_protein_tokens), num_samples=1
                ).view(B, T)

            # 

            data  = self.sc_packing(samples,pdb_path,t,sample_save_folder,models=self.sc,infer_cfg=self.infer_cfg,device=device)
            
            sample_mask = torch.multinomial(
                step_probs.view(-1, num_protein_tokens), num_samples=1
            ).view(B, T) == residue_tokens['<MASK>']

            samples[sample_mask] = residue_tokens['<MASK>']

            if mask_interact:
                samples[:,interact_res_index] = residue_tokens['<MASK>']


            # print(round(t,2),'sample',samples[sample_is_interact[:,:,0].bool()])
            # print(round(t,2),step_probs[:,sample_is_interact.squeeze().bool(),1])

            t = t + dt

        pass


    

    
    def adaptive_sample(self, pdb_path, num_step=10, argmax_final=True,temp = 0.1,noise=1,threshold=0.8,regular_residue =True,fix_mask = None):

        self.model.eval()
        device = next(self.model.parameters()).device
        data = pdb2data(pdb_path,device)
        if 'cif' in pdb_path:
            import prody
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.cif','')
            os.makedirs(sample_save_folder,exist_ok=True)
            structure = prody.parseMMCIF(pdb_path)
            atom = structure.select("not water and not hydrogen")
            for chain in atom.getHierView():
                # Split the chain identifier and keep only the letter part
                chain_id = chain.getChid().split('.')[-1]
                chain.setChids(chain_id)
            new_filename = os.path.join(sample_save_folder,pdb_path.split('/')[-1].replace('.cif','.pdb'))
            
            prody.proteins.pdbfile.writePDB(new_filename,atom)
            _clean_pdb_file(new_filename)
            pdb_path = new_filename
        else:
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.pdb','')
            os.makedirs(sample_save_folder,exist_ok=True)
            os.system(f'cp {pdb_path} {sample_save_folder}')
        true_tokens = data["residue_token"][data["is_center"] & data["is_protein"]]
        true_tokens_onehot = F.one_hot(true_tokens,num_classes=num_protein_tokens).float()
        t = 0.0
        sample_times = 0
        samples = (
            torch.ones_like(
                data["residue_token"][data["is_center"] & data["is_protein"]],
                device=device,
            ).unsqueeze(0)
            * residue_tokens["<MASK>"]
        )
        B, T = samples.size()
        
        while t <= 1.0:
            data, noisy_data = self.corrupt_data_by_sample(data, t, samples)
            logits, _ = self.model(noisy_data, torch.tensor([[t]], device=device))
            
            if regular_residue: #only sample the regular residue
                logits[:,22:] = -float('inf')
                logits[:,0:2] = -float('inf')

            rr =(logits.argmax(dim=1) == true_tokens).sum()/true_tokens.shape[0]
            print('time:', round(t,3), round(rr.item(),4),'context_num',noisy_data['residue_token'].shape[1])

            if round(t,3) >= 1.0 or sample_times >= num_step:
                if argmax_final:
                    samples_final = logits.argmax(dim=-1).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]
                else:
                    samples_final = torch.multinomial(
                        F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
                    ).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
                return samples,logits

            pt_x1_probs = F.softmax(
                logits/temp, dim=-1
            )  # (B, T, V_size)   self.config.diffusion.x1_temp
            if fix_mask is not None:
                pt_x1_probs[fix_mask] = true_tokens_onehot[fix_mask]
            conserve_mask = (pt_x1_probs.max(dim=-1)[0] > threshold).view(B, T)
            conserve_sample = pt_x1_probs.argmax(dim=-1).view(B, T)
            next_t = (conserve_mask.sum()/conserve_mask.shape[1]).item()
            dt = next_t - t

            sample_is_mask = (samples == residue_tokens["<MASK>"]).view(B, T, 1).float()

            step_probs = (
                dt * pt_x1_probs * ((1 + noise * t) / ((1 - t)))
            )  # (B, T, V_size)

            step_probs = step_probs * sample_is_mask

            # average number of dimensions that get re-masked each timestep
            step_probs += (
                dt
                * (1 - sample_is_mask)
                * F.one_hot(
                    torch.tensor(residue_tokens["<MASK>"]),
                    num_classes=num_protein_tokens,
                )
                .view(1, 1, -1)
                .to(device)
                * noise
            )

            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = 0.0
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = (
                1.0 - torch.sum(step_probs, dim=-1).flatten()
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            no_mask_prob = step_probs.clone()
            no_mask_prob[0,:,residue_tokens['<MASK>']] = 0  
            if (no_mask_prob.sum(dim=-1) <= 0).sum() > 0:  
                
                no_mask_prob[no_mask_prob.sum(dim=-1) != 1] = pt_x1_probs.view(B, T,-1)[no_mask_prob.sum(dim=-1) != 1].to(no_mask_prob.dtype)


            samples = torch.multinomial(
                no_mask_prob.view(-1, num_protein_tokens), num_samples=1
            ).view(B, T)
            # samples = no_mask_prob.argmax(dim=-1).view(B, T)



            samples = samples * ~conserve_mask + conserve_sample * conserve_mask

            data  = self.sc_packing(samples,pdb_path,t,sample_save_folder,models=self.sc,infer_cfg=self.infer_cfg,device=device)

            samples[~conserve_mask] = residue_tokens['<MASK>']
            t = t + dt
            sample_times += 1

                

        if argmax_final:
            samples_final = logits.argmax(dim=-1).view(B, T)
            samples_mask = samples == residue_tokens["<MASK>"]
            samples[samples_mask] = samples_final[samples_mask]
        else:
            samples_final = torch.multinomial(
                F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
            ).view(B, T)
            samples_mask = samples == residue_tokens["<MASK>"]
            samples[samples_mask] = samples_final[samples_mask]
        print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
        return samples,logits
    
    def sc_packing(self,samples,pdb_path,t,sample_save_folder,models,infer_cfg,device):    
        protein_name = pdb_path.split('/')[-1].replace('.pdb','')+'_0'
        decode_mapping = {j:i for i,j in residue_tokens.items()}
        restype_3to1 = {v:k for k,v in restype_1to3.items()}
        samples[0][samples[0] > 21] = 10 #replace the non standard amino acid with GlY
        seq = ''.join([restype_3to1[decode_mapping[i.item()]] for i in samples[0]])

        proteins = replace_protein_sequence(vars(from_pdb_file(pdb_path, mse_to_met=True,ignore_non_std = False)), pdb_path, [[seq]])
        proteins = [(protein[0], transform_structure(protein[1], sc_d_mask_from_seq=True)) for protein in proteins]

        batch = collate_fn([proteins[0][1]])

        sample_results = sample_epoch(models, batch, infer_cfg['temperature'], device, n_recycle=1, resample=False, resample_args=infer_cfg['resample_args'])


        protein_strings = pdbs_from_prediction(sample_results)
        save_path = os.path.join(sample_save_folder, f'side_chain_t={round(t,3)}')
        if os.path.exists(save_path) == False:
            os.makedirs(save_path,exist_ok=True)

        
        with open(os.path.join(save_path, protein_name + '.pdb'), 'w') as f:
            f.write(protein_strings[0])

        make_merged_pdb(pdb_path, os.path.join(save_path, protein_name + '.pdb'))
        data = pdb2data(os.path.join(save_path, protein_name + '.pdb'),next(self.model.parameters()).device) 

        return data

    def forward(self, data):
        """
        idx is the corrupted tokens (b, t)
        time is the time in the corruption process (b,)
        targets is the clean data (b, t)
        target_mask is 1.0 for points in the sequence that have been corrupted
            and should have loss calculated on them (b, t)
        do_self_cond_loop is whether to do two passes to train the self conditioning
        """
        # data,noisy_data,times = self.corrupt_token(data)

        noisy_data = {k: v.clone() for k, v in data.items()}
        noisy_data["residue_token"] = data["noisy_residue_token"]
        times = data["time_step"]

        #debug check token
        # print(check_categories(data['residue_token'][data['not_pad_mask']&data['is_center']]))
        b, t = noisy_data["residue_token"].size()
        assert (times < 1.1).all()  # 0 to 1 not 0 to 1000

        logits, _ = self.model(noisy_data, times)
        loss = self.compute_loss(logits, data)
        return logits, loss

    def sample_fix(self, pdb_path, dt=0.1, argmax_final=True,temp = 0.1,noise=1,mask_interact=False,interact_noise=1,regular_residue=True,fix_mask = None):
        self.model.eval()
        device = next(self.model.parameters()).device
        data = pdb2data(pdb_path,device)
        if 'cif' in pdb_path:
            import prody
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.cif','')
            os.makedirs(sample_save_folder,exist_ok=True)
            structure = prody.parseMMCIF(pdb_path)
            atom = structure.select("not water and not hydrogen")
            for chain in atom.getHierView():
                # Split the chain identifier and keep only the letter part
                chain_id = chain.getChid().split('.')[-1]
                chain.setChids(chain_id)
            new_filename = os.path.join(sample_save_folder,pdb_path.split('/')[-1].replace('.cif','.pdb'))
            
            prody.proteins.pdbfile.writePDB(new_filename,atom)
            pdb_path = new_filename
        else:
            sample_save_folder = self.sample_save_path+pdb_path.split('/')[-1].replace('.pdb','')
            os.makedirs(sample_save_folder,exist_ok=True)
            os.system(f'cp {pdb_path} {sample_save_folder}')
        true_tokens = data["residue_token"][data["is_center"] & data["is_protein"]]
        true_tokens_onehot = F.one_hot(true_tokens,num_classes=num_protein_tokens).float()
        t = 0.0
        samples = (
            torch.ones_like(
                data["residue_token"][data["is_center"] & data["is_protein"]],
                device=device,
            ).unsqueeze(0)
            * residue_tokens["<MASK>"]
        )
        B, T = samples.size()
        interact_res_index = data['residue_index'][data['is_protein']][data['interact_non_protein_res']].unique()

        # samples_time = 0
        # fig,axis = plt.subplots(10,2,figsize=(5, 10))
        while t <= 1.0:
            data, noisy_data = self.corrupt_data_by_sample(data, t, samples)
            logits, _ = self.model(noisy_data, torch.tensor([[t]], device=device))

            if regular_residue: #only sample the regular residue
                logits[:,22:] = -float('inf')
                logits[:,0:2] = -float('inf')
            rr = (logits.argmax(dim=1) == true_tokens).sum()/true_tokens.shape[0]
            print('time:', round(t,3), round(rr.item(),4),'context_num',noisy_data['residue_token'].shape[1])
            


            # print('logits:')
            # print(t,F.softmax(logits,dim=-1).max(1)[0][:100])
            # interact_non_protein_res = data['residue_index'][data['is_protein']][data['interact_non_protein_res']].unique()
            # print('interact_non_protein_res logits')
            # print(F.softmax(logits,dim=-1)[interact_non_protein_res].max(1)[0])



            if round(t+dt,3) >= 1.0 or dt >=1.0:
                if argmax_final:
                    samples_final = logits.argmax(dim=-1).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                    # plt.savefig(f'{sample_save_folder}/'+pdb_path.split('/')[-1].replace('.pdb','')+f'_final_{samples_time}.png',dpi=300,bbox_inches='tight')
                
                
                else:
                    samples_final = torch.multinomial(
                        F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
                    ).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                if fix_mask is not None:
                    samples = samples.squeeze()
                    samples[fix_mask] = true_tokens[fix_mask]
                print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
                return samples,logits

            pt_x1_probs = F.softmax(
                logits, dim=-1
            )  # (B, T, V_size)   self.config.diffusion.x1_temp
            if fix_mask is not None:
                pt_x1_probs[fix_mask] = true_tokens_onehot[fix_mask]
            '''
            visualize the distribution of the logits
            '''
            # print(pt_x1_probs.max(1))
            
            # axis[samples_time,0].imshow(pt_x1_probs.detach().cpu().numpy()[:,:23].T,vmin=0,vmax=1,aspect='auto')
            # # axis[samples_time,0].set_title(f'time:{round(t,3)}')
            # copy_x1 = pt_x1_probs.detach().cpu().numpy().copy()
            # non_interact_index = data['residue_index'][data['is_protein']][~data['interact_non_protein_res']].unique().cpu().numpy()
            # copy_x1[non_interact_index] = 0
            # axis[samples_time,1].imshow(copy_x1[:,:23].T,vmin=0,vmax=1,aspect='auto')
            # samples_time += 1
            


            sample_is_mask = (samples == residue_tokens["<MASK>"]).view(B, T, 1).float()

            # print('mask ratio:',sample_is_mask.sum()/sample_is_mask.shape[1])
            # if t != 0:
            #     print('unmask rr:',(samples[~sample_is_mask[:,:,0].bool()] == true_tokens[~sample_is_mask.squeeze().bool()]).sum().item()/true_tokens[~sample_is_mask.squeeze().bool()].shape[0])


            step_probs = (
                dt * pt_x1_probs * ((1 + self.config.diffusion.noise * t) / ((1 - t)))
            )  # (B, T, V_size)

            step_probs = step_probs * sample_is_mask


            # average number of dimensions that get re-masked each timestep
            step_probs += (
                dt
                * (1 - sample_is_mask)
                * F.one_hot(
                    torch.tensor(residue_tokens["<MASK>"]),
                    num_classes=num_protein_tokens,
                )
                .view(1, 1, -1)
                .to(device)
                * noise
            )

            step_probs += (
                dt
                * (1 - sample_is_mask)
                * F.one_hot(
                    torch.tensor(residue_tokens["<MASK>"]),
                    num_classes=num_protein_tokens,
                )
                .view(1, 1, -1)
                .to(device)
                * noise
            )




            # sample_is_interact = torch.zeros_like(sample_is_mask)
            # sample_is_interact[:,interact_res_index.squeeze(),:] =1

            # # print('before adding noise',step_probs[:,sample_is_interact.squeeze().bool(),1])
            # step_probs += (
            #     dt
            #     * sample_is_interact
            #     * F.one_hot(
            #         torch.tensor(residue_tokens["<MASK>"]),
            #         num_classes=num_protein_tokens,
            #     )
            #     .view(1, 1, -1)
            #     .to(device)
            #     * interact_noise
            # )

            # print('after adding noise',step_probs[:,sample_is_interact.squeeze().bool(),1])



            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = 0.0
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = (
                1.0 - torch.sum(step_probs, dim=-1).flatten()
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            no_mask_prob = step_probs.clone()
            no_mask_prob[0,:,residue_tokens['<MASK>']] = 0  

            if argmax_final:
                samples = no_mask_prob.argmax(dim=-1).view(B, T)
            else:
                samples = torch.multinomial(
                    no_mask_prob.view(-1, num_protein_tokens), num_samples=1
                ).view(B, T)

            # 

            data  = self.sc_packing(samples,pdb_path,t,sample_save_folder,models=self.sc,infer_cfg=self.infer_cfg,device=device)
            
            sample_mask = torch.multinomial(
                step_probs.view(-1, num_protein_tokens), num_samples=1
            ).view(B, T) == residue_tokens['<MASK>']

            samples[sample_mask] = residue_tokens['<MASK>']

            if mask_interact:
                samples[:,interact_res_index] = residue_tokens['<MASK>']


            # print(round(t,2),'sample',samples[sample_is_interact[:,:,0].bool()])
            # print(round(t,2),step_probs[:,sample_is_interact.squeeze().bool(),1])

            t = t + dt

        pass