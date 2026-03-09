---
license: mit
---
**Github repo: https://github.com/westlake-repl/ProTrek**

## Overview
ProTrek is a multimodal model that integrates protein sequence, protein structure, and text information for better 
protein understanding. It adopts contrastive learning to learn the representations of protein sequence and structure. 
During the pre-training phase, we calculate the InfoNCE loss for each two modalities as [CLIP](https://arxiv.org/abs/2103.00020)
does.

## Model architecture
**Protein sequence encoder**:  [esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

**Protein structure encoder**: foldseek_t12_35M (identical architecture with esm2 except that the vocabulary only contains 3Di tokens)

**Text encoder**:  [BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

## Obtain embeddings and calculate similarity score (please clone our repo first)
```
import torch

from model.ProtTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq

# Load model
config = {
    "protein_config": "weights/ProTrek_35M_UniRef50/esm2_t12_35M_UR50D",
    "text_config": "weights/ProTrek_35M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_35M_UniRef50/foldseek_t12_35M",
    "load_protein_pretrained": False,
    "load_text_pretrained": False,
    "from_checkpoint": "weights/ProTrek_35M_UniRef50/ProTrek_35M_UniRef50.pt"
}

device = "cuda"
model = ProTrekTrimodalModel(**config).eval().to(device)

# Load protein and text
pdb_path = "example/8ac8.cif"
seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"])["A"]
aa_seq = seqs[0]
foldseek_seq = seqs[1].lower()
text = "Replication initiator in the monomeric form, and autogenous repressor in the dimeric form."

with torch.no_grad():
    # Obtain protein sequence embedding
    seq_embedding = model.get_protein_repr([aa_seq])
    print("Protein sequence embedding shape:", seq_embedding.shape)
    
    # Obtain protein structure embedding
    struc_embedding = model.get_structure_repr([foldseek_seq])
    print("Protein structure embedding shape:", struc_embedding.shape)
    
    # Obtain text embedding
    text_embedding = model.get_text_repr([text])
    print("Text embedding shape:", text_embedding.shape)
    
    # Calculate similarity score between protein sequence and structure
    seq_struc_score = seq_embedding @ struc_embedding.T / model.temperature
    print("Similarity score between protein sequence and structure:", seq_struc_score.item())

    # Calculate similarity score between protein sequence and text
    seq_text_score = seq_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein sequence and text:", seq_text_score.item())
    
    # Calculate similarity score between protein structure and text
    struc_text_score = struc_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein structure and text:", struc_text_score.item())
   

"""
Protein sequence embedding shape: torch.Size([1, 1024])
Protein structure embedding shape: torch.Size([1, 1024])
Text embedding shape: torch.Size([1, 1024])
Similarity score between protein sequence and structure: 38.83826446533203
Similarity score between protein sequence and text: 17.90523338317871
Similarity score between protein structure and text: 18.044755935668945
"""
```