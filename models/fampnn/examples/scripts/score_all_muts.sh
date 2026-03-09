#!/bin/bash
# Example script for scoring all mutations for a given PDB
# Usage: bash examples/scripts/score_all_muts.sh
python3 fampnn/inference/score_all_muts.py \
        checkpoint_path=weights/fampnn_0_3_cath.pt \
        pdb_path=data/pdbs/3e1r.pdb \
        out_dir=examples/outputs/score_all_muts \
        batch_size=16
