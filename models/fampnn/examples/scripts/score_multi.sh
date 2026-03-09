#!/bin/bash
# Example script for scoring specific mutations for a given PDB
# Usage: bash examples/scripts/score_multi.sh
python3 fampnn/inference/score_multi.py \
        checkpoint_path=weights/fampnn_0_3_cath.pt \
        pdb_path=data/pdbs/3e1r.pdb \
        mutations_path=examples/scoring/mutations.csv \
        out_dir=examples/outputs/score_multi \
        batch_size=16
