#!/bin/bash
# Example script for conditional sequence design on CASP15 targets
# Usage: bash examples/scripts/seq_design_conditional.sh
python3 fampnn/inference/seq_design.py \
        checkpoint_path=weights/fampnn_0_3.pt \
        pdb_dir=data/casp15/pdbs \
        pdb_key_list=examples/pdb_key_lists/casp15_subset.txt \
        fixed_pos_csv=examples/fixed_pos_csvs/casp15_seq_design.csv \
        out_dir=examples/outputs/seq_design_conditional
