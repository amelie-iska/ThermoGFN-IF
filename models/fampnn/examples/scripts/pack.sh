#!/bin/bash
# Example script for sidechain packing on CASP15 targets
# Usage: bash examples/scripts/pack.sh
python3 fampnn/inference/pack.py \
        checkpoint_path=weights/fampnn_0_0.pt \
        pdb_dir=data/casp15/pdbs \
        pdb_key_list=examples/pdb_key_lists/casp15_subset.txt \
        fixed_pos_csv=examples/fixed_pos_csvs/casp15_pack.csv \
        out_dir=examples/outputs/pack
