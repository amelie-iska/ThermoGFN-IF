#!/bin/bash
# Example script for unconditional sequence design on RFdiffusion de novo backbones
# Usage: bash examples/scripts/seq_design_unconditional.sh
python3 fampnn/inference/seq_design.py \
        checkpoint_path=weights/fampnn_0_3.pt \
        pdb_dir=data/denovo500/pdbs \
        pdb_key_list=examples/pdb_key_lists/denovo500_subset.txt \
        out_dir=examples/outputs/seq_design_unconditional
