# ReactZyme ETFlow -> Boltz-TS Pipeline

This repository builds ligand templates (SDF conformers) for ReactZyme reactant/product pairs and generates Boltz-TS YAML inputs that include pocket constraints from `./pocket_cache`.

## Requirements
- Conda envs already created:
  - ETFlow: `/home/iska/miniconda3/envs/etflow-gpu`
  - Boltz-TS: `/home/iska/miniconda3/envs/boltz2`
- CUDA + GPU for faster runs (ETFlow and Boltz-TS can run on CPU but will be slow).

## Setup (one-time)
Run from the repo root (`/home/iska/amelie-ai/generate-constraints`):
```
/home/iska/miniconda3/envs/etflow-gpu/bin/pip install -e ./models/ETFlow
/home/iska/miniconda3/envs/boltz2/bin/pip install -e ./models/boltz-ts
```

Optional: set an ETFlow checkpoint cache location (used by the new ETFlow script):
```
export ETFLOW_CACHE_DIR=./output_sdf_templates/etflow_cache
```

## Step 1: Generate ETFlow SDF templates
Creates one SDF per reactant and product, using 5 conformers and keeping the best.
```
/home/iska/miniconda3/envs/etflow-gpu/bin/python scripts/run_etflow_reactzyme_confs.py \
    --csv-dir data/ReactZyme-CSVs \
    --out-dir output_sdf_templates \
    --cache-dir output_sdf_templates/etflow_cache \
    --num-confs 5 \
    --score mmff \
    --splits train,validation,test \
    --seed 42 \
    --log-level INFO \
    --log-every 500 \
    --max-enzyme-length 2048
```
Outputs:
- `output_sdf_templates/train/*.sdf`
- `output_sdf_templates/validation/*.sdf`
- `output_sdf_templates/test/*.sdf`

## Step 2: Generate Boltz-TS YAMLs
Creates paired YAMLs per reaction with pocket constraints and SDF templates.
```
/home/iska/miniconda3/envs/etflow-gpu/bin/python scripts/convert_reactzyme_to_boltz_yaml.py \
  --csv-dir data/ReactZyme-CSVs \
  --sdf-root output_sdf_templates \
  --pocket-cache pocket_cache \
  --output input_rp_yamls \
  --splits train,validation,test
```
Outputs:
- `input_rp_yamls/train/*__reactant.yaml`
- `input_rp_yamls/train/*__product.yaml`
- `input_rp_yamls/validation/*__reactant.yaml`
- `input_rp_yamls/validation/*__product.yaml`
- `input_rp_yamls/test/*__reactant.yaml`
- `input_rp_yamls/test/*__product.yaml`

## Step 3: Run Boltz-TS
Use the CLI from the `boltz2` env. Example for train:
```
/home/iska/miniconda3/envs/boltz2/bin/boltz predict ./input_rp_yamls/train \
  --out_dir ./data/boltz-ts-results/train \
  --output_format mmcif \
  --accelerator gpu --devices 4 \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --diffusion_samples 1 \
  --sampling_steps 300 \
  --step_scale 1.6 \
  --max_parallel_samples 4 \
  --num_workers 32 \
  --preprocessing-threads 96 \
  --ligand_source template \
  --reconcile_ligand \
  --max_msa_seqs 2048 \
  --override
```
Repeat with `./input_rp_yamls/test` (and `validation` if desired).

## Outputs at a glance
- SDF templates: `output_sdf_templates/{split}/*.sdf`
- YAML inputs: `input_rp_yamls/{split}/*.yaml`
- Boltz-TS results: `data/boltz-ts-results/{split}/`

## Notes
- Dot-separated SMILES are treated as a single multi-fragment ligand entity.
- Pocket constraints are taken from `pocket_cache/*.json`.
- If you change SMILES or templates, re-run Steps 1–2.
