# PLAN-1: ReactZyme -> ETFlow Conformers -> Boltz-TS YAMLs

## Goal
Generate ETFlow SDF conformers for reactant/product pairs in the ReactZyme CSV splits, keep each pair linked by a shared ID, and then create Boltz-TS YAML inputs that reference those SDF templates and use pocket constraints from `./pocket_cache`. The pipeline must:

- Preserve the train/validation/test split in `./data/ReactZyme-CSVs/`.
- Treat dot-separated SMILES as a single multi-fragment ligand entity (no splitting).
- Generate 5 ETFlow conformers per ligand and select the best conformer.
- Output YAMLs to `./input_rp_yamls/{split}/` with names like `{pair_id}__reactant.yaml` and `{pair_id}__product.yaml`.

## Current Relevant Paths
- Inputs
  - `data/ReactZyme-CSVs/reactzyme_{train,validation,test}.csv`
  - `pocket_cache/*.json` (UniProt feature JSONs)
- Outputs
  - `output_sdf_templates/{split}/` (ETFlow SDFs to be created)
  - `input_rp_yamls/{split}/` (Boltz-TS YAMLs to be created)
- Code
  - `models/ETFlow/` (ETFlow package + configs)
  - `scripts/convert_reactzyme_to_boltz_yaml.py` (needs updates)
  - `models/boltz-ts/docs/prediction.md` (Boltz-TS CLI usage)

## Implementation Steps

### 1) Create an ETFlow conformer generation script
File: `scripts/run_etflow_reactzyme_confs.py` (new)

Responsibilities
- Read all ReactZyme CSVs and iterate per split.
- For each row, read `reactant_smiles` and `product_smiles` (use `dataset_smiles` only as a fallback if needed).
- Treat dot-separated SMILES as a single entity (pass full string to RDKit/ETFlow).
- Build a stable pair ID that keeps reactant/product together:
  - `pair_id = f"{split}__{row_id}__{rhea_id_sanitized}"`.
  - If `rhea_id` is empty, fallback to `pair_id = f"{split}__{row_id}"`.
  - If duplicates still exist, append a running index.
- Generate N=5 conformers per ligand using ETFlow:
  - Use `BaseFlow.from_default(model="drugs-o3", cache=<cache_dir>)`.
  - Use `MoleculeFeaturizer.get_data_from_smiles()` + `batched_sampling`.
  - Seed and log for reproducibility.
- Pick the best conformer (default: lowest energy):
  - Use RDKit MMFF94 (fallback to UFF).
  - If MMFF/UFF fails, fallback to first conformer.
- Write a single-conformer SDF per ligand:
  - `output_sdf_templates/{split}/{pair_id}__reactant.sdf`
  - `output_sdf_templates/{split}/{pair_id}__product.sdf`
- Skip generation if output SDF already exists (resume support).
- Write a manifest per split with:
  - `pair_id`, `row_id`, `rhea_id`, `uniprot_id`, `reactant_smiles`, `product_smiles`, `reactant_sdf`, `product_sdf`, `status`.

Planned CLI
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
    --log-every 500
```

Edge cases / safeguards
- Validate SMILES parsing; log failures to `output_sdf_templates/{split}/errors.csv`.
- Handle `*` dummy atoms and multi-fragment SMILES (keep intact).
- Avoid over-writing; add `--clean` flag to clear outputs if needed.

### 2) Update YAML generation for Boltz-TS
File: `scripts/convert_reactzyme_to_boltz_yaml.py` (modify)

Changes
- Add CSV-based pipeline that uses `data/ReactZyme-CSVs/*.csv` (not the `.pt` files).
- Add `--csv-dir`, `--sdf-root`, `--splits` args.
- Update YAML writer to include templates:
  - Use `sdf` from `output_sdf_templates/{split}/{pair_id}__reactant.sdf` or `__product.sdf`.
  - Set `chain_id` = `"R"` or `"P"`, `atom_map` = `"identical"`, `force` = `true`, `threshold` = `0.1` (configurable).
- Use pocket constraints from `pocket_cache`:
  - Map `uniprot_id` from CSV to `pocket_cache/{uniprot_id}.json`.
  - Extract positions using existing `extract_positions_from_features`.
  - Build `contacts` as `[[A, pos], ...]` (FlowList in YAML).
- Output path format:
  - `input_rp_yamls/{split}/{pair_id}__reactant.yaml`
  - `input_rp_yamls/{split}/{pair_id}__product.yaml`
- Add `--clean-output` to reset `input_rp_yamls`.
- Add validation checks:
  - Warn on missing SDFs or missing pockets.
  - Optionally `--require-sdf` to fail on missing templates.

Planned CLI
```
/home/iska/miniconda3/envs/etflow-gpu/bin/python scripts/convert_reactzyme_to_boltz_yaml.py \
  --csv-dir data/ReactZyme-CSVs \
  --sdf-root output_sdf_templates \
  --pocket-cache pocket_cache \
  --output input_rp_yamls \
  --splits train,validation,test
```

### 3) Add root README with end-to-end commands
File: `README.md` (new)

Include:
- Environment paths for `etflow-gpu` and `boltz2`.
- Install instructions for ETFlow and Boltz-TS packages:
  - `/home/iska/miniconda3/envs/etflow-gpu/bin/pip install -e ./models/ETFlow`
  - `/home/iska/miniconda3/envs/boltz2/bin/pip install -e ./models/boltz-ts`
- Step-by-step pipeline:
  1) Run ETFlow conformer generation.
  2) Convert to YAMLs with pockets + SDF templates.
  3) Run Boltz-TS predictions (train + test).
- Output directories and expected artifacts.

### 4) Run Boltz-TS
README instructions should include:
- `boltz predict` usage per `models/boltz-ts/docs/prediction.md`.
- Example command with paths to new YAML directories:
```
/home/iska/miniconda3/envs/boltz2/bin/boltz predict ./input_rp_yamls/train \
  --out_dir ./data/boltz-ts-results/train \
  --output_format mmcif \
  --accelerator gpu --devices 1 \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --diffusion_samples 10 \
  --sampling_steps 300 \
  --step_scale 1.45 \
  --max_parallel_samples 2 \
  --num_workers 12 \
  --preprocessing-threads 24 \
  --ligand_source template \
  --reconcile_ligand \
  --max_msa_seqs 2048 \
  --override
```
- Repeat for `./input_rp_yamls/test` (and `validation` if desired).

## Validation Checklist
- Count of generated SDFs matches CSV rows per split (reactant + product per row).
- Every YAML has a valid SDF template path.
- Pocket constraints present for entries with pocket cache.
- Spot-check: open a few SDFs in RDKit and confirm 3D coordinates exist.

## Deliverables
- `scripts/run_etflow_reactzyme_confs.py` (new)
- Updated `scripts/convert_reactzyme_to_boltz_yaml.py`
- `planning/PLAN-1.md` (this file)
- `README.md` with end-to-end CLI instructions
