# ThermoGFN-IF

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Blog%20Post-ffcc4d)](https://huggingface.co/blog/AmelieSchreiber/thermogfn-if)

ThermoGFN-IF implementation scaffold for tri-fidelity protein design with Method III-first training. 

For details, see the paper in 
```bash
./planning/ThermoGFN-IF.tex
```
The paper does not yet include kinetic parameter oracle GFlowNets style RL, which is repo only at present. 

## Required conda environments

- `ligandmpnn_env`
- `spurs`
- `bioemu`
- `uma-qc`

Optional (legacy generator backend):

- `ADFLIP`

Check readiness:

```bash
./scripts/env/check_envs.sh runs/env_status.json
# optional deep checks
RUN_HEALTH_CHECKS=1 ./scripts/env/check_envs.sh runs/env_status_health.json
```

Prefetch production oracle assets into a writable local cache before training:

```bash
./scripts/env/prefetch_production_oracles.sh
```

BioEmu requires a ColabFold runtime for embedding generation. Provision/check it explicitly:

```bash
./scripts/env/setup_bioemu_colabfold_runtime.sh
./scripts/env/setup_bioemu_colabfold_runtime.sh --check-only
```

If `bioemu` was created with Python 3.12 and ColabFold setup fails, rebuild `bioemu` on Python 3.11:

```bash
./scripts/env/rebuild_bioemu_env_py311.sh --env-name bioemu
```

## Data status

Current ready split:

- canonical path: `rfd3-data/rfd3_splits/unconditional_monomer_protrek35m`
- legacy alias still accepted by the prep scripts: `data/rfd3_splits/unconditional_monomer_protrek35m`

Future splits can be plugged in when generated under `rfd3-data/rfd3_splits/` or the legacy `data/rfd3_splits/` alias.

## Configuration (single source of truth)

Primary runtime config:

- `config/m3_default.yaml`

This now includes:

- generator backend and LigandMPNN generation controls (`batch_size`, `number_of_batches`, `temperature`, atom-context flags),
- Method III round controls (`pool_size`, `bioemu_budget`, `uma_budget`, checkpoint retention),
- teacher/student/surrogate knobs (`surrogate_ensemble_size`, `teacher_steps`, `teacher_gamma_off`, `student_steps`),
- oracle settings (`spurs repo/chain`, `bioemu model+num_samples`, `uma model+workers+replicates`),
- BioEmu VRAM-aware batching (`oracles.bioemu.batch_size_100`, `oracles.bioemu.auto_batch_from_vram`, `oracles.bioemu.target_vram_frac`, min/max bounds),
- periodic test sizing and final inference selection (`periodic_eval.num_candidates`, `inference.final.num_candidates`, `inference.final.top_k`).

All updated orchestration scripts accept `--config` and allow CLI overrides.

## Bootstrap pipeline (D0 creation)

```bash
python scripts/prep/01_validate_monomer_split.py \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --output runs/bootstrap/validate_report.json

python scripts/prep/02_build_training_index.py \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --output runs/bootstrap/design_index.jsonl

# when additional mixed-modality splits are available:
python scripts/prep/02_build_training_index.py \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --split-root rfd3-data/rfd3_splits/<future_ppi_split> \
  --split-root rfd3-data/rfd3_splits/<future_ligand_split> \
  --allow-missing \
  --output runs/bootstrap/design_index_multi.jsonl

python scripts/prep/03_compute_baselines.py \
  --config config/m3_default.yaml \
  --index-path runs/bootstrap/design_index.jsonl \
  --output runs/bootstrap/baselines.jsonl \
  --run-id bootstrap \
  --generator-backend ligandmpnn \
  --ligandmpnn-env ligandmpnn_env \
  --ligandmpnn-model-type ligand_mpnn

python scripts/prep/05_materialize_round0_dataset.py \
  --baselines runs/bootstrap/baselines.jsonl \
  --output-train runs/bootstrap/D_0_train.jsonl \
  --output-test runs/bootstrap/D_0_test.jsonl \
  --output-all runs/bootstrap/D_0_all.jsonl
```

## Method III round run

Single round:

```bash
python scripts/orchestration/m3_run_round.py \
  --config config/m3_default.yaml \
  --run-id m3_demo \
  --round-id 0 \
  --dataset-path runs/bootstrap/D_0_train.jsonl \
  --output-dir runs/m3_demo/round_000 \
  --pool-size 2048 \
  --bioemu-budget 256 \
  --uma-budget 64 \
  --env-status-json runs/env_status_health.json \
  --require-ready \
  --strict-gates
```

Multi-round experiment with periodic train/test monitoring:

```bash
python scripts/orchestration/m3_run_experiment.py \
  --config config/m3_default.yaml \
  --run-id m3_demo \
  --dataset-path runs/bootstrap/D_0_train.jsonl \
  --dataset-test-path runs/bootstrap/D_0_test.jsonl \
  --output-root runs/m3_demo \
  --num-rounds 3 \
  --env-status-json runs/env_status_health.json \
  --require-ready \
  --strict-gates
```

This writes per-round periodic metrics and overfitting diagnostics:

- `round_metrics.json`
- `periodic_test_eval.json`
- `periodic_overfit.json`

One-command full production pipeline (downloads, bootstrap, training, inference, evaluation):

```bash
scripts/orchestration/run_full_production_pipeline.sh \
  --config config/m3_default.yaml \
  --run-id thermogfn_prod \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --rounds 8 \
  --pool-size 50000 \
  --bioemu-budget 512 \
  --uma-budget 64
```

## Kcat-only disjoint stage (KcatNet + GraphKcat)

Kcat mode is wired as a separate Method III loop and does not call SPURS/BioEmu/UMA.

Primary config:

- `config/kcat_m3_default.yaml`

Default oracle environment bindings:

- `KcatNet` oracle wrapper (`scripts/prep/oracles/kcatnet_score.py`) runs in conda env `KcatNet`
- `GraphKcat` oracle wrapper (`scripts/prep/oracles/graphkcat_score.py`) runs in conda env `apodock`

These defaults are controlled by `config/kcat_m3_default.yaml` under:

```yaml
oracles:
  envs:
    kcatnet: KcatNet
    graphkcat: apodock
```

Environment setup and readiness for Kcat stage:

```bash
# manual step-by-step path

# one-time KcatNet env create/repair
./scripts/env/create_kcatnet_env.sh --env-name KcatNet --python 3.10 --cuda 12.1

# one-time GraphKcat env create/repair
./scripts/env/create_graphkcat_env.sh --env-name apodock --python 3.10 --cuda 12.1

# optional explicit fallback solver
./scripts/env/create_graphkcat_env.sh --env-name apodock --solver classic

# optional tighter control over retry behavior
./scripts/env/create_graphkcat_env.sh --env-name apodock --attempts 6 --retry-sleep 20

# environment presence check
./scripts/env/check_kcat_envs.sh runs/env_status_kcat.json ligandmpnn_env KcatNet apodock

# preferred: repair both Kcat oracle envs in one shot
./scripts/env/repair_kcat_envs.sh --kcatnet-solver classic --graphkcat-solver classic

# strict health check for the Kcat oracle envs used by dispatch gating
RUN_HEALTH_CHECKS=1 ./scripts/env/check_kcat_envs.sh runs/env_status_kcat_health.json KcatNet apodock

# optional: include LigandMPNN in the strict gate as well
RUN_HEALTH_CHECKS=1 ./scripts/env/check_kcat_envs.sh runs/env_status_kcat_health.json ligandmpnn_env KcatNet apodock
```

Kcat environment notes:

- `repair_kcat_envs.sh` now defaults the Kcat oracle repairs to the classic solver and keeps the GPU torch/CUDA stack.
- The `apodock` / `graphkcat` health path preloads `$CONDA_PREFIX/lib/libLLVM-15.so` when present to avoid the `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` failure seen on some hosts.
- `repair_kcat_envs.sh` does not force the LigandMPNN deep health gate unless you pass `--include-ligandmpnn-check`.

Kcat data requirement:

- Kcat runs require one of `substrate_smiles`, `Smiles`, `smiles`, or `ligand_smiles` on every training/test record.
- The current unconditional monomer split under `rfd3-data/rfd3_splits/unconditional_monomer_protrek35m` does not include this metadata, so Kcat pipeline runs against that split need an explicit metadata overlay.

If your split specs do not carry those fields, pass a metadata overlay when building the Kcat pipeline:

```bash
scripts/orchestration/run_full_kcat_pipeline.sh \
  --config config/kcat_m3_default.yaml \
  --run-id thermogfn_kcat \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --metadata-overlay path/to/kcat_metadata_overlay.jsonl \
  --rounds 8 \
  --pool-size 50000 \
  --kcatnet-budget 1024 \
  --graphkcat-budget 256
```

Run oracle wrappers directly (for debugging/validation):

```bash
# KcatNet scoring
python scripts/env/dispatch.py \
  --env-name KcatNet \
  --require-ready \
  --env-status-json runs/env_status_kcat_health.json \
  --cmd "python scripts/prep/oracles/kcatnet_score.py \
    --candidate-path runs/kcat_debug/candidates.jsonl \
    --output-path runs/kcat_debug/kcatnet_scored.jsonl \
    --model-root models/KcatNet \
    --checkpoint models/KcatNet/RESULT/model_KcatNet.pt \
    --config-path models/KcatNet/config_KcatNet.json \
    --degree-path models/KcatNet/Dataset/degree.pt \
    --device cuda:0 \
    --batch-size 8 \
    --std-default 0.25"

# GraphKcat scoring
python scripts/env/dispatch.py \
  --env-name apodock \
  --require-ready \
  --env-status-json runs/env_status_kcat_health.json \
  --cmd "python scripts/prep/oracles/graphkcat_score.py \
    --candidate-path runs/kcat_debug/kcatnet_scored.jsonl \
    --output-path runs/kcat_debug/graphkcat_scored.jsonl \
    --model-root models/GraphKcat \
    --checkpoint models/GraphKcat/checkpoint/paper.pt \
    --cfg TrainConfig_kcat_enz \
    --batch-size 2 \
    --device cuda:0"

# Fuse KcatNet + GraphKcat into a single acquisition/reward signal
python scripts/prep/oracles/fuse_kcat_scores.py \
  --candidate-path runs/kcat_debug/graphkcat_scored.jsonl \
  --output-path runs/kcat_debug/kcat_fused.jsonl
```

Single Kcat round:

```bash
python scripts/orchestration/kcat_m3_run_round.py \
  --config config/kcat_m3_default.yaml \
  --run-id kcat_demo \
  --round-id 0 \
  --dataset-path runs/bootstrap/D_0_train.jsonl \
  --output-dir runs/kcat_demo/round_000 \
  --pool-size 50000 \
  --kcatnet-budget 1024 \
  --graphkcat-budget 256 \
  --kcatnet-env-name KcatNet \
  --graphkcat-env-name apodock \
  --env-status-json runs/env_status_kcat_health.json \
  --require-ready \
  --strict-gates
```

Multi-round Kcat experiment with periodic train/test overfit tracking:

```bash
python scripts/orchestration/kcat_m3_run_experiment.py \
  --config config/kcat_m3_default.yaml \
  --run-id kcat_demo \
  --dataset-path runs/bootstrap/D_0_train.jsonl \
  --dataset-test-path runs/bootstrap/D_0_test.jsonl \
  --output-root runs/kcat_demo \
  --num-rounds 8 \
  --kcatnet-env-name KcatNet \
  --graphkcat-env-name apodock \
  --env-status-json runs/env_status_kcat_health.json \
  --require-ready \
  --strict-gates
```

Dry-run the full Kcat orchestration command chain (no training/oracle execution):

```bash
python scripts/orchestration/kcat_m3_run_experiment.py \
  --config config/kcat_m3_default.yaml \
  --run-id kcat_dryrun \
  --dataset-path runs/bootstrap/D_0_train.jsonl \
  --dataset-test-path runs/bootstrap/D_0_test.jsonl \
  --output-root runs/kcat_dryrun \
  --num-rounds 1 \
  --dry-run
```

One-command end-to-end Kcat pipeline:

```bash
scripts/orchestration/run_full_kcat_pipeline.sh \
  --config config/kcat_m3_default.yaml \
  --run-id thermogfn_kcat \
  --split-root rfd3-data/rfd3_splits/unconditional_monomer_protrek35m \
  --metadata-overlay path/to/kcat_metadata_overlay.jsonl \
  --rounds 8 \
  --pool-size 50000 \
  --kcatnet-budget 1024 \
  --graphkcat-budget 256
```

## RosettaFold3 / Foundry ReactZyme docking

The repo now includes a local Foundry RF3 workflow for the ReactZyme ligand-template dataset under `generate-constraints_0`.

What this flow does:

- builds separate RF3 JSON inputs for reactant and product docking states,
- uses the ETFlow-generated SDF templates from `generate-constraints_0/output_sdf_templates/train`,
- keeps the original multi-fragment (`.` separated) SMILES in JSON metadata,
- filters to enzyme sequences with length `<= 600`,
- filters each ligand SDF template to at most `256` total atoms,
- caps each exact enzyme sequence at `2` accepted docking pairs across different reactions,
- emits Boltz-style pocket constraint blocks and routes them into RF3 inference-time token-pair threshold conditioning.

The default strict builder uses:

- `status == reactant:ok|product:ok`,
- sequence present in `generate-constraints_0/data/reactzyme_data_split/cleaned_uniprot_rhea.tsv`,
- pocket annotations present in `generate-constraints_0/pocket_cache`,
- ligand template atom count `<= 256` for every emitted reactant/product state,
- no more than `2` accepted docking pairs for the same exact protein sequence,
- existing reactant and product SDFs.

On the current local snapshot, that yields a clean subset of `198` source rows.

Foundry environment notes:

- Foundry requires Python `>= 3.12`.
- `bash scripts/env/create_foundry_rf3_env.sh` prefers `uv` when available and falls back to `python -m venv` when `uv` is missing.
- In this checkout, the Foundry helper wrappers should be invoked with `bash scripts/...` instead of `./scripts/...`.

### 1. Create a repo-local Foundry RF3 environment

```bash
bash scripts/env/create_foundry_rf3_env.sh --env-tool venv --python python3.12

# optional: also install RF3 checkpoints into ./weights
bash scripts/env/create_foundry_rf3_env.sh \
  --env-tool venv \
  --python python3.12 \
  --install-checkpoints \
  --checkpoint-dir ./weights

# validate imports and optional paths
bash scripts/env/check_foundry_rf3_env.sh \
  --checkpoint rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa
```

This uses a repo-local virtualenv under `.venvs/foundry-rf3` rather than a conda env.

### 2. One-time MMSeqs2-GPU workspace setup

The RF3 MSA preparation step reuses the MMSeqs2-GPU workspace from the sibling `../enzyme-quiver` repo.

The shared path `../enzyme-quiver/MMseqs2/local_msa` is a symlink to:

- `/opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa`

If that target does not exist yet, initialize it once with:

```bash
bash scripts/env/setup_local_mmseqs2_uniref100_workaround.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --gpu-binary \
  --create-index \
  --gpu-index \
  --threads "$(nproc)"
```

This wrapper generates and runs a patched temporary copy of `../enzyme-quiver/scripts/setup_local_mmseqs2_uniref100.sh`, which avoids a clone-order bug in the sibling script when `ColabFold` already exists as a stale non-git directory. It writes the local server config, clones the required helper repos, downloads/builds the UniRef100 DB, and creates the padded GPU index used by `mmseqs-server`.

### 3. Start the shared MMSeqs2-GPU server

```bash
bash ../enzyme-quiver/scripts/start_local_mmseqs2_server.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --gpu-backend
```

The default local server URL expected by the RF3 prep scripts is `http://127.0.0.1:8080`.

### 4. Build RF3 JSON inputs from ReactZyme

```bash
source .venvs/foundry-rf3/bin/activate

python scripts/rf3/build_reactzyme_rf3_inputs.py \
  --source-root generate-constraints_0 \
  --output-root runs/rf3_reactzyme_inputs \
  --max-seq-len 600
```

The builder defaults also enforce:

- `--max-ligand-atoms 256`
- `--max-pairs-per-sequence 2`

Outputs:

- per-example JSONs under `runs/rf3_reactzyme_inputs/examples/reactant` and `.../product`
- aggregate JSON lists `runs/rf3_reactzyme_inputs/reactant.json` and `.../product.json`
- `runs/rf3_reactzyme_inputs/manifest.jsonl`
- `runs/rf3_reactzyme_inputs/summary.json`

The emitted JSONs use:

- protein chain `A`
- ligand chain `B`
- whole-ligand SDF templating via `ground_truth_conformer_selection=["B"]`
- Boltz-style `constraints[].pocket` records with `max_distance`

### 5. Generate local MSAs and run Foundry RF3

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs \
  --prepared-root runs/rf3_reactzyme_inputs_with_msa \
  --out-root runs/rf3_reactzyme_out \
  --ckpt-path rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --reuse-cache
```

The wrapper first prepares local MSAs and attaches `msa_path`, then runs Foundry RF3 on both reactant and product JSON bundles.

Useful RF3 overrides can be passed through with repeated `--hydra-override`, for example:

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs \
  --prepared-root runs/rf3_reactzyme_inputs_with_msa \
  --out-root runs/rf3_reactzyme_out \
  --ckpt-path ./weights/rf3_foundry_01_24_latest_remapped.ckpt \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --reuse-cache \
  --hydra-override verbose=true
```

### Pocket constraint format

The RF3 input JSONs now accept Boltz-style pocket constraints:

```json
{
  "constraints": [
    {
      "pocket": {
        "binder": "B",
        "contacts": [["A", 465]],
        "max_distance": 8.0,
        "force": true
      }
    }
  ]
}
```

When present, RF3 converts these constraints into inference-time token-pair threshold conditioning. When absent, RF3 keeps its previous behavior.

Kcat records must include substrate chemistry metadata (`substrate_smiles` or `Smiles`) and structural pointers (`protein_path`/`cif_path`, optional `ligand_path`) for oracle inference.

## Inference workflow

```bash
python scripts/infer/generate_unconditioned.py \
  --student-ckpt runs/m3_demo/round_000/models/student_round_0.ckpt \
  --seed-dataset runs/bootstrap/D_0_train.jsonl \
  --output-path runs/infer/candidates.jsonl \
  --run-id infer_demo \
  --num-candidates 256

python scripts/infer/rescore_and_select.py \
  --input-path runs/infer/candidates.jsonl \
  --output-path runs/infer/top_candidates.jsonl \
  --top-k 32
```

Notes:

- All oracle and orchestration paths are production-only.
- Runtime observability is enabled by default across prep/train/oracle/inference scripts:
  - timestamped structured logs (`INFO` default),
  - tqdm progress bars on candidate/record loops,
  - per-step timing in `m3_run_round.py` and `m3_run_experiment.py`.
- Common controls:
  - `--log-level DEBUG|INFO|WARNING|ERROR`,
  - `--no-progress` to disable tqdm bars per script,
  - `THERMOGFN_NO_PROGRESS=1` to disable tqdm globally.
- Any unavailable dependency (weights, model cache, permissions, network, environment) now fails immediately.
- Required runtime prerequisites for successful round execution:
  - writable Hugging Face cache directory for `spurs` and `bioemu` environments (or pre-populated local model assets),
  - network access to model artifact hosts unless caches are already populated,
  - valid FAIRChem/UMA runtime in `uma-qc`.

## Existing generation utilities

Existing RF3 generation/split scripts are retained:

- `scripts/run_rfd3_inference.sh`
- `scripts/protrek_cluster_split.py`
- `scripts/run_protrek_split_unconditional_monomer.sh`
