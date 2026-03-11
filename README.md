# ThermoGFN-IF

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Blog%20Post-ffcc4d)](https://huggingface.co/blog/AmelieSchreiber/thermogfn-if) [![bioRxiv](https://img.shields.io/badge/bioRxiv-Preprint-007a33.svg)](https://www.biorxiv.org/) [![Paper](https://img.shields.io/badge/PDF-Download%20Paper-blue)](./assets/paper/main.pdf)

ThermoGFN-IF implementation scaffold for multi-fidelity protein design with Method III-first training. 

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
- in SMILES mode, skips ligand pairs with dummy atoms (`*`) or ligands that fail RDKit 3D embedding instead of substituting atoms,
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

The preferred database on this host is the already prepared GPU-ready UniRef30 bundle at:

- `/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302`

If the shared `local_msa` workspace is missing its helper repos or MMSeqs binaries, install just the workspace pieces first and skip any DB download:

```bash
bash scripts/env/setup_local_mmseqs2_uniref100_workaround.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --gpu-binary \
  --skip-db-download
```

Then point that workspace at the existing UniRef30 DB and optionally remove accidental local UniRef100 artifacts:

```bash
bash scripts/env/configure_local_mmseqs2_uniref30.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --uniref30-root /opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302 \
  --cleanup-uniref100
```

`configure_local_mmseqs2_uniref30.sh` writes `config.uniref30.json` and a compatibility `config.uniref100.json` in the local workspace, both pointing at the existing UniRef30 DB. It does not download or rebuild the database.

If the existing padded UniRef30 bundle is missing MMSeqs GPU `.idx` artifacts,
the configurator now runs a one-time `mmseqs createindex` repair in place
before writing the config. On this host that repair takes a couple of minutes
and is required for `gpuserver` to preload the UniRef30 index successfully.

If you truly need to build a fresh database from scratch, the `setup_local_mmseqs2_uniref100_workaround.sh` path still exists, but it is not the preferred workflow for the current RF3 setup.

### 3. Start the shared MMSeqs2-GPU server

```bash
bash scripts/env/start_local_mmseqs2_uniref30_server.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --uniref30-root /opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302 \
  --cleanup-uniref100
```

The UniRef30 helper now defaults to a faster local server config:

- `--local-workers 4`
- `--parallel-databases 2`
- `--parallel-stages`
- `--cuda-devices 0,1,2,3`

The MMSeqs GPU backend uses all visible GPUs. The wrapper now makes that
explicit by exporting `CUDA_VISIBLE_DEVICES=0,1,2,3` by default before starting
`mmseqs-server`.

When the server is healthy on this host, `nvidia-smi` should show roughly
`13-14 GiB` resident on each of the four L40S GPUs from the preloaded UniRef30
GPU index, before any RF3 MSA jobs are submitted.

The default local server URL expected by the RF3 prep scripts is `http://127.0.0.1:8080/api`.

If a previous RF3/MMSeqs run left stale jobs queued, stop the server, clear the
job queue, and restart it before launching a new large RF3 batch:

```bash
kill "$(cat ../enzyme-quiver/MMseqs2/local_msa/run/mmseqs-server.pid)"
rm -f ../enzyme-quiver/MMseqs2/local_msa/run/mmseqs-server.pid
rm -rf /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa/jobs/*

bash scripts/env/start_local_mmseqs2_uniref30_server.sh \
  --msa-root /opt/dlami/nvme/enzyme-quiver/MMseqs2/local_msa \
  --uniref30-root /opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302 \
  --cleanup-uniref100 \
  --local-workers 4 \
  --parallel-databases 2 \
  --parallel-stages
```

### 4. Build RF3 JSON inputs from ReactZyme

```bash
source .venvs/foundry-rf3/bin/activate

python scripts/rf3/build_reactzyme_rf3_inputs.py \
  --source-root generate-constraints_0 \
  --output-root runs/rf3_reactzyme_inputs \
  --max-seq-len 600

# no-template mode: use multi-fragment SMILES directly and keep pocket constraints
python scripts/rf3/build_reactzyme_rf3_inputs.py \
  --source-root generate-constraints_0 \
  --output-root runs/rf3_reactzyme_inputs_smiles \
  --max-seq-len 600 \
  --ligand-source smiles

# larger split-table build from the full cleaned ReactZyme sequence/reaction tables
# this preserves pocket constraints, writes shard JSONs, and avoids emitting
# hundreds of thousands of per-example files
python scripts/rf3/build_reactzyme_rf3_inputs.py \
  --source-root generate-constraints_0 \
  --input-source reactzyme_split \
  --sequence-tsv generate-constraints_0/data/reactzyme_data_split/cleaned_uniprot_rhea.tsv \
  --rhea-molecules-tsv generate-constraints_0/data/reactzyme_data_split/rhea_molecules.tsv \
  --output-root runs/rf3_reactzyme_inputs_smiles_full \
  --max-seq-len 600 \
  --ligand-source smiles \
  --max-docked-pairs 2000 \
  --shards 256 \
  --no-example-files \
  --no-state-json
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

If you pass `--ligand-source smiles`, the builder omits `templates`,
`template_selection`, and `ground_truth_conformer_selection`, and instead emits
the ReactZyme multi-fragment SMILES directly as the ligand component while
keeping the same pocket constraints.

For RF3 output quality, the SMILES path now rejects chemically underspecified or
non-embeddable ligands rather than inventing replacement atoms. In practice,
dummy-atom (`*`) ligands and ligands that fail RDKit ETKDG embedding are
skipped before MSA prep / RF3 inference.

If you pass `--input-source reactzyme_split`, the builder explodes the full
`cleaned_uniprot_rhea.tsv` plus `rhea_molecules.tsv` tables instead of using the
smaller ETFlow train manifest. On the current local dataset, that path yields a
much larger pocket-constrained candidate set than the template-manifest subset.

The builder now defaults to `--max-docked-pairs 2000`. This cap is on accepted
reactant/product docking pairs after invalid dummy-atom / unparsable pairs are
filtered out and before state expansion. Use `--max-docked-pairs 0` to disable it.

### 5. Generate local MSAs and run Foundry RF3

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs_smiles_full \
  --prepared-root runs/rf3_reactzyme_inputs_smiles_full_with_msa \
  --out-root runs/rf3_reactzyme_out_smiles_full \
  --ckpt-path rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --msa-batch-size 64 \
  --msa-depth 2048 \
  --reuse-cache
```

When shard JSONs are present under `input-root/shards/<state>/`, the MSA prep
step and the RF3 runner process those shards directly rather than requiring a
single monolithic `<state>.json`.

`run_foundry_rf3_local_msa.sh` also defaults to `--max-docked-pairs 2000`, so
the same cap is enforced at MSA-prep/prediction time even if the input root
contains a much larger prebuilt set. The wrapper also accepts the legacy alias
`--max-examples` for the same cap. Use `--max-docked-pairs 0` to process the
entire input root.

The MSA-prep path now also defaults to `--msa-depth 2048`, which trims each
written `.a3m` to at most 2048 sequences including the query before RF3 reads
it. Use `--msa-depth 0` to disable trimming.

The RF3 local-MSA wrapper also now defaults to:

- `--msa-backend local_direct`
- `--msa-batch-size 64`
- `--msa-concurrency 8`
- `--use-filter`
- `--cuda-devices 0,1,2,3`
- `--rf3-gpus 4`
- `--rf3-launch-mode auto`

In `local_direct` mode, the MSA prep step bypasses the ColabFold ticket API and
runs local MMSeqs2 jobs directly against the shared UniRef30 database under
`../enzyme-quiver/MMseqs2/local_msa`. The wrapper binds up to four concurrent
MSA chunk workers across `CUDA_VISIBLE_DEVICES=0,1,2,3`, and now defaults to
eight total chunk workers so CPU-heavy MMSeqs post-processing can overlap while
GPU search workers continue to feed the four visible GPUs instead of leaving
them idle between stages.

The legacy server-backed route is still available if needed:

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs_smiles_full \
  --prepared-root runs/rf3_reactzyme_inputs_smiles_full_with_msa \
  --out-root runs/rf3_reactzyme_out_smiles_full \
  --ckpt-path rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --msa-backend server \
  --reuse-cache
```

For RF3 itself, the wrapper now defaults to `--rf3-launch-mode auto`. On a
multi-GPU host that resolves to `sharded_single`, which launches one
single-process RF3 shard per visible GPU instead of using PyTorch DDP/NCCL.
That is the preferred path on this host because it avoids the NCCL/TCPStore
watchdog failures seen with the upstream multi-rank Hydra launch while still
keeping all four GPUs busy. The wrapper still exports
`CUDA_VISIBLE_DEVICES=0,1,2,3` by default and fails fast if the requested GPU
count and visible device list do not match.

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
    --env-dir .venvs/foundry-rf3 \
    --input-root runs/rf3_reactzyme_inputs_smiles_full \
    --prepared-root runs/rf3_reactzyme_inputs_smiles_full_with_msa \
    --out-root runs/rf3_reactzyme_out_smiles_full \
    --ckpt-path rf3 \
    --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
    --reuse-cache \
    --max-examples 3000 \
    --msa-depth 2048 \
    --msa-concurrency 8 \
    --rf3-gpus 4 \
    --cuda-devices 0,1,2,3
```

If you explicitly need the upstream Hydra / DDP launcher, opt into it:

```bash
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs_smiles_full \
  --prepared-root runs/rf3_reactzyme_inputs_smiles_full_with_msa \
  --out-root runs/rf3_reactzyme_out_smiles_full \
  --ckpt-path rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --rf3-launch-mode ddp \
  --reuse-cache
```

`--hydra-override` is only supported in `ddp` mode. In the default
`sharded_single` mode, the wrapper calls the RF3 inference engine directly per
shard instead of routing through the Hydra CLI.

Startup validation progress is now reported with a `Validate Pairs` tqdm bar.
That stage cheaply rejects dummy-atom ligands up front and selects the first
valid docking pairs needed for the requested cap. The startup log now reports
`selected_pairs=<cap>` separately from `scanned_candidate_pairs=<window>` so the
requested cap is visibly enforced after filtering rather than looking like a
pre-filter cutoff.

MSA progress is now reported with outer `MSA Chunks` and `MSA Seqs` tqdm bars.
In `local_direct` mode, the `MSA Chunks` postfix shows which chunk is currently
bound to each active worker, and `MSA Seqs` now starts at the number of cached
unique sequences and reports cached / completed / active counts immediately, so
long-running chunk phases no longer look like a frozen run. In `server` mode,
the older inner Boltz/MMSeqs time-estimate bar is suppressed so retries and
chunk restarts no longer look like lost global progress.

During RF3 input loading, invalid SMILES-only examples that still fail atomworks
/ RDKit ligand construction are now skipped per example instead of aborting the
entire shard.

The wrapper first prepares local MSAs and attaches `msa_path`, then runs Foundry RF3 on both reactant and product JSON bundles.

For long runs on this host, `tmux` is the safest way to launch and monitor the
job:

```bash
tmux new -s rf3_3k

# inside tmux
source .venvs/foundry-rf3/bin/activate
bash scripts/rf3/run_foundry_rf3_local_msa.sh \
  --env-dir .venvs/foundry-rf3 \
  --input-root runs/rf3_reactzyme_inputs_smiles_full \
  --prepared-root runs/rf3_reactzyme_inputs_smiles_full_with_msa \
  --out-root runs/rf3_reactzyme_out_smiles_full \
  --ckpt-path rf3 \
  --local-msa-root ../enzyme-quiver/MMseqs2/local_msa \
  --reuse-cache \
  --max-examples 3000 \
  --msa-depth 2048 \
  --msa-concurrency 8 \
  --rf3-gpus 4 \
  --cuda-devices 0,1,2,3

# detach from tmux
# Ctrl-b then d

# reattach later
tmux attach -t rf3_3k
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



Observe It

  - Attach: tmux attach -t rf3_3k
  - Detach: Ctrl-b then d
  - Follow the log without attaching: tail -f /home/ubuntu/amelie/ThermoGFN/runs/rf3_reactzyme_out_smiles_full_sharded/
    tmux.log
  - Stop the run: tmux send-keys -t rf3_3k C-c
  - Kill the session: tmux kill-session -t rf3_3k