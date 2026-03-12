# ThermoGFN-IF

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Blog%20Post-ffcc4d)](https://huggingface.co/blog/AmelieSchreiber/thermogfn-if) [![bioRxiv](https://img.shields.io/badge/bioRxiv-Preprint-007a33.svg)](https://www.biorxiv.org/) [![Paper](https://img.shields.io/badge/PDF-Download%20Paper-blue)](./assets/paper/main.pdf)

ThermoGFN-IF implementation scaffold for multi-fidelity protein design with Method III-first training.

Important implementation note:

- the edit-trajectory GFlowNet formulation in the paper is the target methodology,
- the currently implemented `Method III` training loop in this repo is now a **real trajectory-balance GFlowNet teacher** over canonical edit trajectories, followed by one-shot student distillation for fast deployment.

The default catalytic path is nevertheless fully real in its oracle stack: packed-structure GraphKcat scoring with MC-dropout uncertainty, whole-enzyme UMA broad screening, forward and reverse sMD, optional PMF, and fused reward-based dataset updates.

## Required conda environments

Default catalytic RL path (`UMA-cat + GraphKcat`):

- `ligandmpnn_env`
- `mora-uma`
- `apodock`

Legacy / non-default stability-binding pipeline:

- `spurs`
- `bioemu`
- `uma-qc`

Optional legacy generator backend:

- `ADFLIP`

Minimal readiness checks for the default catalytic path:

```bash
conda run -n ligandmpnn_env python -c "import torch; print(torch.cuda.is_available())"
conda run -n mora-uma python -c "import fairchem"
conda run -n apodock python -c "import torch; print(torch.cuda.is_available())"
```

Full production-path readiness checks for the older stability/binding pipeline:

```bash
./scripts/env/check_envs.sh runs/env_status.json
# optional deep checks
RUN_HEALTH_CHECKS=1 ./scripts/env/check_envs.sh runs/env_status_health.json
```

Prefetch production oracle assets into a writable local cache before running the older stability/binding path:

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

Primary runtime configs:

- `config/m3_default.yaml` for the older stability/binding Method III path
- `config/uma_cat_m3_default.yaml` for the default catalytic Method III path

`config/m3_default.yaml` includes:

- generator backend and LigandMPNN generation controls (`batch_size`, `number_of_batches`, `temperature`, atom-context flags),
- Method III round controls (`pool_size`, `bioemu_budget`, `uma_budget`, checkpoint retention),
- teacher/student/surrogate knobs (`surrogate_ensemble_size`, `teacher_steps`, `teacher_gamma_off`, `student_steps`),
- oracle settings (`spurs repo/chain`, `bioemu model+num_samples`, `uma model+workers+replicates`),
- BioEmu VRAM-aware batching (`oracles.bioemu.batch_size_100`, `oracles.bioemu.auto_batch_from_vram`, `oracles.bioemu.target_vram_frac`, min/max bounds),
- periodic test sizing and final inference selection (`periodic_eval.num_candidates`, `inference.final.num_candidates`, `inference.final.top_k`).

`config/uma_cat_m3_default.yaml` is the default self-contained catalytic RL config and includes:

- generator backend and LigandMPNN packing controls,
- UMA catalytic broad-screen controls,
- forward / reverse sMD controls,
- optional PMF controls,
- GraphKcat refinement controls,
- catalytic fusion weights,
- Method III round budgets for `uma_cat_budget` and `graphkcat_budget`,
- W&B defaults under `logging.wandb` for round-level and experiment-level telemetry.

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

## Default catalytic RL path: whole-enzyme UMA + sMD/PMF + GraphKcat

This is now the primary catalytic Method III path in the repo. It is self-contained and does not depend on the sibling `enzyme-quiver` repository at runtime.

Core idea:

- build catalytic candidates from RF3 reactant-bound and product-bound outputs,
- repack the full student pool with LigandMPNN so GraphKcat sees mutant-specific structures rather than seed structures,
- run GraphKcat first across that packed mutant pool,
- keep the top GraphKcat tranche as a catalytic prefilter,
- run a broad whole-enzyme UMA equilibrium screen from the reactant-bound basin,
- optionally run forward and reverse steered UMA dynamics between reactant-bound and product-bound states,
- optionally reconstruct a path umbrella PMF from the sMD-derived path,
- use GraphKcat MC-dropout as the first-pass structural catalytic oracle,
- fuse UMA-cat and GraphKcat into the Method III reward.

The implemented catalytic scalar is a transition-state-style log-rate proxy:

```text
log10 k_proxy = log10(k_B T / h) - (Delta G_gate + Delta G_barrier) / (RT ln 10)
```

where:

- `Delta G_gate` comes from productive-pose / gNAC occupancy in the broad whole-enzyme UMA screen,
- `Delta G_barrier` comes from the PMF if enabled, otherwise from the sMD Jarzynski-style barrier estimate.

The default catalytic uncertainty propagated into reward fusion is also physically structured:

```text
sigma(log10 k_proxy) = sqrt(sigma(Delta G_gate)^2 + sigma(Delta G_barrier)^2) / (RT ln 10)
```

where:

- `sigma(Delta G_gate)` comes from productive-pose occupancy uncertainty,
- `sigma(Delta G_barrier)` comes from PMF barrier uncertainty when PMF is enabled,
- otherwise `sigma(Delta G_barrier)` comes from the sMD barrier uncertainty plus a forward/reverse hysteresis term.

The default config is:

- `config/uma_cat_m3_default.yaml`

Default active oracle env bindings:

```yaml
oracles:
  envs:
    packer: ligandmpnn_env
    uma_cat: mora-uma
    graphkcat: apodock
```

Default catalytic routing in that config:

- `generator.backend = ligandmpnn`
- `round.graphkcat_prefilter_fraction = 0.5`
- `round.graphkcat_prefilter_risk_kappa = 0.5`
- `oracles.uma_cat.smd.enabled = true`
- `oracles.uma_cat.smd.reverse = true`
- `oracles.uma_cat.pmf.enabled = false`
- GraphKcat enabled
- SPURS, BioEmu, KcatNet, MMKcat, and thermostability UMA are not part of this default RL loop

That means the default round order is:

1. student proposes the pool
2. LigandMPNN packs the full pool onto the endpoint complexes
3. GraphKcat scores the packed mutant pool with MC-dropout
3. top `50%` by risk-adjusted `graphkcat_log_kcat` survive the prefilter
4. UMA-cat acquisition selects the expensive subset from within that screened set
5. UMA-cat runs broad dynamics plus optional sMD / PMF
6. fused reward is computed from UMA-cat and GraphKcat

The implemented GraphKcat prefilter statistic is the risk-adjusted score

```text
graphkcat_prefilter_score = graphkcat_log_kcat - kappa * graphkcat_std
```

with `graphkcat_std` coming from MC-dropout by default.

Important environment note:

- `apodock` needs `huggingface_hub` available so Uni-Mol can fetch its weights on first use.

### Tested RF3-to-catalytic dataset build

This bridge uses prepared RF3 inputs plus the finished reactant/product RF3 outputs:

```bash
conda run -n mora-uma python scripts/rf3/build_uma_cat_dataset.py \
  --prepared-input-root runs/rf3_reactzyme_inputs_smiles_full_with_msa_v7 \
  --reactant-root runs/rf3_reactzyme_out_smiles_full_sharded_v9/reactant \
  --product-root runs/rf3_reactzyme_out_smiles_full_sharded_v9/product \
  --output-path runs/tmp/uma_cat_smoke_dataset.jsonl \
  --run-id uma_cat_smoke \
  --split train \
  --round-id 0 \
  --limit 2
```

Each row carries:

- `sequence`
- `substrate_smiles`
- `product_smiles`
- `reactant_complex_path`
- `product_complex_path`
- `protein_chain_id`
- `ligand_chain_id`
- `pocket_positions`

### Pack reactant and product endpoints with LigandMPNN

```bash
head -n 1 runs/tmp/uma_cat_smoke_dataset.jsonl > runs/tmp/uma_cat_smoke_dataset_1.jsonl

conda run -n ligandmpnn_env python scripts/prep/oracles/ligandmpnn_pack_candidates.py \
  --candidate-path runs/tmp/uma_cat_smoke_dataset_1.jsonl \
  --output-path runs/tmp/uma_cat_smoke_packed.jsonl \
  --output-root runs/tmp/uma_cat_smoke_packed_structures \
  --ligandmpnn-root models/LigandMPNN \
  --checkpoint-sc models/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt \
  --device cuda:0 \
  --pack-with-ligand-context 1 \
  --repack-everything 1 \
  --sc-num-denoising-steps 1 \
  --sc-num-samples 1
```

This writes:

- `reactant_complex_packed_path`
- `reactant_protein_packed_path`
- `product_complex_packed_path`
- `product_protein_packed_path`

### Run whole-enzyme UMA catalytic scoring

Broad screen + forward/reverse sMD + PMF smoke command:

```bash
conda run -n mora-uma python scripts/prep/oracles/uma_catalytic_score.py \
  --candidate-path runs/tmp/uma_cat_smoke_packed.jsonl \
  --output-path runs/tmp/uma_cat_smoke_scored.jsonl \
  --artifact-root runs/tmp/uma_cat_smoke_artifacts \
  --model-name uma-s-1p1 \
  --device cuda:0 \
  --calculator-workers 1 \
  --temperature-k 300 \
  --broad-steps 5 \
  --broad-replicas 1 \
  --broad-save-every 5 \
  --run-smd 1 \
  --run-reverse-smd 1 \
  --smd-images 3 \
  --smd-steps-per-image 1 \
  --smd-replicas 1 \
  --run-pmf 1 \
  --pmf-windows 3 \
  --pmf-steps-per-window 2 \
  --pmf-save-every 1 \
  --pmf-replicas 1
```

What this stage computes:

- broad productive-pose occupancy `uma_cat_p_gnac`
- gating free energy `uma_cat_delta_g_gate_kcal_mol`
- forward and reverse work statistics
- sMD barrier `uma_cat_delta_g_smd_barrier_kcal_mol`
- optional PMF barrier `uma_cat_delta_g_pmf_barrier_kcal_mol`
- forward/reverse mismatch `uma_cat_forward_reverse_gap_kcal_mol`
- near-TS candidate count `uma_cat_near_ts_count`
- final catalytic scalar `uma_cat_log10_rate_proxy`

Per-candidate artifacts are written under `--artifact-root`, including:

- `summary.json`
- `broad_rows.jsonl`
- `broad_replicates.json`
- `smd_work_profile.jsonl`
- `smd_summary.json`
- `smd_reverse_summary.json`
- `smd_near_ts.json`
- `pmf_summary.json`

### Run GraphKcat on the same packed candidates

```bash
conda run -n apodock python scripts/prep/oracles/graphkcat_score.py \
  --candidate-path runs/tmp/uma_cat_smoke_packed.jsonl \
  --output-path runs/tmp/uma_cat_smoke_graph.jsonl \
  --model-root models/GraphKcat \
  --checkpoint models/GraphKcat/checkpoint/paper.pt \
  --cfg TrainConfig_kcat_enz \
  --batch-size 1 \
  --device cuda:0 \
  --distance-cutoff-a 8.0 \
  --std-default 0.25 \
  --mc-dropout-samples 8 \
  --mc-dropout-seed 13 \
  --work-dir runs/tmp/uma_cat_smoke_graph_work
```

By default this wrapper now:

- materializes the **packed mutant protein structure** for GraphKcat scoring,
- runs MC-dropout predictive inference,
- writes:
  - `graphkcat_log_kcat`
  - `graphkcat_log_km`
  - `graphkcat_log_kcat_km`
  - `graphkcat_std`
  - `graphkcat_log_km_std`
  - `graphkcat_log_kcat_km_std`

Compatibility fallback note:

- if MC-dropout is explicitly disabled or the model does not emit uncertainty columns, the wrapper can still fall back to `--std-default`,
- but that fallback is **not** the default methodology and is not used in the default catalytic config.

### Fuse UMA-cat and GraphKcat into the RL reward

```bash
python scripts/prep/oracles/fuse_catalytic_scores.py \
  --candidate-path runs/tmp/uma_cat_smoke_scored.jsonl \
  --graphkcat-path runs/tmp/uma_cat_smoke_graph.jsonl \
  --output-path runs/tmp/uma_cat_smoke_fused.jsonl
```

The fused rows include:

- `rho_UCAT`
- `rho_G`
- `z_UCAT`
- `z_GK`
- `z_agree`
- `score`
- `reward`

### Run a full UMA-cat Method III round

```bash
python scripts/orchestration/uma_cat_m3_run_round.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_demo \
  --round-id 0 \
  --dataset-path runs/bootstrap/uma_cat_D_0_train.jsonl \
  --output-dir runs/uma_cat_demo/round_000 \
  --pool-size 50000 \
  --uma-cat-budget 256 \
  --graphkcat-prefilter-fraction 0.5
```

Dry-run version:

```bash
python scripts/orchestration/uma_cat_m3_run_round.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_smoke_round \
  --round-id 0 \
  --dataset-path runs/tmp/uma_cat_smoke_dataset_1.jsonl \
  --output-dir runs/tmp/uma_cat_round_dry \
  --pool-size 4 \
  --uma-cat-budget 1 \
  --graphkcat-prefilter-fraction 0.5 \
  --teacher-steps 1 \
  --student-steps 1 \
  --dry-run \
  --no-progress
```

### Run a multi-round UMA-cat Method III experiment

```bash
python scripts/orchestration/uma_cat_m3_run_experiment.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_demo \
  --dataset-path runs/bootstrap/uma_cat_D_0_train.jsonl \
  --output-root runs/uma_cat_demo \
  --num-rounds 8
```

Dry-run version:

```bash
python scripts/orchestration/uma_cat_m3_run_experiment.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_smoke_exp \
  --dataset-path runs/tmp/uma_cat_smoke_dataset_1.jsonl \
  --output-root runs/tmp/uma_cat_experiment_dry \
  --num-rounds 1 \
  --pool-size 4 \
  --uma-cat-budget 1 \
  --graphkcat-budget 1 \
  --teacher-steps 1 \
  --student-steps 1 \
  --dry-run \
  --no-progress
```

### W\&B, progress bars, and structured logs

The default catalytic Method III config enables W\&B in `auto` mode:

```yaml
logging:
  wandb:
    enabled: true
    mode: auto
```

That means the same command will sync online when credentials are available and fall back to a local offline run under `./wandb/` otherwise.

To switch to live W\&B logging, export your credentials and override the mode:

```bash
export WANDB_API_KEY=...

python scripts/orchestration/uma_cat_m3_run_experiment.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_demo_online \
  --dataset-path runs/bootstrap/uma_cat_D_0_train.jsonl \
  --output-root runs/uma_cat_demo_online \
  --num-rounds 2 \
  --wandb-mode online \
  --wandb-project thermogfn \
  --wandb-group uma_cat_demo
```

The current implementation logs all of the following into W\&B and to structured files on disk:

- surrogate bootstrap metrics: bootstrap index, bootstrap size, target mean/std, coefficient norm;
- trajectory-balance teacher metrics: `loss`, `off_loss`, `on_loss`, `reg_loss`, `delta_abs`, `lr`, `grad_norm_pre_clip`, `grad_norm_post_clip`, `mean_stop_prob`, `mean_log_z`;
- student-distillation metrics: sampled-fraction, sampled mutation-count mean, mutation-count entropy, distinct sampled `K`, distinct seed families;
- generated-pool summary metrics: pool size, mutation-order summary, sequence uniqueness;
- GraphKcat pool and selected-set summaries, including success fraction and mean predicted `log_kcat`;
- UMA-cat summary metrics, including success fraction and mean catalytic `log10_rate_proxy`;
- round-level summaries, teacher-student evaluation summaries, append summaries, and gate reports.

For a catalytic round, the main metric files are:

- `runs/<run_id>/round_<id>/metrics/surrogate_history_round_<id>.jsonl`
- `runs/<run_id>/round_<id>/metrics/teacher_history_round_<id>.jsonl`
- `runs/<run_id>/round_<id>/metrics/student_history_round_<id>.jsonl`
- `runs/<run_id>/round_<id>/metrics/student_pool_metrics_round_<id>.json`
- `runs/<run_id>/round_<id>/metrics/graphkcat_pool_summary_round_<id>.json`
- `runs/<run_id>/round_<id>/metrics/graphkcat_selected_summary_round_<id>.json`
- `runs/<run_id>/round_<id>/metrics/uma_cat_summary_round_<id>.json`
- `runs/<run_id>/round_<id>/metrics/round_metrics.json`
- `runs/<run_id>/round_<id>/metrics/teacher_student_eval.json`
- `runs/<run_id>/round_<id>/manifests/append_summary.json`
- `runs/<run_id>/round_<id>/manifests/round_gate_report.json`

Progress reporting is also layered intentionally:

- experiment runner: one tqdm bar over rounds;
- round runner: one tqdm bar over orchestration steps;
- trainer steps: detailed per-step logging for the surrogate, TB teacher, and student distillation;
- oracle stages: their own tqdm or staged logging where available;
- every long-running subprocess still emits heartbeat-style log lines via `--step-heartbeat-sec`.

Use `--no-progress` only when you need log-only operation, for example in CI or when redirecting output to a file.

### Practical notes

- The implemented `Method III` controller is now a real trajectory-balance GFlowNet teacher over the canonical edit DAG, followed by one-shot student distillation.
- The teacher uses explicit `STOP -> position -> amino-acid` factorization on canonical edit trajectories reconstructed from labeled candidates, with per-seed `log Z` and a TB loss on terminal reward.
- The deployed student is still one shot. It is distilled from teacher samples into `K`, position, and residue-replacement marginals so deployment remains fast while teacher training remains truly reward-proportional.
- The default catalytic path contains no mock scoring branch: LigandMPNN packing, GraphKcat inference, UMA broad screening, sMD, and optional PMF are all real runtime stages.
- `oracles.uma_cat.smd.enabled` and `oracles.uma_cat.smd.reverse` control forward/reverse steering.
- `oracles.uma_cat.pmf.enabled` toggles the PMF stage. It is off by default because it is materially more expensive.
- The broad screen, sMD, and PMF are all real FAIRChem/ASE runs through `mora-uma`; this path does not use static proxy replacements.
- GraphKcat is the default auxiliary structural catalytic oracle in this path, and it is now evaluated on packed mutant structures before UMA-cat selection.
- The default telemetry path is also real: the round/experiment runners ingest child histories and oracle summaries back into W\&B rather than emitting only parent-process timestamps.

### Tested end-to-end real rounds

The following real rounds completed successfully after the current fixes.

Bootstrap catalytic round:

```bash
python scripts/orchestration/uma_cat_m3_run_round.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_round_real_v3 \
  --round-id 0 \
  --dataset-path runs/tmp/uma_cat_smoke_dataset_1.jsonl \
  --output-dir runs/tmp/uma_cat_round_real_v3 \
  --pool-size 2 \
  --uma-cat-budget 1 \
  --graphkcat-budget 1 \
  --graphkcat-prefilter-fraction 0.5 \
  --graphkcat-mc-dropout-samples 2 \
  --teacher-steps 1 \
  --student-steps 1 \
  --uma-broad-steps 1 \
  --uma-broad-replicas 1 \
  --uma-broad-save-every 1 \
  --uma-run-smd 1 \
  --uma-run-reverse-smd 1 \
  --uma-smd-images 2 \
  --uma-smd-steps-per-image 1 \
  --uma-smd-replicas 1 \
  --uma-run-pmf 0 \
  --step-heartbeat-sec 20 \
  --no-progress
```

Strict labeled round with actual TB teacher training on oracle-derived reward:

```bash
python scripts/orchestration/uma_cat_m3_run_round.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_tb_round_v2 \
  --round-id 1 \
  --dataset-path runs/tmp/uma_cat_round_real_v3/data/D_1.jsonl \
  --output-dir runs/tmp/uma_cat_tb_round_v2 \
  --pool-size 2 \
  --uma-cat-budget 1 \
  --graphkcat-budget 1 \
  --graphkcat-prefilter-fraction 0.5 \
  --graphkcat-mc-dropout-samples 2 \
  --teacher-steps 64 \
  --student-steps 128 \
  --uma-broad-steps 1 \
  --uma-broad-replicas 1 \
  --uma-broad-save-every 1 \
  --uma-run-smd 1 \
  --uma-run-reverse-smd 1 \
  --uma-smd-images 2 \
  --uma-smd-steps-per-image 1 \
  --uma-smd-replicas 1 \
  --uma-run-pmf 0 \
  --step-heartbeat-sec 20 \
  --no-progress
```

The strict labeled round passed all round gates, including teacher-student evaluation:

- `teacher_mode = trajectory_balance_gflownet`
- `is_true_gflownet = true`
- `teacher_student_kl = 0.0`

The full catalytic round completed:

- surrogate fit
- trajectory-balance teacher fit
- student distillation
- packed-pool GraphKcat scoring with MC-dropout
- GraphKcat top-50% prefilter
- UMA-cat broad screen
- forward and reverse sMD
- catalytic fusion
- dataset append
- design metrics
- teacher-student evaluation

## Legacy Kcat-only disjoint stage (KcatNet + GraphKcat)

This remains available as an auxiliary sequence-first catalytic loop, but it is not the default catalytic RL method anymore.

Kcat mode is wired as a separate Method III loop and does not call the self-contained whole-enzyme UMA catalytic stack.

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
