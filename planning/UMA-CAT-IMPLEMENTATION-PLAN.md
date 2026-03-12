# Self-Contained UMA-Cat + GraphKcat Method III Plan

## Objective

Implement and document a real catalytic GFlowNet-style RL tuning path for LigandMPNN that is fully self-contained inside this repository and uses:

- LigandMPNN as the generator / packer,
- whole-enzyme UMA dynamics as the primary catalytic oracle,
- optional forward and reverse steered UMA dynamics between reactant-bound and product-bound states,
- optional sMD-seeded path umbrella PMFs,
- GraphKcat as the structural catalytic refinement oracle,
- YAML-controlled Method III orchestration.

This is now the default catalytic RL path. The older KcatNet / MMKcat stage remains a separate auxiliary path, not the default training route.

## Default Training Policy

The default catalytic RL configuration is:

- config: `config/uma_cat_m3_default.yaml`
- generator backend: `ligandmpnn`
- active oracles during RL:
  - `uma_cat`
  - `graphkcat`
- inactive by default in this path:
  - `spurs`
  - `bioemu`
  - `uma-qc` thermostability branch
  - `KcatNet`
  - `MMKcat`

The implemented round loop therefore learns from:

1. whole-enzyme catalytic preorganization under UMA,
2. nonequilibrium reactant-to-product steering under UMA,
3. optional PMF refinement under UMA,
4. GraphKcat turnover feedback.

## Implemented Mathematical Model

### 1. Broad catalytic preorganization screen

For a reactant-bound complex with equilibrium density `rho_A(R)`, define productive-pose indicators from actual whole-enzyme coordinates:

`q(R) = (r_lig(R), c(R), d_min(R), r_pocket(R))`

where:

- `r_lig` is ligand RMSD after pocket alignment,
- `c` is pocket-ligand heavy-atom contact count,
- `d_min` is the minimum pocket-ligand heavy-atom distance,
- `r_pocket` is local pocket RMSD.

The implemented hard productive indicator is:

`I_gNAC(R) = 1[r_lig <= r_max and c >= c_min and d_min >= d_safe and r_pocket <= p_max]`

The productive-pose population is estimated from replicated UMA Langevin trajectories:

`p_gNAC = mean_t,k I_gNAC(R_t^(k))`

The gating free energy is:

`Delta G_gate = -RT log(p_gNAC / (1 - p_gNAC))`

The runtime also computes:

- soft productive score `p_soft`,
- productive visit count,
- mean productive dwell,
- first-hit frame,
- effective sample size,
- lower confidence bound on `p_gNAC`,
- instability penalty from clash-like undersafe pocket-ligand contacts.

### 2. Forward / reverse steered whole-enzyme UMA dynamics

Given reactant-bound basin `A` and product-bound basin `B`, the runtime builds a ligand-atom map between endpoints and applies:

- strong harmonic steering to mapped ligand atoms,
- weak harmonic anchoring to non-pocket protein heavy atoms.

The time-dependent biased potential is:

`U_tot(R, t) = U_UMA(R) + U_steer(R, lambda_t) + U_anchor(R)`

with:

`U_steer(R, lambda_t) = sum_i (k_steer / 2) ||r_i - r_i*(lambda_t)||^2`

`U_anchor(R) = sum_j (k_anchor / 2) ||r_j - r_j^(0)||^2`

The implementation records per-step cumulative work and builds:

- forward work traces,
- optional reverse work traces,
- Jarzynski free-energy profile along the pulling schedule,
- final mean work and work standard deviation,
- forward/reverse work-gap diagnostic,
- averaged path images,
- near-TS-like candidate frames ranked from local work, midpoint symmetry, and endpoint distance balance.

### 3. Optional path umbrella PMF

If PMF is enabled, the code constructs umbrella windows along the sMD-derived ligand path. Each window uses a fixed target on the mapped ligand coordinates plus the same anchor restraint on the enzyme scaffold:

`U_j(R) = U_UMA(R) + (k_window / 2) ||x_map(R) - x_j*||^2 + U_anchor(R)`

The runtime then:

- runs biased Langevin dynamics in each window,
- records reduced bias energies across all windows,
- solves MBAR-like free-energy offsets,
- reconstructs a PMF over path progress,
- estimates:
  - `Delta G_pmf_barrier`,
  - `Delta G_pmf_react_to_prod`.

When PMF is enabled and successful, the catalytic barrier source is `pmf`; otherwise it is `smd`.

### 4. Catalytic rate proxy

The implemented catalytic scalar is a transition-state-style log-rate proxy:

`log10 k_proxy = log10(k_B T / h) - (Delta G_gate + Delta G_barrier) / (RT ln 10)`

where:

- `Delta G_barrier = Delta G_pmf_barrier` when PMF succeeds,
- otherwise `Delta G_barrier = Delta G_smd_barrier`.

### 5. GraphKcat fusion

The implemented fused Method III reward uses:

- UMA-cat log-rate proxy,
- GraphKcat turnover channel (`graphkcat_log_kcat`),
- an agreement penalty.

The current implementation in `train/thermogfn/uma_cat_reward.py` computes:

`score = w_UCAT rho_UCAT z_UCAT + w_G rho_G z_GK + w_agree rho_UCAT rho_G z_agree`

with:

- `z_UCAT = mean_UMA - kappa_UMA * std_UMA`
- `z_GK = mean_GK - kappa_GK * std_GK`
- `z_agree = -|UMA_log10_rate - GraphKcat_log_kcat|`

If UMA-cat is missing, the score is clamped to a strong negative value rather than silently becoming neutral.

Reward is then:

`reward = 1e-6 + exp(clip(score, -8, 8))`

## Repository Implementation

### Core runtime

- `train/thermogfn/uma_cat_runtime.py`
  - whole-enzyme structure loading
  - pocket / ligand / anchor atom selection
  - broad productive-pose screening
  - self-contained ligand endpoint mapping
  - forward / reverse sMD
  - near-TS harvesting
  - path-image extraction
  - umbrella PMF reconstruction
  - artifact writing helpers

- `train/thermogfn/uma_cat_reward.py`
  - gating free energy
  - TST-style log-rate proxy
  - risk adjustment
  - UMA-cat + GraphKcat reward fusion

### Dataset bridge

- `scripts/rf3/build_uma_cat_dataset.py`
  - builds catalytic Method III datasets from RF3 reactant/product outputs
  - requires both reactant and product RF3 complexes
  - materializes:
    - `candidate_id`
    - `sequence`
    - `substrate_smiles`
    - `product_smiles`
    - `reactant_complex_path`
    - `product_complex_path`
    - `protein_chain_id`
    - `ligand_chain_id`
    - `pocket_positions`

### Packer

- `scripts/prep/oracles/ligandmpnn_pack_candidates.py`
  - packs each candidate sequence onto both reactant and product complexes
  - writes:
    - `reactant_complex_packed_path`
    - `reactant_protein_packed_path`
    - `product_complex_packed_path`
    - `product_protein_packed_path`
  - fixed to set `chain_mask` correctly for LigandMPNN sidechain packing

### Oracle stage

- `scripts/prep/oracles/uma_catalytic_score.py`
  - broad whole-enzyme UMA screen
  - optional forward / reverse sMD
  - optional PMF
  - writes structured artifact bundles per candidate

- `scripts/prep/oracles/graphkcat_score.py`
  - structure-aware catalytic refinement
  - fixed in-repo for multi-fragment ligands and empty contact-graph edge cases

- `scripts/prep/oracles/fuse_catalytic_scores.py`
  - fuses UMA-cat and GraphKcat into the final Method III reward fields

### Method III orchestration

- `scripts/train/m3_select_uma_cat_batch.py`
- `scripts/orchestration/uma_cat_m3_run_round.py`
- `scripts/orchestration/uma_cat_m3_run_experiment.py`

The round runner now:

1. validates catalytic metadata,
2. trains / refreshes surrogate, teacher, and student,
3. samples a pool,
4. selects a UMA-cat batch,
5. packs candidates with LigandMPNN,
6. runs whole-enzyme UMA broad stage,
7. runs optional forward / reverse sMD,
8. runs optional PMF,
9. selects a GraphKcat subset,
10. fuses rewards,
11. appends the next round dataset.

## YAML Contract

The default config is:

- `config/uma_cat_m3_default.yaml`

Key toggles:

```yaml
generator:
  backend: ligandmpnn

oracles:
  envs:
    packer: ligandmpnn_env
    uma_cat: mora-uma
    graphkcat: apodock
  uma_cat:
    broad:
      steps: 500
      replicas: 2
    smd:
      enabled: true
      reverse: true
      images: 24
      steps_per_image: 25
      replicas: 2
    pmf:
      enabled: false
      windows: 16
      steps_per_window: 100
      replicas: 1
  graphkcat:
    batch_size: 2
```

This means:

- broad whole-enzyme UMA screen is on,
- forward and reverse sMD are on,
- PMF is available but off by default,
- GraphKcat is on,
- the default RL loop is already the new UMA-cat + GraphKcat method.

## Actual Smoke-Tested Path

The following path was exercised on real RF3 outputs from:

- `runs/rf3_reactzyme_inputs_smiles_full_with_msa_v7`
- `runs/rf3_reactzyme_out_smiles_full_sharded_v9/reactant`
- `runs/rf3_reactzyme_out_smiles_full_sharded_v9/product`

### 1. Build catalytic dataset from RF3 outputs

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

### 2. Pack one candidate with LigandMPNN

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

### 3. Run whole-enzyme UMA broad screen + forward/reverse sMD + PMF

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

Observed output fields included:

- `uma_cat_status = ok`
- `uma_cat_delta_g_gate_kcal_mol`
- `uma_cat_delta_g_smd_barrier_kcal_mol`
- `uma_cat_delta_g_pmf_barrier_kcal_mol`
- `uma_cat_barrier_source = pmf`
- `uma_cat_forward_reverse_gap_kcal_mol`
- `uma_cat_near_ts_count`
- `uma_cat_log10_rate_proxy`

### 4. Run GraphKcat on the same packed candidate

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
  --work-dir runs/tmp/uma_cat_smoke_graph_work
```

### 5. Fuse UMA-cat + GraphKcat

```bash
python scripts/prep/oracles/fuse_catalytic_scores.py \
  --candidate-path runs/tmp/uma_cat_smoke_scored.jsonl \
  --graphkcat-path runs/tmp/uma_cat_smoke_graph.jsonl \
  --output-path runs/tmp/uma_cat_smoke_fused.jsonl
```

Observed fused fields included:

- `rho_UCAT = 1.0`
- `rho_G = 1.0`
- `z_UCAT`
- `z_GK`
- `z_agree`
- `score`
- `reward`

### 6. Dry-run the Method III round / experiment orchestration

```bash
python scripts/orchestration/uma_cat_m3_run_round.py \
  --config config/uma_cat_m3_default.yaml \
  --run-id uma_cat_smoke_round \
  --round-id 0 \
  --dataset-path runs/tmp/uma_cat_smoke_dataset_1.jsonl \
  --output-dir runs/tmp/uma_cat_round_dry \
  --pool-size 4 \
  --uma-cat-budget 1 \
  --graphkcat-budget 1 \
  --teacher-steps 1 \
  --student-steps 1 \
  --dry-run \
  --no-progress
```

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

## Required Runtime Environments

For the default catalytic path:

- `ligandmpnn_env`
- `mora-uma`
- `apodock`

Important note:

- `apodock` needed `huggingface_hub` installed so Uni-Mol could fetch weights.

## Acceptance Criteria

The self-contained UMA-cat implementation is considered operational when:

1. `build_uma_cat_dataset.py` builds valid candidate records from RF3 outputs.
2. `ligandmpnn_pack_candidates.py` writes packed reactant and product complexes.
3. `uma_catalytic_score.py` produces:
   - broad-screen artifacts,
   - forward/reverse sMD summaries,
   - near-TS candidates,
   - optional PMF summaries.
4. `graphkcat_score.py` runs on the packed candidates.
5. `fuse_catalytic_scores.py` produces positive rewards.
6. `uma_cat_m3_run_round.py` and `uma_cat_m3_run_experiment.py` dry-run successfully.

These criteria have now been met on RF3-derived smoke examples.

## Remaining Hardening Work

The core methodology is implemented and tested, but the following are still worth tightening:

- add richer near-TS ranking diagnostics beyond the current work/symmetry heuristic,
- add explicit forward/reverse Crooks-style diagnostics to the saved summaries,
- add longer PMF convergence tests on a small real candidate panel,
- benchmark `broad.steps`, `smd.images`, and `pmf.windows` on a fixed catalyst subset,
- add explicit train-time evaluation plots for `uma_cat_log10_rate_proxy` versus GraphKcat.

These are hardening tasks, not blockers for the default self-contained UMA-cat + GraphKcat Method III workflow.
