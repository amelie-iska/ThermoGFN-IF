# ThermoGFN-IF Implementation Plan (Codebase-Aligned, Method III First)

## 0) Objective and Scope

This plan updates and strengthens the previous implementation plan using the **actual repository structure** and model code currently in `./`.

Primary implementation priority:
- Build **Method III (OneShot-AL)** first (round-based teacher-student active learning).

Then implement:
- Method I (`Simul-MF`) and Method II (`Seq-Curr`) on the same shared stack.

Hard constraints from request:
- **All preparation, training, and inference entry scripts live under `./scripts`.**
- When behavior/details are unclear, resolve against `./planning/ThermoGFN-IF.tex`.
- Use existing monomer split now; add mixed-modality splits later as they are generated.

---

## 1) Codebase Review Summary

## 1.1 Current top-level structure

Observed in repo root:
- `README.md`
- `config/`
- `data/`
- `models/`
- `planning/`
- `scripts/`
- `train/` (currently empty)
- `weights/`

Existing root scripts:
- `scripts/run_rfd3_inference.sh`
- `scripts/protrek_cluster_split.py`
- `scripts/run_protrek_split_unconditional_monomer.sh`
- `scripts/build_ccd_ligand_library.py`
- `scripts/decompress_rfd3_cifs.sh`

Existing split already present:
- `data/rfd3_splits/unconditional_monomer_protrek35m/{train,test,metadata}`

This split should be treated as the canonical initial dataset for Method III bootstrapping.

## 1.2 Model READMEs reviewed and inference entrypoints

Reviewed top-level model READMEs:
- `models/ADFLIP/README.md`
- `models/SPURS/README.md`
- `models/bioemu/README.md`
- `models/fairchem/README.md`
- `models/gflownets/README.md`
- `models/fampnn/README.md`
- `models/foundry/README.md`
- `models/ProTrek/README.md`

Reviewed concrete inference code paths:

### ADFLIP
- Example inference: `models/ADFLIP/test/design.py`
- Core sampler methods: `models/ADFLIP/model/discrete_flow_aa.py` (`sample`, `adaptive_sample`)
- Sidechain packing dependency through PIPPack:
  - `models/ADFLIP/PIPPack/inference.py`

### SPURS
- Inference API: `models/SPURS/spurs/inference.py`
  - `get_SPURS_from_hub`, `get_SPURS_multi_from_hub`
  - `parse_pdb`, `parse_pdb_for_mutation`

### BioEmu
- Main sampler entrypoint: `models/bioemu/src/bioemu/sample.py`
- Sidechain/relax utility: `models/bioemu/src/bioemu/sidechain_relax.py`
- Supports steering configuration and denoiser configs.

### UMA / FAIRChem
- Pretrained model loader: `models/fairchem/src/fairchem/core/calculate/pretrained_mlip.py`
- ASE calculator wrapper: `models/fairchem/src/fairchem/core/calculate/ase_calculator.py`
- MD usage via ASE/Langevin (task-specific heads incl. `omol`).

### GFlowNet baseline repo
- Active-learning examples:
  - `models/gflownets/mols/gflownet_activelearning.py`
  - `models/gflownets/grid/toy_grid_dag_al.py`

### Foundry (RFD3/RF3/MPNN)
- RFD3 inference entry:
  - `models/foundry/models/rfd3/src/rfd3/run_inference.py`
  - `models/foundry/models/rfd3/src/rfd3/cli.py`
- RF3 inference:
  - `models/foundry/models/rf3/src/rf3/inference.py`
- MPNN inference:
  - `models/foundry/models/mpnn/src/mpnn/inference.py`

### ProTrek
- Local server orchestration: `models/ProTrek/demo/run_pipeline.py`
- Database building for retrieval index: `models/ProTrek/scripts/generate_database.py`
- Model/init/search utilities:
  - `models/ProTrek/demo/modules/init_model.py`
  - `models/ProTrek/demo/modules/search.py`

## 1.3 Integration-critical findings

1. Environment versions differ materially across model repos:
- SPURS readme targets Python 3.7 + Torch 1.12.
- ADFLIP targets Python 3.10 (+ specific torch-cluster/scatter builds).
- BioEmu targets Python 3.10+.
- FAIRChem targets Python 3.10+.

2. Therefore: do **not** force a single runtime env initially.
- Use per-model env execution wrappers (subprocess-level orchestration) and strict I/O contracts.
- Canonical required env names for implementation:
  - `ADFLIP` (generator/seed + structural refresh flows),
  - `spurs` (cheap oracle single/multi scoring),
  - `bioemu` (ensemble dynamics oracle),
  - `uma-qc` (FAIRChem/UMA MD oracle).

3. Existing repo already includes data-gen/split tooling using Foundry+ProTrek.
- Plan should leverage this and avoid redundant generators.

---

## 2) Implementation Principles (Enforced)

1. Script-first execution:
- Every operational entrypoint is in `./scripts`.
- Scripts can import internal Python modules under `train/` (or `scripts/lib`) for reuse.

2. Reproducibility:
- Every script must accept explicit `--seed`, `--config`, and `--run-id`.
- Every run writes manifest + resolved config + version info.

3. Idempotence:
- Scripts should be restart-safe (`--resume` / skip existing outputs).

4. Model isolation:
- Use environment-specific launcher scripts per model family.
- Exchange artifacts via files (`jsonl/parquet/npz/pdb/cif`) rather than in-process imports across incompatible envs.

5. Paper alignment:
- For ambiguous behavior, fallback rules from `planning/ThermoGFN-IF.tex` control decisions.

---

## 3) Target Repository Layout (Aligned to current repo)

## 3.1 Keep current directories; add implementation modules conservatively

- `config/`: method and environment YAMLs.
- `data/`: datasets, splits, cached oracle outputs.
- `scripts/`: all runnable prep/train/infer/eval entrypoints.
- `train/`: reusable Python modules (optional, currently empty and suitable).
- `planning/`: methodology docs and plans.

## 3.2 Script taxonomy under `./scripts`

Create these subdirectories:

```text
scripts/
  prep/
  train/
  infer/
  eval/
  orchestration/
  env/
```

- `scripts/prep/*`: dataset prep, indexing, baseline scoring, split materialization.
- `scripts/train/*`: Method III/II/I training entrypoints.
- `scripts/infer/*`: conditioned/unconditioned design generation and rescoring.
- `scripts/eval/*`: oracle validation, teacher-student diagnostics, design metrics.
- `scripts/orchestration/*`: round runners and multi-stage pipelines.
- `scripts/env/*`: wrappers to invoke model-specific env commands safely.

---

## 4) Script Inventory to Implement

All filenames below are planned entrypoints in `./scripts`.

## 4.1 Environment and health scripts

- `scripts/env/check_envs.sh`
  - Verify required conda envs with exact names:
    - required: `ADFLIP`, `spurs`, `bioemu`, `uma-qc`
    - optional: `protrek`, `foundry`, `rfd3`
  - Emit machine-readable status:
    - `ready`: env exists and basic import check passes,
    - `exists_unchecked`: env exists but import check skipped,
    - `missing`: env not found.
- `scripts/env/print_versions.py`
  - Capture torch/cuda/package versions into run metadata.
- `scripts/env/dispatch.py`
  - Unified subprocess runner:
    - `--env-name`
    - `--cmd`
    - `--require-ready` (fail if env not marked `ready` by `check_envs`)
    - `--env-status-json` (path produced by `check_envs.sh`)
    - structured stdout/stderr capture.

## 4.2 Preparation scripts

- `scripts/prep/01_validate_monomer_split.py`
  - Validate integrity of `data/rfd3_splits/unconditional_monomer_protrek35m`.
- `scripts/prep/02_build_training_index.py`
  - Build design index table from split files (paths, sequence length, chain metadata).
- `scripts/prep/03_compute_baselines.py`
  - For each seed (`x_base`), compute baseline property vector placeholders and oracle eligibility tags.
- `scripts/prep/04_prepare_complex_registry.py`
  - Prepare registry for future mixed-modality datasets (no-op until splits arrive).
- `scripts/prep/05_materialize_round0_dataset.py`
  - Build initial `D0` for Method III using monomer split.

## 4.3 Oracle adapter scripts

- `scripts/prep/oracles/spurs_score_single.py`
- `scripts/prep/oracles/spurs_score_multi.py`
- `scripts/prep/oracles/bioemu_sample_and_features.py`
- `scripts/prep/oracles/uma_md_screen.py`
- `scripts/prep/oracles/fuse_scores.py`

These scripts should read/write standardized records, never rely on hidden mutable state.

## 4.4 Method III training scripts (first implementation target)

- `scripts/train/m3_fit_surrogate.py`
- `scripts/train/m3_train_teacher_gfn.py`
- `scripts/train/m3_distill_student.py`
- `scripts/train/m3_generate_student_pool.py`
- `scripts/train/m3_select_bioemu_batch.py`
- `scripts/train/m3_select_uma_batch.py`
- `scripts/train/m3_append_labels.py`
- `scripts/orchestration/m3_run_round.py`
- `scripts/orchestration/m3_run_experiment.py`

## 4.5 Method I and Method II scripts

- `scripts/train/m1_train_simul_mf.py`
- `scripts/train/m2_stage1_spurs.py`
- `scripts/train/m2_stage2_bioemu.py`
- `scripts/train/m2_stage3_uma.py`
- `scripts/orchestration/m2_run_curriculum.py`

## 4.6 Inference scripts

- `scripts/infer/generate_target_conditioned.py`
- `scripts/infer/generate_unconditioned.py`
- `scripts/infer/rescore_and_select.py`
- `scripts/infer/report_candidate_card.py`

## 4.7 Evaluation scripts

- `scripts/eval/eval_oracle_calibration.py`
- `scripts/eval/eval_m3_teacher_student.py`
- `scripts/eval/eval_design_metrics.py`
- `scripts/eval/eval_ablations.py`
- `scripts/eval/eval_large_protein_breakdown.py`

## 4.8 Existing scripts to retain and integrate

Do not remove current scripts; integrate them into the new orchestration layer:
- `scripts/run_rfd3_inference.sh`
  - Keep as primary structure-generation script for monomer/dimer/ligand design assets.
- `scripts/protrek_cluster_split.py`
  - Keep as primary sequence/structure clustering split engine.
- `scripts/run_protrek_split_unconditional_monomer.sh`
  - Keep as convenience wrapper for monomer split refresh.
- `scripts/build_ccd_ligand_library.py` and `scripts/decompress_rfd3_cifs.sh`
  - Keep as utilities for ligand preparation and file handling.

New orchestrators should call these existing scripts rather than re-implementing their logic.

---

## 5) Standardized Data Contracts (Required for robustness)

## 5.1 Candidate record (`candidate.jsonl` or parquet)

Required fields:
- `candidate_id`
- `backbone_id`
- `split` (`train|test|val`)
- `task_type` (`monomer|ppi|ligand`)
- `sequence`
- `mutations` (list)
- `K`
- `seed_id`
- `structure_path` (if materialized)
- `prepared_atom_count`
- `eligibility.bioemu`
- `eligibility.uma_whole`
- `eligibility.uma_local`
- `run_id`, `round_id`

## 5.2 Oracle record (`oracle_scores.parquet`)

Required fields:
- `candidate_id`
- `spurs_mean`, `spurs_std`, `spurs_mode` (`single|double|higher`)
- `bioemu_features` (serialized struct)
- `bioemu_calibrated`, `bioemu_std`
- `uma_features` (serialized struct)
- `uma_calibrated`, `uma_std`
- `rho_B`, `rho_U`
- fused components (`z_*`) and final `reward`
- timestamps and env metadata.

## 5.3 Round dataset (`D_r`)

- `D_r` is append-only by round.
- Track source provenance:
  - teacher-generated,
  - student-generated,
  - replay-elite,
  - oracle-promoted.

---

## 6) Method III Implementation Plan (Detailed)

## 6.1 Phase M3-0: Infrastructure hardening

Deliverables:
- Script scaffolding under `scripts/*`.
- Config loading and manifest writing.
- Environment-dispatched oracle wrappers.
- Candidate and oracle data contracts.

Acceptance:
- `scripts/orchestration/m3_run_round.py --dry-run` completes with full planned steps and file paths.

## 6.2 Phase M3-1: Monomer-only functional loop (using existing split)

Dataset source:
- `data/rfd3_splits/unconditional_monomer_protrek35m`.

Round flow:
1. Build `D0` from monomer split (seed objects + baseline metadata).
2. Fit surrogate ensemble on `D_r`.
3. Train teacher GFlowNet (on-policy + off-policy reconstructed trajectories).
4. Distill one-shot student.
5. Generate student pool (`10^4` to `10^5` range).
6. SPURS dense rescore.
7. Select BioEmu subset (initially optional in M3-1, then required in M3-2).
8. Append results to `D_{r+1}`.

Acceptance:
- End-to-end for 1-2 rounds.
- Teacher-student KL is computed by mutation-order bin.
- No collapse to trivial low-K only proposals.

## 6.3 Phase M3-2: Full tri-fidelity in round loop

Add:
- BioEmu feature extraction and calibration.
- UMA sparse branch with atom-budget routing.
- Reliability-gated reward fusion.
- High-order quota logic for expensive evaluations.

Acceptance:
- Rounds complete with SPURS + BioEmu + UMA labels.
- Expensive budget controls respected.
- Disagreement-triggered acquisition operational.

## 6.4 Phase M3-3: Complex and target-conditioned extension hooks

Given current data state:
- Keep monomer as production path.
- Implement complex-capable interfaces now (no hard data dependency) so mixed splits can plug in later.

Add:
- Decomposition-map capable candidate schema.
- Bound/separated bookkeeping in scorer APIs.
- Target-conditioned generation with bounded retries and best-failed fallback.

Acceptance:
- API-level tests pass on synthetic complex stubs.
- Real complex data remains feature-flagged until splits exist in `data/`.

## 6.5 Phase M3-4: Validation suite

Mandatory diagnostics:
- Teacher-student KL on held-out pools (stratified by `x0`, `K=1`, `K=2`, `K>=3`).
- Top-k overlap by identity and cluster.
- Tail coverage (`K>=3`).
- Oracle-stage preservation (post-SPURS and post-BioEmu ranking consistency).

## 6.6 Method III round contract (strict I/O)

`scripts/orchestration/m3_run_round.py` should enforce this deterministic sequence:

1. Input load:
- Inputs:
  - `D_r` path
  - split selector (`train` by default)
  - round config yaml
  - `round_id`, `run_id`, `seed`
- Outputs:
  - `round_manifest.json` initialized

2. Surrogate fit (`m3_fit_surrogate.py`):
- Input:
  - `D_r` labeled rows
- Output:
  - `surrogate_round_{r}.ckpt`
  - calibration report and uncertainty diagnostics

3. Teacher train (`m3_train_teacher_gfn.py`):
- Input:
  - replay + reconstructed trajectories from `D_r`
  - reward fusion config
- Output:
  - `teacher_round_{r}.ckpt`
  - rollout logs by mutation-order bin

4. Student distill (`m3_distill_student.py`):
- Input:
  - teacher samples + replay elites + evaluated set
- Output:
  - `student_round_{r}.ckpt`
  - student calibration stats (`K`, ranking, anchor losses)

5. Student proposal pool (`m3_generate_student_pool.py`):
- Input:
  - `student_round_{r}.ckpt`
  - pool size config
- Output:
  - `candidate_pool_round_{r}.parquet`

6. Dense SPURS pass:
- Input:
  - candidate pool
- Output:
  - `candidate_pool_spurs_round_{r}.parquet`

7. BioEmu selection + scoring:
- Input:
  - SPURS-scored pool
- Output:
  - `bioemu_scored_round_{r}.parquet`
  - `bioemu_features_round_{r}.parquet`

8. UMA selection + scoring:
- Input:
  - BioEmu-scored pool
- Output:
  - `uma_scored_round_{r}.parquet`
  - MD diagnostics + uncertainty files

9. Fusion + append:
- Input:
  - SPURS/BioEmu/UMA partial labels
- Output:
  - `D_{r+1}.parquet`
  - `round_summary_{r}.json`

10. Validation:
- Input:
  - teacher, student, `D_{r+1}`
- Output:
  - KL/top-k/tail/oracle-stage metrics for round gatekeeping.

Round should fail closed:
- if any mandatory artifact is missing,
- if schema validation fails,
- if run metadata does not match requested config hash.

---

## 7) Model Adapter Design (per reviewed inference code)

## 7.1 ADFLIP adapter

Adapter target:
- Wrap `DiscreteFlow_AA.sample` and `adaptive_sample` behavior while preserving sidechain packing requirements.

Implementation notes:
- Initial callable can shell to `models/ADFLIP/test/design.py` for reproducibility.
- Mid-term: direct Python import adapter from `models/ADFLIP/model/discrete_flow_aa.py`.
- Keep PIPPack post-processing explicit.

I/O contract:
- Input: structure path, sampling mode, thresholds/steps.
- Output: designed sequence, logits summary, optional packed structure paths.

## 7.2 SPURS adapter

Use:
- `models/SPURS/spurs/inference.py` APIs.

Expose:
- single mutation matrix scoring,
- multi-mutation scoring,
- uncertainty via checkpoint ensemble (top-5 default per paper).

## 7.3 BioEmu adapter

Use:
- `models/bioemu/src/bioemu/sample.py` (`main`) with deterministic seeds.

Expose:
- sampling call,
- feature extraction (folded/disrupted occupancy and summary stats),
- optional sidechain relax on shortlisted outputs.

## 7.4 UMA adapter (FAIRChem)

Use:
- `pretrained_mlip.get_predict_unit`
- `FAIRChemCalculator`
- ASE Langevin setup for screening MD.

Expose:
- whole-protein vs local/hybrid routing,
- temperature ladder execution,
- calibrated scalar + uncertainty decomposition.

## 7.5 GFlowNet baseline adapter

Use:
- concepts and utilities from `models/gflownets/*` as references, not as drop-in production code.

Reason:
- Legacy toy/molecule pipelines do not directly match ThermoGFN state/reward/object schema.

---

## 8) Dataset and Split Plan (current + future)

## 8.1 Current status (use immediately)

Use existing monomer split:
- `data/rfd3_splits/unconditional_monomer_protrek35m`

Actions now:
- validate and index this split,
- derive `D0` bootstrap set,
- run Method III monomer rounds.

## 8.2 Future mixed-modality splits (to add when generated)

Planned directories:
- `data/rfd3_splits/unconditional_dimer_*`
- `data/rfd3_splits/small_molecule_binders_*`
- `data/rfd3_splits/complex_mixed_*`

Plan behavior until then:
- keep complex logic in code and configs,
- keep training/inference feature flags off for absent modalities,
- avoid placeholder synthetic claims.

---

## 9) Hyperparameters and Defaults (paper-following registry)

This section is the implementation source of truth for defaults/ranges.

## 9.1 Core GFlowNet / ADFLIP

- Base ADFLIP checkpoint: released all-atom checkpoint.
- Seed generation steps: default `10`, range `5-20`.
- Seed purity threshold: `tau_seed = 0.9`.
- Seed max Euler steps: `32`.
- Final seed PIPPack pass: `1`.
- Mutation policy: no hard Hamming cap.
- `K_min` mixture:
  - `0.30 * delta(1)`
  - `0.35 * Uniform{2,3,4}`
  - `0.20 * Uniform{5..8}`
  - `0.15 * Uniform{9..min(L,16)}`
- Optional revision rollout guard: up to `2L` (ablation only).
- Objective: `SubTB(lambda)` default.
- `lambda`: default `0.9`, range `0.7-0.97`.
- Full-context refresh steps: default `4`, range `2-8`.
- Terminal repack steps: default `50`, range `20-60`.

## 9.2 Loss weights

Shared regularizers:
- `lambda_anchor = 0.1`
- `lambda_distill = 0.2`
- `lambda_den = 0.05`
- `lambda_pack = 0.05`

Goal-conditioned extras:
- `lambda_prop in [0.1, 0.5]`
- `lambda_bind in [0.05, 0.25]`

Method III student:
- `(lambda_distill, lambda_K, lambda_rank, lambda_anchor, lambda_side) = (1.0, 0.2, 0.25, 0.5, 0.2)`

## 9.3 Optimizer

- AdamW.
- New heads LR: `1e-4` (range `5e-5` to `3e-4`).
- Unfrozen ADFLIP LR: `2e-5` (range `1e-5` to `5e-5`).
- Scheduler: cosine.
- Warmup: `5%`.

## 9.4 Replay and batching

- Replay size default `100k` (range `20k-500k`).
- On-policy:replay:anchor = `0.6:0.3:0.1`.
- Replay bins: `x_base/x0`, `K=1`, `K=2`, `K>=3`.

## 9.5 Reward transform and weights

Transform:
- `epsilon_R = 1e-6`
- clip `[-8, +8]`
- `tau_R = 1`

Weights:
- `w_S = 1.0`
- `w_B = 0.75` overall; `1.0` in large-protein setting
- `w_U = 1.0` whole-protein UMA; `0.25-0.6` local/hybrid
- `w_bind = 0.5` complex tasks; `0` monomer
- `w_comp = 0.25` complex tasks
- `w_target = 0` unconstrained; `0.75` conditioned
- `w_pack = 0.25`
- `w_ord = 0.25`
- `w_rad = 0.0` default; optional `0.05-0.20`
- `w_OOD = 0.15`

Other reward defaults:
- SPURS uncertainty ensemble: top `5` checkpoints.
- `packunc` penalty threshold: `0.25` (range `0.15-0.40`).

Reliability gates:
- `rho_B`: `1` in-scope, `0.5` approximate, `0` skip.
- `rho_U`: `1` in-scope/calibrated, `0.35-0.6` above soft range, `0` skip.

## 9.6 Acquisition defaults

BioEmu acquisition coefficients:
- `(alpha1..alpha6) = (0.40, 0.15, 0.15, 0.10, 0.10, 0.10)`

UMA acquisition coefficients:
- `(beta1..beta6) = (0.35, 0.15, 0.20, 0.10, 0.10, 0.10)`

Method I screening rates:
- BioEmu default `25%` (range `15-50%`).
- UMA default `3% overall` (range `1-5%`).

## 9.7 BioEmu defaults

- Train-time samples `M_B = 2048` (range `512-4096`).
- Shortlist `10k` (range `5k-20k`).
- `T_ref ~ 300 K`.
- `epsilon_B = 1e-4`.
- Basin thresholds:
  - `(qF_B, qU_B, rF_B, rU_B, hF_B) = (0.70, 0.40, 3.0A, 7.0A, 0.65)`

## 9.8 UMA defaults

- Preferred whole-protein budget: `<=8000` atoms.
- Soft transition: `8000-10000`.
- Training hard cap: `<=10000` atoms.
- BioEmu-dominant switch: `N_atom > 8000` or weak UMA calibration.
- MD timestep: `0.1 fs` (range `0.05-0.25`).
- Temperature ladder: `300,330,360,390,420 K`.
- Screening production: `25 ps` per temp (range `10-50`).
- Replicates: `4` (range `2-8`).
- Local fallback radius: `8A` (range `6-10`).
- Minimization threshold: max force `<0.05 eV/A`.
- Equilibration: `5 ps` per temp.
- Shortlist production: `>=100 ps`.
- Protonation default: deterministic at `pH 7.0`.
- Basin thresholds:
  - `(qF,qU,rF,rU,hF) = (0.75,0.35,2.5A,6.0A,0.7)`

## 9.9 Complex and target-conditioned defaults

- `U0 = 25k` (range `10k-100k`).
- `U1 = 75k` (range `30k-150k`).
- Steady-state mix default `0.333/0.333/0.333`.
- Alternate profile `0.60/0.25/0.15`.
- `r_ctx = 10A` (range `8-12`).
- Max context atoms/residue: `256` (range `128-512`).
- Symmetry tying: on when map available.
- Target-conditioning dropout `0.20` (range `0.05-0.40`).
- `w_target` default `0.75` (range `0.25-1.5`).
- Tm tolerance default `+-2K`.
- dGbind/ddGbind tolerance default `+-0.5 kcal/mol`.
- Candidates/retry default `64` (range `16-256`).
- `R_max = 16` (range `4-64`).

## 9.10 Method III defaults

- Active-learning rounds: `8` (range `5-15`).
- `gamma_off`: `0.5` early -> `0.25` later (range `0.25-0.60`).
- Teacher steps/round: `30k` (range `20k-40k`).
- Student distill steps/round: `15k` (range `10k-25k`).
- Surrogate ensemble size: `8` (range `4-16`).
- Student pool/round: `50k` (range `1e4-1e5`).
- BioEmu batch/round: `512` (range `128-2048`).
- UMA batch/round: `64` (range `16-128`).
- Control tokens: `{beta, q, rho}`.
- KL tolerance: `0.10 nats/token` (range `0.05-0.20`).
- High-order expensive quota: `30%` (range `20-40%`).

---

## 10) Method I and Method II (post Method III)

## 10.1 Method I (`Simul-MF`)

Implement after Method III is stable:
- on-policy rollouts,
- asynchronous BioEmu/UMA acquisition,
- hierarchical correction models,
- stagewise curriculum:
  - baseline + full-redesign + single,
  - explicit double,
  - SPURS+BioEmu,
  - full tri-fidelity.

## 10.2 Method II (`Seq-Curr`)

Implement staged runner:
- Stage 1A SPURS exploration,
- Stage 1B explicit double epistasis,
- Stage 2 BioEmu refinement,
- Stage 3 UMA refinement,
- optional deployment distillation.

---

## 11) Evaluation and Acceptance Criteria

## 11.1 Oracle validation before design claims

Order:
1. SPURS calibration.
2. BioEmu calibration.
3. UMA validation (whole vs fallback).
4. Cross-oracle agreement audit.

## 11.2 Method III specific

Must report per round:
- KL (teacher vs student),
- top-k overlap,
- tail coverage,
- oracle-stage preservation.

## 11.3 Design metrics

- Best-of-N and top-k fused score.
- Diversity and cluster count.
- Mutation-order histogram.
- Structure plausibility.
- Cross-oracle agreement.
- Large-protein stratification.

## 11.4 Ablations

Minimum:
- no BioEmu,
- no UMA,
- no uncertainty terms,
- SPURS-guided proposals,
- higher-order SPURS variant,
- LoRA/adapters vs full fine-tune,
- MINT and LigandMPNN swap-ins,
- teacher-free one-shot baseline,
- hard vs soft BioEmu-dominant routing.

## 11.5 Robustness test matrix (must pass before large runs)

1. Dry-run matrix:
- `--dry-run` for every script type (`prep`, `train`, `infer`, `eval`).

2. Resume/idempotence:
- Re-run same `run_id` and confirm no duplicate candidate IDs.

3. Seed determinism:
- Same seed/config yields identical candidate ordering before oracle stochastic branches.

4. Schema enforcement:
- Corrupt one input field and confirm explicit validation failure.

5. Env routing:
- Force wrong env for a model adapter and confirm readable failure diagnostics.

6. Oracle dropout:
- Run with BioEmu or UMA disabled and verify fusion still operates with gates.

7. Large-protein routing:
- Verify candidates above 8k atoms route BioEmu-dominant and respect soft UMA behavior.

---

## 12) Risk Register and Mitigations

1. Multi-env fragility:
- Mitigate with env dispatcher + strict file contracts + retries.

2. Oracle drift / reward hacking:
- Mitigate with reliability gates, structural penalties, disagreement-driven acquisition.

3. Student mode collapse:
- Mitigate with per-bin KL monitoring + tail quota + teacher-only refresh samples.

4. Data heterogeneity across future complex modalities:
- Mitigate by schema-first design and feature-flagged task activation.

---

## 13) Execution Timeline

Phase A (Week 1-2):
- Script scaffolding + env dispatch + monomer split validation.

Phase B (Week 3-5):
- Method III M3-1 monomer loop with SPURS + surrogate + teacher-student.

Phase C (Week 6-8):
- Method III M3-2 with BioEmu + UMA integration.

Phase D (Week 9-10):
- Method III validation and ablation harness.

Phase E (Week 11-13):
- Method I implementation.

Phase F (Week 14-15):
- Method II implementation.

Phase G (ongoing):
- integrate future mixed-modality splits as they are generated in `data/`.

---

## 14) Immediate Next Actions

1. Implement `scripts/prep/01_validate_monomer_split.py` and emit integrity report.
2. Implement standardized candidate/oracle parquet schemas.
3. Implement env dispatcher and SPURS adapter first (fastest loop closure).
4. Build M3 round runner skeleton (`--dry-run`, `--round-id`, `--resume`).
5. Wire teacher training + student distillation on monomer split only.
6. Add BioEmu branch, then UMA branch.
7. Turn on full validation metrics before beginning Method I/II.

---

## 15) Adapter-Accurate Runtime Profiles (from `./models` READMEs + inference code)

This section is binding for wrapper implementation in `./scripts`. If any wrapper behavior conflicts with this section, wrapper behavior is incorrect.

## 15.1 ADFLIP wrapper profile

Source-anchored facts:
- Environment in `models/ADFLIP/README.md`: conda env `ADFLIP`, Python 3.10.
- Reference inference entrypoint: `models/ADFLIP/test/design.py`.
- Supported sampling modes from code: `fixed` and `adaptive`.
- Key runtime args from code:
  - `--pdb`
  - `--ckpt`
  - `--device`
  - `--method {fixed|adaptive}`
  - `--dt`
  - `--steps`
  - `--threshold`

Wrapper requirements:
- `scripts/env/dispatch.py` must support running `models/ADFLIP/test/design.py` as subprocess in `ADFLIP` env.
- `scripts/prep/03_compute_baselines.py` and later inference scripts must expose ADFLIP seed settings:
  - default: `method=adaptive`, `steps=32` for seed pass, `threshold=0.9`.
  - range: steps `5-32` for seed generation.
- Wrapper must parse generated sequence from stdout and write structured output record.
- If direct import is used instead of subprocess, behavior must remain equivalent to CLI defaults above.

## 15.2 SPURS wrapper profile

Source-anchored facts:
- Environment in `models/SPURS/README.md`: conda env `spurs`, Python 3.7, torch 1.12.
- Inference APIs in `models/SPURS/spurs/inference.py`:
  - `get_SPURS` / `get_SPURS_from_hub`
  - `get_SPURS_multi_from_hub`
  - `parse_pdb`
  - `parse_pdb_for_mutation`

Wrapper requirements:
- `scripts/prep/oracles/spurs_score_single.py`:
  - use `get_SPURS_from_hub` by default.
  - call `parse_pdb` once per structure.
  - return per-position x amino-acid ddG matrix, with explicit sign convention field.
- `scripts/prep/oracles/spurs_score_multi.py`:
  - use `get_SPURS_multi_from_hub`.
  - convert mutation lists to tensors via `parse_pdb_for_mutation`.
  - support `K=2` explicit stage as primary supervised mode.
- Both scripts must emit:
  - raw SPURS outputs,
  - harmonized score (`-ddG_destab`),
  - uncertainty estimate (ensemble/std if enabled).

## 15.3 BioEmu wrapper profile

Source-anchored facts:
- Environment in `models/bioemu/README.md`: pip package, Python 3.10-3.12.
- Inference entrypoint: `python -m bioemu.sample` / `bioemu.sample.main`.
- Monomer scope limitation is explicit in README.
- Key args in `bioemu.sample.main`:
  - `sequence`, `num_samples`, `output_dir`
  - `batch_size_100`
  - `model_name` (`bioemu-v1.0|v1.1|v1.2`)
  - `filter_samples`
  - `base_seed`
  - optional steering config.

Wrapper requirements:
- `scripts/prep/oracles/bioemu_sample_and_features.py` must:
  - enforce monomer eligibility gate before invocation.
  - call sampling with default `model_name=bioemu-v1.1` unless config overrides.
  - default `filter_samples=True`.
  - compute folded/disrupted occupancy features and calibrated scalar.
- Sidechain reconstruction (`bioemu.sidechain_relax`) stays optional and is only used for shortlisted candidates, not default training throughput.

## 15.4 UMA wrapper profile (FAIRChem)

Source-anchored facts:
- Runtime environment for this repo implementation: conda env `uma-qc`.
- `models/fairchem/README.md` specifies ASE + `FAIRChemCalculator` usage.
- Predictor API: `pretrained_mlip.get_predict_unit("uma-s-1p1"|"uma-m-1p1", ...)`.
- Molecular task for protein-like systems: `task_name="omol"`.
- MD examples use ASE Langevin at `0.1 fs`.
- Multi-GPU acceleration uses `workers=N`.

Wrapper requirements:
- `scripts/prep/oracles/uma_md_screen.py` must:
  - execute in env `uma-qc` (never in `ADFLIP`, `spurs`, or `bioemu` envs).
  - expose `--model-name` (`uma-s-1p1` default; `uma-m-1p1` optional).
  - expose `--workers` and `--inference-settings`.
  - implement whole-protein vs local/hybrid routing based on prepared atom count and policy gates.
  - run temperature ladder `300/330/360/390/420 K` with replicate control.
  - emit calibrated scalar and full uncertainty decomposition.

## 15.5 Existing top-level script integration requirement

- `scripts/run_rfd3_inference.sh` remains source for structure-generation assets.
- New preparation scripts must consume outputs from this script, never duplicate its generation logic.

---

## 16) Script-Level CLI Contracts and Exit Codes

All new scripts in `./scripts` must include:
- `--config` (yaml path, optional if defaults fully defined)
- `--run-id` (required for mutating operations)
- `--dry-run` (prints resolved inputs/outputs and exits)
- deterministic seed controls where applicable.
- `--env-name` for adapter scripts that invoke model runtimes via dispatcher.

Standard exit codes:
- `0`: success.
- `2`: argument/config validation failure.
- `3`: schema validation failure.
- `4`: upstream adapter runtime failure.
- `5`: missing mandatory artifact.
- `6`: round gate failure (quality criteria not met).

## 16.1 Mandatory args per critical script

`scripts/orchestration/m3_run_round.py`:
- required:
  - `--run-id`
  - `--round-id`
  - `--dataset-path`
  - `--output-dir`
- optional:
  - `--resume`
  - `--max-retries-per-step` (default 2)
  - `--strict-gates` (default on)
  - `--dry-run`

`scripts/orchestration/m3_run_experiment.py`:
- required:
  - `--run-id`
  - `--num-rounds`
  - `--dataset-path`
  - `--output-root`
- optional:
  - `--start-round` (default 0)
  - `--resume`
  - `--stop-after-round`

`scripts/train/m3_fit_surrogate.py`:
- required:
  - `--input-dr`
  - `--round-id`
  - `--output-dir`
- optional:
  - `--ensemble-size` (default 8)
  - `--seed`

`scripts/train/m3_train_teacher_gfn.py`:
- required:
  - `--input-dr`
  - `--round-id`
  - `--output-dir`
- optional:
  - `--steps` (default 30000)
  - `--gamma-off` (default 0.5 early, scheduler-supported)
  - `--beta-min`, `--beta-max`

`scripts/train/m3_distill_student.py`:
- required:
  - `--teacher-ckpt`
  - `--candidate-pool`
  - `--round-id`
  - `--output-dir`
- optional:
  - `--steps` (default 15000)
  - `--kl-target` (default 0.10)

`scripts/prep/oracles/spurs_score_single.py`:
- required:
  - `--candidate-path`
  - `--output-path`
- optional:
  - `--repo-id` (default `cyclization9/SPURS`)
  - `--ensemble-checkpoints` (default 5)

`scripts/prep/oracles/bioemu_sample_and_features.py`:
- required:
  - `--candidate-path`
  - `--output-path`
- optional:
  - `--num-samples` (default 2048)
  - `--model-name` (default `bioemu-v1.1`)
  - `--batch-size-100` (default from config)

`scripts/prep/oracles/uma_md_screen.py`:
- required:
  - `--candidate-path`
  - `--output-path`
- optional:
  - `--model-name` (default `uma-s-1p1`)
  - `--workers` (default 1)
  - `--temps` (default `300,330,360,390,420`)
  - `--replicates` (default 4)

## 16.2 Required script-to-env routing matrix

This mapping is mandatory for implementation and orchestration:

- `ADFLIP` env:
  - `scripts/prep/03_compute_baselines.py` (when invoking ADFLIP seed generation)
  - ADFLIP adapter calls from training/inference orchestration
- `spurs` env:
  - `scripts/prep/oracles/spurs_score_single.py`
  - `scripts/prep/oracles/spurs_score_multi.py`
- `bioemu` env:
  - `scripts/prep/oracles/bioemu_sample_and_features.py`
  - optional `bioemu.sidechain_relax` shortlist jobs
- `uma-qc` env:
  - `scripts/prep/oracles/uma_md_screen.py`
  - any FAIRChem/ASE UMA utilities

Hard rule:
- Orchestration scripts (`m3_run_round.py`, `m3_run_experiment.py`) run in a neutral controller env and must call model scripts through `scripts/env/dispatch.py` with explicit `--env-name`.
- No model adapter may assume it is running in the correct env implicitly.

---

## 17) Artifact Tree, Naming, and Manifest Requirements

Canonical run layout:

```text
runs/
  {run_id}/
    config/
      resolved_config.yaml
      config_hash.txt
    rounds/
      round_{r:03d}/
        manifests/
          round_manifest.json
          round_gate_report.json
        data/
          D_r.parquet
          candidate_pool_round_{r}.parquet
          candidate_pool_spurs_round_{r}.parquet
          bioemu_scored_round_{r}.parquet
          uma_scored_round_{r}.parquet
          D_{r+1}.parquet
        models/
          surrogate_round_{r}.ckpt
          teacher_round_{r}.ckpt
          student_round_{r}.ckpt
        metrics/
          surrogate_metrics.json
          teacher_metrics.json
          student_metrics.json
          round_metrics.json
        logs/
          *.stdout.log
          *.stderr.log
```

Artifact naming rules:
- Must include `run_id` and `round_id` in metadata and filenames where applicable.
- No overwriting immutable artifacts:
  - use `_v2`, `_v3` suffix for retries.
- Every artifact must have:
  - checksum,
  - schema version,
  - producer script version.

---

## 18) Strict Schema Definitions (with required types)

## 18.1 Candidate record type contract

- `candidate_id`: string (uuid or deterministic hash).
- `run_id`: string.
- `round_id`: int.
- `task_type`: enum `monomer|ppi|ligand`.
- `backbone_id`: string.
- `seed_id`: string.
- `sequence`: string (AA uppercase).
- `mutations`: list of strings (e.g., `["W1A","V2Y"]`).
- `K`: int, must equal `len(mutations)`.
- `prepared_atom_count`: int.
- `eligibility.bioemu`: bool.
- `eligibility.uma_whole`: bool.
- `eligibility.uma_local`: bool.
- `source`: enum `teacher|student|replay|oracle_promoted`.
- `schema_version`: string.

Validation invariants:
- If `task_type=monomer`, component decomposition fields are optional.
- If `task_type in {ppi, ligand}`, decomposition map must be present.
- `K=0` only allowed for baseline/seed records.

## 18.2 Oracle score record type contract

- `candidate_id`: string (foreign key).
- `spurs_mean`: float nullable.
- `spurs_std`: float nullable.
- `spurs_mode`: enum `single|double|higher`.
- `bioemu_calibrated`: float nullable.
- `bioemu_std`: float nullable.
- `uma_calibrated`: float nullable.
- `uma_std`: float nullable.
- `rho_B`: float in `[0,1]`.
- `rho_U`: float in `[0,1]`.
- `z_S`, `z_B`, `z_U`, `z_bind`, `z_comp`, `z_target`, `z_pack`, `z_ord`, `z_rad`, `z_OOD`: float nullable.
- `reward`: float (strictly positive).

Validation invariants:
- `reward > 0` always.
- if all oracle channels null and no fused fallback: hard fail.
- if `K>=3` and only SPURS available: `z_ord` penalty must be active unless explicitly waived in config.

---

## 19) Round Gates and Promotion Criteria (Method III)

Each round must pass gates before `D_{r+1}` is accepted as input to next round.

Gate A: data integrity
- schema pass rate `100%`.
- duplicate `candidate_id` rate `0%`.

Gate B: teacher-student quality
- KL on held-out pool by bin:
  - `x0/K=1/K=2`: <= `0.10` nats/token.
  - `K>=3`: <= `0.15` nats/token.
- top-k overlap (cluster-level): >= `0.60`.

Gate C: diversity and collapse prevention
- fraction of unique sequences in top-1000 >= `0.85`.
- `K>=3` share in expensive (BioEmu+UMA) batch >= `0.30`.
- immediate-stop or trivial proposals do not exceed configured ceiling (default `<=0.35` of pool).

Gate D: oracle process health
- SPURS success rate >= `0.99`.
- BioEmu success rate >= `0.95` on eligible candidates.
- UMA success rate >= `0.90` on selected candidates.

If any gate fails:
- round status = `failed_gate`.
- no promotion to `D_{r+1}`.
- mandatory remediation plan generated in `round_gate_report.json`.

---

## 20) Failure, Retry, Resume, and Idempotence Policy

Retry policy:
- transient adapter failures:
  - max retries per item: `2` (default).
  - exponential backoff: `10s`, `30s`.
- persistent per-item failures:
  - mark item as failed with reason code.
  - keep pipeline progressing if failure rate remains below gate thresholds.

Resume policy:
- orchestration scripts must be resumable by step.
- completed steps are skipped only if:
  - artifact exists,
  - checksum matches,
  - schema validates,
  - producer script/config hash matches.
- otherwise step is recomputed and previous artifact archived.

Idempotence policy:
- rerunning same `(run_id, round_id)` with same config hash must not create duplicate logical candidates.
- deterministic candidate hash key:
  - hash(`backbone_id`, `sequence`, `task_type`, `seed_id`).

Fail-closed conditions:
- missing `round_manifest.json`.
- config hash mismatch.
- schema drift without explicit migration.

---

## 21) Test Pyramid and CI Acceptance

Unit tests:
- schema validators.
- reward fusion math.
- acquisition score computation.
- routing logic (`rho_B`, `rho_U`, atom-count branches).

Integration tests:
- per-adapter smoke tests:
  - ADFLIP wrapper command construction.
  - SPURS single and multi mode.
  - BioEmu sampling + feature extraction.
  - UMA md wrapper dry-run.
- round orchestration dry-run with fixture dataset.

End-to-end tests:
- one mini Method III round on tiny monomer subset.
- produces all mandatory artifacts and metrics.
- resumes successfully after forced interruption at each step boundary.

CI minimum pass criteria:
- all unit tests pass.
- integration smoke suite pass.
- e2e mini-round pass within configured time budget.

---

## 22) Ambiguity Resolution Rule (Tie-break to paper)

Whenever implementation details are unclear or conflicting:

1. Primary source of truth: `planning/ThermoGFN-IF.tex`.
2. Secondary source: this `PLAN-1.md`.
3. Tertiary source: model READMEs and inference code under `./models`.

Decision process:
- log ambiguity and chosen interpretation in `runs/{run_id}/config/decision_log.md`.
- include exact section reference from `ThermoGFN-IF.tex`.
- if ambiguity affects default value, include both:
  - chosen default,
  - paper-allowed range.

No silent deviations are allowed from paper-defined defaults/ranges for:
- reward construction,
- mutation-order curriculum,
- temperature ladders,
- acquisition rates,
- uncertainty/risk-adjusted scoring.

---

## 23) Updated Immediate Build Sequence (Execution-Ready)

1. Implement env dispatcher + standardized exit codes (`scripts/env/dispatch.py`).
2. Implement and test schema validators (`candidate`, `oracle`, `round_manifest`).
3. Implement SPURS wrappers (`single`, `multi`) and integrate into one mini round.
4. Implement Method III core loop (surrogate -> teacher -> student -> pool -> SPURS).
5. Add round gates (A-D) and enforce fail-closed behavior.
6. Add BioEmu wrapper and feature calibration path.
7. Add UMA wrapper with atom-count routing and uncertainty decomposition.
8. Enable full tri-fidelity fusion and run 2-3 monomer rounds.
9. Freeze monomer baseline; then prepare feature-flag hooks for mixed modalities when splits arrive.

---

## 24) Conda Environment Readiness and Enforcement

Implementation must treat env readiness as a first-class precondition.

## 24.1 Required env set

- `ADFLIP`
- `spurs`
- `bioemu`
- `uma-qc`

## 24.2 Current machine snapshot (from `conda env list` during planning update)

- detected: `ADFLIP`, `bioemu`, `uma-qc`
- not detected at check time: `spurs`

Note:
- This aligns with user context that some envs are still being created.
- `scripts/env/check_envs.sh` must be re-run before first implementation run and before long experiment launches.

## 24.3 Enforcement policy

- `m3_run_round.py` and `m3_run_experiment.py` must fail fast if required envs are not `ready`.
- exception:
  - a stage can run with a reduced oracle set only when explicitly configured (e.g., `--allow-missing-oracle-env spurs`) and logged in round manifest.
- all missing-env overrides must be recorded in:
  - `runs/{run_id}/config/decision_log.md`
  - `round_manifest.json` under `env_overrides`.

## 24.4 Required health checks per env

- `ADFLIP`: import torch + torch-scatter/torch-cluster + dry invocation of `models/ADFLIP/test/design.py --help`.
- `spurs`: import torch + `spurs.inference` and minimal hub/load check.
- `bioemu`: `python -m bioemu.sample --help` and package import check.
- `uma-qc`: import `fairchem.core`, import ASE modules, predictor construction smoke test (no long MD).

---

## 25) Key Pseudocode Extracts (from `ThermoGFN-IF.tex`, implementation-facing)

These are the mandatory control-flow templates for implementation. They are concise transcriptions of the paper algorithms and must be treated as canonical.

## 25.1 Method III round loop (teacher-student active learning)

```text
INPUT: D_0, pretrained ADFLIP params, round budget R, BioEmu budget b_B, UMA budget b_U
FOR r in 0..R-1:
  fit/update surrogate ensemble M_r on D_r
  define acquisition reward R_r(x) from surrogate mean + uncertainty + diversity penalties
  train teacher GFlowNet pi_theta,r using on-policy rollouts + off-policy reconstructed trajectories from D_r
  build candidate pool C_r from teacher samples + replay elites + evaluated examples
  distill teacher terminal distribution on C_r into one-shot student q_phi,r
  sample large proposal batch B_r from student (one-shot proposals)
  run validity filters and dense SPURS scoring on B_r
  select diverse subset of size b_B, run BioEmu, compute dynamics features and calibrated scores
  select subset of size b_U from BioEmu-scored set, run UMA using whole/local routing by atom budget
  append all newly labeled tuples to D_{r+1}
END
OUTPUT: student q_phi,R, teacher artifacts, dataset D_R
```

## 25.2 Target-conditioned inference with bounded retries

```text
INPUT: complex/monomer context C, target vector y*, tolerances eps, retry budget R_max
best_x <- None; best_dev <- +inf
FOR retry in 1..R_max:
  sample candidate batch from conditioned model p_theta(x | C, y*)
  run SPURS on all, then BioEmu/UMA by acquisition routing
  for each candidate:
    compute predicted properties y_hat(x) and target deviation d_target(x; y*)
    update best_x if d_target improves
    if all active targets satisfied within tolerance AND structure checks pass:
      return success candidate
  optionally widen sampling temperature for next retry
return best_x with explicit failure flag and unmet target channels
```

## 25.3 Complex/component oracle runner

```text
INPUT: object C, baseline sequence x_base, candidate x, decomposition map D, budgets
score baseline bound object + separated components -> y_base
score candidate bound object + separated components:
  SPURS where admissible
  BioEmu where admissible
  UMA where selected by acquisition
translate to dG_bind, ddG_bind, delta Tm, component floors, target violations
return merged property bundle B(x)
```

## 25.4 Full-context ADFLIP refresh after accepted edit

```text
INPUT: state (B, s_t, X_sc_t, C), accepted edit (j, aa), refresh steps N
apply mutation to get s_{t+1}
warm-start tokens/sidechains from prior state
run ADFLIP full-context refresh for N steps on full object
run full PIPPack pass
compute pack/clash diagnostics and updated pack uncertainty
return refreshed state
```

## 25.5 Two-stage acquisition logic

```text
BioEmu acquisition A_B(x): weighted combination of SPURS score, SPURS uncertainty, novelty,
pack uncertainty, explicit K=2 indicator, mutation-order ratio K/L
Select cluster-balanced top candidates for BioEmu

UMA acquisition A_U(x): weighted combination of BioEmu score, BioEmu uncertainty,
BioEmu-SPURS disagreement, novelty, atom-budget eligibility, higher-order indicator
Select cluster-balanced top candidates for UMA
```

---

## 26) Inference Code Examples (README + oracle adapters + ADFLIP)

These examples are implementation references. They should be mirrored by wrapper scripts under `./scripts`.

## 26.1 ADFLIP direct inference example

```bash
conda run -n ADFLIP python models/ADFLIP/test/design.py \
  --pdb path/to/input.pdb \
  --ckpt results/weights/ADFLIP_ICML_camera_ready.pt \
  --device cuda:0 \
  --method adaptive \
  --steps 32 \
  --threshold 0.9
```

Equivalent dispatch form (required in orchestration):

```bash
python scripts/env/dispatch.py \
  --env-name ADFLIP \
  --cmd "python models/ADFLIP/test/design.py --pdb path/to/input.pdb --method adaptive --steps 32 --threshold 0.9"
```

## 26.2 SPURS single-mutant inference example

```python
from spurs.inference import get_SPURS_from_hub, parse_pdb
import torch

model, cfg = get_SPURS_from_hub()
pdb = parse_pdb("path/to/target.pdb", "target_name", "A", cfg)
model.eval()
with torch.no_grad():
    ddg_matrix = model(pdb, return_logist=True)
```

Dispatch wrapper example:

```bash
python scripts/env/dispatch.py \
  --env-name spurs \
  --cmd "python scripts/prep/oracles/spurs_score_single.py --candidate-path runs/R1/rounds/round_000/data/candidate_pool_round_0.parquet --output-path runs/R1/rounds/round_000/data/candidate_pool_spurs_round_0.parquet"
```

## 26.3 SPURS multi-mutant inference example

```python
from spurs.inference import get_SPURS_multi_from_hub, parse_pdb, parse_pdb_for_mutation
import torch

model, cfg = get_SPURS_multi_from_hub()
pdb = parse_pdb("path/to/target.pdb", "target_name", "A", cfg)
mut_ids, append_tensors = parse_pdb_for_mutation([["V2C","P3T"], ["W1A","V2Y"]])
pdb["mut_ids"] = mut_ids
pdb["append_tensors"] = append_tensors.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
with torch.no_grad():
    ddg_multi = model(pdb)
```

## 26.4 BioEmu sampling example

CLI:

```bash
conda run -n bioemu python -m bioemu.sample \
  --sequence GYDPETGTWG \
  --num_samples 2048 \
  --output_dir runs/R1/rounds/round_000/bioemu_samples/candidate_0001
```

Python:

```python
from bioemu.sample import main as sample

sample(
    sequence="GYDPETGTWG",
    num_samples=2048,
    output_dir="runs/R1/rounds/round_000/bioemu_samples/candidate_0001",
    model_name="bioemu-v1.1",
    filter_samples=True,
    base_seed=1234,
)
```

Dispatch wrapper example:

```bash
python scripts/env/dispatch.py \
  --env-name bioemu \
  --cmd "python scripts/prep/oracles/bioemu_sample_and_features.py --candidate-path runs/R1/rounds/round_000/data/candidate_pool_spurs_round_0.parquet --output-path runs/R1/rounds/round_000/data/bioemu_scored_round_0.parquet --num-samples 2048"
```

## 26.5 UMA/FAIRChem MD example

```python
from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms.calc = calc
dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
dyn.run(steps=1000)
```

Dispatch wrapper example:

```bash
python scripts/env/dispatch.py \
  --env-name uma-qc \
  --cmd "python scripts/prep/oracles/uma_md_screen.py --candidate-path runs/R1/rounds/round_000/data/bioemu_scored_round_0.parquet --output-path runs/R1/rounds/round_000/data/uma_scored_round_0.parquet --model-name uma-s-1p1 --temps 300,330,360,390,420 --replicates 4"
```

## 26.6 Wrapper coding requirement

Each oracle wrapper script must include in the output metadata:
- `source_readme_path`
- `source_inference_module`
- `runtime_env_name`
- `invocation_cmd`
- `model_identifier` (checkpoint or hub model id)

This ensures exact traceability from plan -> implementation -> runtime.
