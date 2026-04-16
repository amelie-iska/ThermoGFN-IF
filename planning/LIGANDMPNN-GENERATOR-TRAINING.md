# Real LigandMPNN Sequence-Generator Training Plan

Status: superseded by `planning/LIGANDMPNN-GFLOWNET-FINETUNING.md`. This file
records the first intermediate implementation pass, where LigandMPNN was trained
with reward-weighted supervised likelihood. The current default implementation
uses trajectory-balance GFlowNet fine-tuning and writes
`ligandmpnn_gflownet_round_<r>.pt`; keep this document only as historical
context for the fallback `reward_weighted_nll` objective.

## Problem

The catalytic Method III loop currently trains a lightweight trajectory-balance teacher and a one-shot student sampler over edit statistics. LigandMPNN is only used for side-chain packing. That is not actual training of the sequence generator. The default generator for this repo is now LigandMPNN; ADFLIP remains an ablation option for later.

The fix is to make each catalytic round update a real LigandMPNN sequence-model checkpoint and generate the next candidate pool from that checkpoint.

## Design Principles

- Train the actual LigandMPNN `ProteinMPNN` sequence model, not a proxy sampler.
- Keep LigandMPNN as the default generator backend and reserve ADFLIP for a future ablation path.
- Use real endpoint complexes as model context. For catalysis, the default structural context is the reactant-bound enzyme-ligand complex.
- Keep the existing oracle stack unchanged: LigandMPNN generator proposes sequences, LigandMPNN side-chain packer materializes endpoint structures, GraphKcat optionally prefilters, UMA-cat runs real broad dynamics plus sMD/PMF, and fused labels are appended to the round dataset.
- Make training stable on the small Catalyst-GT bootstrap set by using reward-weighted supervised likelihood with configurable anchoring and trainable-scope control.
- Save ordinary LigandMPNN-compatible `.pt` checkpoints so later scripts can load the tuned generator without special wrappers.

## Data Flow Per Round

1. Load `D_r` from JSONL.
2. Fit the existing surrogate and TB teacher for diagnostics/acquisition continuity.
3. Fine-tune the real LigandMPNN sequence checkpoint on records in `D_r`.
   - Input structure: `reactant_complex_path` by default.
   - Target sequence: `record.sequence`.
   - Mutable residues: `protein_chain_id` chain.
   - Loss: weighted negative log-likelihood from `ProteinMPNN.score(...)`.
   - Record weights: positive oracle reward when available, otherwise a configurable baseline weight.
   - Optional anchor regularization: L2 distance from the starting checkpoint.
4. Save `models/ligandmpnn_generator_round_<r>.pt`.
5. Generate `candidate_pool_round_<r>.jsonl` by autoregressive sampling from the tuned LigandMPNN checkpoint with ligand atom context enabled.
6. Preserve all catalytic metadata from seed records.
7. Pack generated candidates onto reactant/product endpoints with the LigandMPNN side-chain packer.
8. Run GraphKcat and UMA-cat real oracles.
9. Fuse oracle labels and append to `D_{r+1}`.

## Training Objective

For record `x_i`, LigandMPNN provides `log p_theta(s_i | structure_i, ligand_i, order_i)`.

The per-record sequence loss is:

```text
L_i = - mean_{j in design chain} log p_theta(s_i[j] | structure_i, ligand_i)
```

The round loss is:

```text
L = mean_i w_i L_i + lambda_anchor ||theta - theta_0||_2^2
```

where:

- `w_i` is derived from `reward` if present and positive;
- baseline records without oracle reward receive `baseline_weight`;
- weights are clipped and normalized by their batch mean;
- `theta_0` is the source checkpoint for the current round, normally the previous round checkpoint or the base LigandMPNN checkpoint.

Default trainable scope is `decoder`, which updates `W_s`, decoder layers, and `W_out`. This gives real generator training while limiting drift on small catalytic datasets. `train_scope: full` remains available for larger datasets.

## Sampling

Pool generation uses `ProteinMPNN.sample(...)` from the tuned checkpoint:

- `chain_mask` selects the protein chain;
- ligand atom context is enabled by default;
- the sampled full chain replaces the seed chain sequence;
- mutations and `K` are computed against the seed sequence;
- duplicate full sequences are removed by deterministic `candidate_id`.

The candidate source remains schema-compatible as `source: student`, with explicit metadata:

```json
{
  "generator_backend": "ligandmpnn",
  "generator_checkpoint": ".../ligandmpnn_generator_round_0.pt",
  "source_model": "LigandMPNN"
}
```

## Implementation Tasks

1. Add shared LigandMPNN generator helpers under `train/thermogfn/ligandmpnn_generator.py`.
2. Add `scripts/train/m3_train_ligandmpnn_generator.py`.
3. Add `scripts/train/m3_generate_ligandmpnn_pool.py`.
4. Extend `uma_cat_m3_run_round.py` so `generator.backend: ligandmpnn` trains and samples the real LigandMPNN generator.
5. Add config keys under `generator.ligandmpnn.train` and `generator.ligandmpnn.sample`.
6. Add smoke tests that run one real optimization step and one real sampling pass on a Catalyst-GT record.
7. Update `README.md` so it no longer describes LigandMPNN as only a packer in the default training loop.

## Acceptance Criteria

- A catalytic round writes `models/ligandmpnn_generator_round_<r>.pt`.
- The candidate pool records contain `generator_backend=ligandmpnn`.
- At least one smoke test performs a real backward/optimizer step through `ProteinMPNN.score(...)`.
- At least one smoke test samples candidates from the tuned checkpoint with ligand context.
- Existing UMA-cat and GraphKcat oracle stages consume the generated pool without schema changes.
- README instructions make clear that LigandMPNN is the trained default sequence generator and ADFLIP is future ablation work.
