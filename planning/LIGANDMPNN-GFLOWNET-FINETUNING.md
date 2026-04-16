# Real GFlowNet Fine-Tuning of LigandMPNN Plan

## Goal

Implement the diagrammed loop literally enough for the catalytic training path:
the round-level feedback from UMA-cat / GraphKcat must fine-tune the actual
LigandMPNN sequence generator with a GFlowNet objective, not merely train a
separate teacher and not merely run reward-weighted imitation.

LigandMPNN remains the default generator. ADFLIP remains a future ablation.

## Current Gap

The repository now trains LigandMPNN weights with reward-weighted sequence
likelihood. That is real generator training, but it is not itself a GFlowNet
objective. The existing TB teacher is a separate lightweight edit policy; it
does not update LigandMPNN parameters.

## Target Implementation

Add a LigandMPNN-parameterized trajectory-balance GFlowNet trainer:

```text
D_r labeled/replay dataset
    |
    v
canonical edit trajectories seed -> terminal sequence
    |
    v
LigandMPNN scores each intermediate edit state
    |
    v
TB loss backpropagates through LigandMPNN logits
    |
    v
ligandmpnn_gflownet_round_<r>.pt
    |
    v
candidate pool sampled from the fine-tuned LigandMPNN GFlowNet policy
```

## State, Action, and Trajectory

State:

- a catalytic seed record with reactant/product endpoint metadata;
- current sequence `s_t`;
- edited-position set `E_t`;
- fixed protein structure and ligand atom context from `reactant_complex_path`.

Terminal object:

- a full mutant sequence with mutations relative to the seed sequence.

Canonical trajectory:

- reconstruct mutations by comparing `seed_sequence` and `terminal_sequence`;
- sort edits by sequence position;
- apply one mutation per step;
- terminate with `STOP`.

Actions:

- `EDIT(position, amino_acid)` for allowed mutable positions and replacement
  amino acids;
- `STOP`.

Default mutable set:

- positions from `record.mutations` for labeled replay trajectories;
- for pool sampling, `pocket_random` mutable masks as in the current
  LigandMPNN generator sampler.

## Forward Policy Parameterization

At each intermediate state, run LigandMPNN `ProteinMPNN.score(...)` on the
current sequence and fixed reactant-bound complex context.

For a candidate edit action `(j, a)`:

```text
logit_EDIT(j, a) = log p_LigandMPNN(a at j | current sequence, structure, ligand)
```

For `STOP`, add a small trainable stop head:

```text
logit_STOP = stop_bias + stop_len_weight * K_t
```

The forward action probability is the softmax over:

- STOP;
- all allowed non-noop edit actions at currently mutable/not-yet-edited
  positions.

The TB trajectory log-probability is the sum of the selected action log
probabilities, including STOP.

## Backward Policy

Use a fixed canonical backward policy:

```text
log P_B(trajectory | terminal) = - log(K!)
```

because canonical reverse deletion can be treated as uniformly deleting one of
the `K` applied edits. For sorted canonical forward trajectories this constant
is simple, stable, and differentiability is not required.

## TB Objective

For a terminal candidate `x` with positive reward `R(x)`:

```text
L_TB = (log Z(seed) + log P_F(tau | seed) - log R(x) - log P_B(tau | x))^2
```

where:

- `log Z(seed)` is a trainable scalar table keyed by `seed_id`;
- `log R(x)` uses fused oracle `reward` when present;
- baseline rows without oracle labels get a configurable small reward floor;
- rewards are clipped to avoid pathological early-round gradients.

This directly trains LigandMPNN weights through the forward policy terms.

## Stabilization

Defaults:

- train scope: `decoder`, updating `W_s`, decoder layers, and `W_out`;
- stop head and logZ table are always trained;
- gradient clipping enabled;
- small L2 anchor to the source checkpoint;
- optional supervised NLL auxiliary term with a small weight for round-0
  stability;
- short teacher diagnostics are still allowed, but no longer define generator
  training.

## Sampling

Pool generation continues to use real LigandMPNN autoregressive sampling with a
mutable mask. Because the TB objective trains the same LigandMPNN logits that
drive sampling, the generated pool now comes from a GFlowNet-fine-tuned
LigandMPNN checkpoint.

Candidate records will include:

```json
{
  "generator_backend": "ligandmpnn",
  "generator_training_objective": "trajectory_balance",
  "generator_checkpoint": ".../ligandmpnn_gflownet_round_0.pt"
}
```

## Implementation Tasks

1. Extend `train/thermogfn/ligandmpnn_generator.py` with trajectory helpers:
   seed lookup, mutation reconstruction, action-space construction, TB log-prob
   computation, and checkpoint metadata support.
2. Replace or extend `scripts/train/m3_train_ligandmpnn_generator.py` so the
   default objective is `trajectory_balance`, with reward-weighted NLL retained
   only as an ablation/fallback.
3. Save GFlowNet checkpoint metadata in the LigandMPNN `.pt` file:
   stop-head state, logZ table, objective name, reward config, and source
   checkpoint.
4. Update `scripts/train/m3_generate_ligandmpnn_pool.py` to preserve and report
   `generator_training_objective`.
5. Update `uma_cat_m3_run_round.py` config wiring to pass the TB objective by
   default.
6. Run smoke tests:
   - one real TB step through LigandMPNN;
   - candidate sampling from the GFlowNet checkpoint;
   - side-chain packing of one generated candidate;
   - orchestration dry-run command construction.
7. Update `README.md` to describe actual GFlowNet fine-tuning of LigandMPNN.

## Acceptance Criteria

- The default catalytic config sets
  `generator.ligandmpnn.train.objective: trajectory_balance`.
- A smoke run logs nonzero `tb_loss` and writes
  `ligandmpnn_gflownet_round_0.pt`.
- The generated pool references that GFlowNet checkpoint and objective.
- The round runner calls the real GFlowNet trainer before oracle stages.
- README states that LigandMPNN is directly fine-tuned with TB GFlowNet loss.
