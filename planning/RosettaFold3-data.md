# RosettaFold3 ReactZyme Integration Plan

## Goal

Set up a reproducible RosettaFold3 workflow under this repository that:

1. uses the local Foundry RF3 implementation in `models/foundry`,
2. uses the shared MMSeqs2-GPU installation under `../enzyme-quiver/MMseqs2`,
3. builds RF3-ready JSON inputs from the ReactZyme-derived multi-fragment ligand data under `generate-constraints_0`,
4. supports Boltz-style pocket residue constraints with a distance threshold during RF3 inference,
5. documents the full setup and repair flow in the repository `README.md`, including the already-repaired catalysis oracle environments.

## Scope

This work is split into five concrete deliverables:

1. RF3 environment and runtime wrappers inside this repo.
2. ReactZyme-to-RF3 input preparation scripts.
3. RF3 inference extension for pocket constraints.
4. MMSeqs2-GPU-backed MSA preparation and RF3 run wrappers.
5. Documentation updates in the top-level `README.md`.

## Constraints And Assumptions

- The writable workspace is limited to this repository, so environment creation outside the repo, global checkpoint installation, and mutation of sibling repos must be driven by scripts and docs rather than executed directly here.
- Network access is restricted in this session, so checkpoint downloads and new package installs cannot be completed end-to-end here.
- The shared MMSeqs2-GPU stack already exists in `../enzyme-quiver/MMseqs2` and should be reused rather than duplicated.
- The most reliable ReactZyme source in this repo is `generate-constraints_0`, not `generate-constraints`.
- RF3 already supports:
  - protein sequences plus optional `msa_path`,
  - ligand inputs from SDF/CIF paths,
  - whole-ligand templating via `ground_truth_conformer_selection`,
  - token-level templating via `template_selection`.
- RF3 does not currently expose interface or pocket constraints at inference time; that is the main model-side extension required here.

## Data Findings That Drive The Design

### Foundry / RF3

- RF3 JSON inputs are parsed in `models/foundry/models/rf3/src/rf3/utils/inference.py`.
- The inference path already converts JSON `components` into an `AtomArray` plus `chain_info`.
- RF3 already consumes token-pair template features through:
  - `has_distogram_condition`
  - `distogram_condition_noise_scale`
  - `distogram_condition`
- The recycler always invokes `RF3TemplateEmbedder`, but inference disables meaningful conditioning by setting `allowed_chain_types_for_conditioning=None`.
- RF3 also has a pair-bias mechanism in the pairformer attention stack via `Beta_II`, but it is not currently threaded from inference inputs.

### ReactZyme / SDF Templates

- The relevant train manifest is `generate-constraints_0/output_sdf_templates/train/manifest.csv`.
- Multi-fragment ligands are common:
  - 2 fragments: 6517 rows
  - 3 fragments: 2229 rows
  - higher-fragment ligands also exist
- Usable rows for the strict initial workflow, requiring:
  - `status == reactant:ok|product:ok`
  - protein sequence present
  - protein length `<= 600`
  - pocket annotations present in `generate-constraints_0/pocket_cache`
  - reactant and product SDFs present
  gives `198` rows.
- The pocket cache format is UniProt feature JSON. We can reuse the same extraction rule already used by the older Boltz YAML script:
  - include `Binding site`, `binding_site`, `binding`, `Active site`
  - expand ranges when present

### MSA / MMSeqs2-GPU

- `../enzyme-quiver/scripts/prepare_rf3_local_inputs.py` already contains working local-MSA logic using `boltz.data.msa.mmseqs2.run_mmseqs2`.
- `../enzyme-quiver/MMseqs2/local_msa` is already wired to the fast local storage path and should be treated as the shared source of truth.
- The ThermoGFN implementation should follow the same pattern:
  - generate or reuse `.a3m`,
  - write `msa_path` into RF3 JSON,
  - run Foundry RF3 against those prepared JSONs.

## Implementation Plan

## Phase 1: Add RF3 Runtime Scaffolding

### Deliverables

- `scripts/env/create_foundry_rf3_env.sh`
- `scripts/env/check_foundry_rf3_env.sh`
- `scripts/rf3/run_foundry_rf3_local_msa.sh`

### Work

- Create a repo-local RF3 environment setup script that:
  - prefers `uv`,
  - creates a dedicated environment for Foundry RF3,
  - installs `models/foundry` and RF3 in editable mode,
  - records the expected checkpoint location,
  - documents the checkpoint install path and local cache layout.
- Add a check script that validates:
  - `python -c "import foundry, rf3"`,
  - presence of the selected RF3 checkpoint,
  - visibility of the shared MMSeqs2 local MSA workspace.
- Add a wrapper for running Foundry RF3 with:
  - explicit `ckpt_path`,
  - `FOUNDRY_CHECKPOINT_DIRS`,
  - `PYTHONPATH` pointing at the local Foundry tree when needed,
  - output directory and input JSON list handling.

### Notes

- Actual package installation and checkpoint download may still need to be run manually on the machine because of sandbox and network restrictions.

## Phase 2: Build ReactZyme-to-RF3 Input Generation

### Deliverables

- `scripts/rf3/build_reactzyme_rf3_inputs.py`
- a machine-readable manifest for produced RF3 jobs

### Work

- Read:
  - `generate-constraints_0/output_sdf_templates/train/manifest.csv`
  - `generate-constraints_0/data/reactzyme_data_split/cleaned_uniprot_rhea.tsv`
  - `generate-constraints_0/data/reactzyme_data_split/rhea_molecules.tsv`
  - `generate-constraints_0/pocket_cache/*.json`
- Filter to a high-confidence initial set:
  - require sequence,
  - require sequence length `<= 600`,
  - require both reactant and product SDFs,
  - default to `reactant:ok|product:ok`,
  - require at least one pocket position.
- For each accepted row, emit two RF3 JSON examples:
  - reactant
  - product
- Each JSON example should include the equivalents of the Boltz YAML fields:
  - protein sequence as chain `A`
  - ligand template SDF from `generate-constraints_0/output_sdf_templates/train/...`
  - ligand chemistry metadata from the original multi-fragment SMILES string
  - pocket constraint record with binder chain and residue contacts
  - metadata block describing source rows and filtering provenance
- Emit:
  - per-example JSONs or sharded JSON lists,
  - a summary JSON,
  - a CSV/JSONL manifest with one row per reactant/product RF3 job.

### Representation choice

- Initial RF3 ligand representation will use the template SDF path as the actual ligand component because:
  - the SDF already captures the intended 3D multi-fragment geometry,
  - Foundry RF3 natively supports ligand inputs from `path`,
  - whole-ligand templating is available through `ground_truth_conformer_selection`.
- The original dotted SMILES string will still be stored in metadata for traceability.

## Phase 3: Add Pocket Constraint Support To RF3

### Deliverables

- RF3 JSON schema extension for pocket constraints
- inference-time feature generation for those constraints
- model-side consumption of the new constraint features

### Work

- Extend `InferenceInput` parsing in `models/foundry/models/rf3/src/rf3/utils/inference.py` to accept a new top-level `constraints` block compatible with the Boltz-style structure:

```json
{
  "constraints": [
    {
      "pocket": {
        "binder": "R",
        "contacts": [["A", 465]],
        "max_distance": 8.0,
        "force": true
      }
    }
  ]
}
```

- Normalize that input into an internal `pocket_constraints` representation carried through `InferenceInput.to_pipeline_input()`.
- Add a small inference-only transform utility that converts pocket constraints into token-pair conditioning features.

### Conditioning strategy

Use a deterministic, inference-only token-pair bias path rather than trying to pretend a threshold-only pocket constraint is a full ground-truth distogram.

Specifically:

- map binder chain tokens and target protein residue tokens,
- build a symmetric token-pair bias matrix `pocket_pair_bias`,
- scale the bias by `max_distance` and `force`,
- thread that bias into:
  - the pairformer single update attention,
  - the diffusion conditioning / diffusion transformer path where practical.

### Why this path

- The existing RF3 template conditioning path expects distogram-like supervision, not threshold-only contacts.
- A pair bias directly matches the semantics of "encourage proximity between this ligand and these residues".
- This is the smallest robust extension that leaves existing RF3 behavior unchanged when no pocket constraints are provided.

### Compatibility requirements

- If `constraints` is absent, RF3 behavior must remain byte-for-byte compatible.
- If constraint parsing fails for one example, the error should clearly identify the example name and offending constraint.

## Phase 4: Local MMSeqs2-GPU MSA Preparation

### Deliverables

- `scripts/rf3/prepare_reactzyme_rf3_local_msas.py`
- `scripts/rf3/run_reactzyme_rf3_local_msa.sh`

### Work

- Adapt the proven logic from `../enzyme-quiver/scripts/prepare_rf3_local_inputs.py`.
- For each unique protein sequence:
  - generate or reuse a local `.a3m`,
  - use the shared Boltz client source in the MMSeqs2 local workspace,
  - cache MSAs under a ThermoGFN-owned output directory,
  - populate `msa_path` in the generated RF3 JSONs.
- The runtime wrapper should:
  - validate MMSeqs2 local server reachability,
  - prepare MSAs if missing,
  - launch Foundry RF3 over the prepared JSON shards.

### Default path assumptions

- MMSeqs2 root: `../enzyme-quiver/MMseqs2/local_msa`
- local server URL: `http://127.0.0.1:8080`
- output root: `runs/rf3_reactzyme/...`

## Phase 5: Documentation

### Deliverables

- update the repository `README.md`

### Work

- Add a new RF3 / Foundry section covering:
  - repo-local environment creation,
  - checkpoint install options,
  - MMSeqs2-GPU local server expectations,
  - ReactZyme RF3 input preparation,
  - RF3 execution,
  - pocket constraint support and JSON schema.
- Update the Kcat environment section so it accurately reflects the repaired state:
  - preferred solver defaults,
  - `repair_kcat_envs.sh` behavior,
  - the GraphKcat `libLLVM-15.so` preload workaround,
  - health-check commands that now pass.

## Verification Plan

## Code-Level Verification

- `python -m py_compile` or equivalent syntax checks on all new Python files.
- `bash -n` on all new shell scripts.
- unit or smoke tests for:
  - pocket cache extraction,
  - ReactZyme row filtering,
  - RF3 JSON emission,
  - pocket constraint parsing and token mapping.

## Data Verification

- run the ReactZyme builder in a dry-run mode and verify:
  - row counts,
  - filtered row counts,
  - reactant/product example counts,
  - protein length cap enforcement,
  - SDF path existence.

## RF3 Verification

- smoke test RF3 JSON parsing on at least one generated reactant example and one product example.
- verify pocket constraint tensors are created and shape-compatible.
- verify inference still runs for an example with no `constraints`.

## Manual Runtime Verification On The Machine

- create or repair the Foundry RF3 environment,
- confirm checkpoint resolution,
- generate local MSAs using the shared MMSeqs2-GPU service,
- run one or two RF3 examples end-to-end,
- confirm outputs land in the expected run directory.

## Risks And Mitigations

- Risk: RF3 attention bias threading may require deeper plumbing than expected.
  - Mitigation: keep the extension inference-only and local to `InferenceInput`, feature preparation, and a narrow model hook.
- Risk: some ReactZyme SDFs may not be suitable for RF3 ingestion.
  - Mitigation: keep strict default filters and emit explicit skip reasons.
- Risk: MMSeqs2 local server availability is external to this repo.
  - Mitigation: wrapper scripts should fail early with a clear health-check message.
- Risk: the usable high-confidence subset is smaller than the full dataset.
  - Mitigation: expose flags to widen filters later, but start with the clean subset of `198` rows.

## Order Of Execution

1. Add the plan document.
2. Implement ReactZyme RF3 JSON preparation and summary reporting.
3. Implement local MSA preparation and RF3 run wrappers.
4. Add RF3 pocket constraint parsing and model plumbing.
5. Update the top-level `README.md`.
6. Run local syntax and smoke verification.
