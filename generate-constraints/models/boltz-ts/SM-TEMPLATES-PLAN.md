# Small-Molecule / Transition-State Templates Plan (Boltz-2)

## Context and Goal
- Transition state geometry is in `data/local_maxima_6.xyz` (27 atoms; charge -1, spin 1, has per-atom forces). Enzyme sequence for binding is in `data/cleaned_uniprot_rhea.tsv` line 17983 (id `A0A068Q609`, EC `1.14.14.44;1.14.14.77`, RHEA `52156/52164`).
- Current Boltz-2 supports protein templates only; templates are ignored during affinity prediction. We want to reuse the Boltz-2 checkpoints (structure + affinity) and extend template handling to small molecules/transition states so the TS geometry can steer docking/affinity.

## What the Code Does Today (template path)
- YAML parse (`boltz/data/parse/schema.py`): `templates` entries must map to protein chains; alignments computed from sequences; parsed templates stored as `StructureV2` and `TemplateInfo` with residue ranges, `force`, `threshold`. Non-protein chains are rejected.
- Tokenization (`boltz/data/tokenize/boltz2.py`): proteins tokenized per residue with frames; ligands/non-polymers tokenized per atom (UNK protein token, no frames).
- Featurization (`boltz/data/feature/featurizerv2.py`): builds template features only when `data.templates` is present **and** `compute_affinity` is False. Features assume per-residue protein info: `template_restype`, frames (`template_frame_rot/t`), `template_cb/ca`, masks, and `query_to_template`. Non-protein templates have no path.
- Model usage (`boltz/model/modules/trunkv2.py`): `TemplateModule`/`TemplateV2Module` expect backbone-like features and aggregate templates into pairwise `z`. Potentials (`TemplateReferencePotential`) enforce CB distances for forced templates. Affinity head skips templates entirely.
- I/O: processed templates saved/loaded as `StructureV2` NPZs in `processed/templates/` (see `boltz/main.py`, `boltz/data/module/inferencev2.py`).

## Targeted Outcome
Enable ligand/transition-state templates to steer structure generation and affinity scoring:
- Accept small-molecule templates (SDF/PDB/mmCIF) plus atom-level mappings.
- Convert `.xyz` TS to a Boltz-readable structure (prefer SDF/PDB with explicit hydrogens/charges).
- Build template features for non-polymers (atom-level) and feed them into the template stack + potentials.
- Allow templates during affinity prediction to guide docking and refine binding affinity.

## Data Prep for the Transition-State Template
- Convert `data/local_maxima_6.xyz` → `data/local_maxima_6.sdf` (or `.pdb`) with RDKit:
  - Read xyz, assign bonds (ETKDG + `AllChem.EmbedMolecule` as needed), preserve total charge -1; store spin as metadata if needed for later.
  - Sanitize and keep explicit hydrogens; freeze coordinates as the reference conformer.
  - Optional: compute MMFF/UFF minimization if bond assignment fails; otherwise keep provided coords.
- Map to a ligand chain id (e.g., `T`) in the YAML, ensuring atom ordering in the template matches the ligand SMILES/CCD ordering or providing an explicit atom map.
- Store converted template in `processed/templates/{target_id}_ts.npz` via `StructureV2.dump` when preprocessing.

## YAML / Schema Extensions
- Extend `templates` entries to allow ligand templates and atom maps:
```yaml
version: 1
sequences:
  - protein:
      id: [A]
      sequence: <enzyme_sequence_from_line_17983>
      msa: <path or 0>
  - ligand:
      id: T
      smiles: "<TS SMILES placeholder or generated from SDF>"
templates:
  - sdf: data/local_maxima_6.sdf      # allow sdf/pdb/mmcif in addition to cif
    chain_id: T                      # ligand chain to template
    template_id: T                   # optional; keeps naming consistent
    atom_map: identical              # or explicit list [[ligand_atom_idx, template_atom_idx], ...]
    force: true
    threshold: 1.0                   # Å deviation allowed from template
properties:
  - affinity:
      binder: T
```
- New schema rules: `chain_id` may reference ligand/nonpolymer chains; `atom_map` optional (default: identity); `force` still requires `threshold`. Backward-compatible with protein templates.

## Code Changes (by layer)
1) **Parsing / Schema (`boltz/data/parse/schema.py`)**
   - Allow `templates` with keys `sdf`/`mol2`/`xyz` (after conversion)/`pdb`/`cif`.
   - Permit `chain_id` pointing to non-polymer chains; skip protein-sequence alignment for non-polymers. For ligands, create `TemplateInfo` using atom ranges; support `atom_map` (explicit) or auto-map via RDKit substructure/Isomorphism on element+bond graph.
   - If `xyz` provided, convert to RDKit Mol with coordinates before creating `StructureV2`. Persist converted file into processed area for reuse.

2) **Template Storage (`boltz/main.py`, `boltz/data/module/inferencev2.py`)**
   - When preprocessing, write ligand templates to `processed/templates/{record_id}_{template_name}.npz` using `StructureV2.dump`.
   - Ensure loader accepts ligand templates and attaches them to `Input.templates`.

3) **Tokenization (`boltz/data/tokenize/boltz2.py`)**
   - Keep ligand tokenization per atom. Expose mapping for template atoms (e.g., `template_atom_index`) to preserve atom order for `atom_map`.

4) **Featurization (`boltz/data/feature/featurizerv2.py`)**
   - Generalize `process_template_features`:
     - Detect ligand templates; build atom-level features: element one-hot, distance matrix bins, optional partial charges, masks, and visibility ids.
     - Maintain current protein path (frames/CB/CA). Produce unified keys, e.g., `template_atom_coords`, `template_atom_mask`, `template_atom_element`, `template_type` flag, plus keep existing protein keys for compatibility.
   - Remove `compute_affinity` guard so templates are available to affinity predictions.

5) **Model Template Stack (`boltz/model/modules/trunkv2.py`)**
   - Extend `TemplateModule`/`TemplateV2Module` to branch on `template_type`: protein branch uses backbone frames; ligand branch uses atom features (element embedding + distance/angle encodings) to update `z`.
   - Ensure asym masking respects ligand chains and mixed complexes (protein-ligand pairs).

6) **Potentials (`boltz/model/potentials/potentials.py`)**
   - Add ligand template potential using `template_atom_coords`/mask with `force`/`threshold`; allow per-atom thresholds if provided.
   - Keep CB-based potential for proteins; route `template_force` masks to the correct template type.

7) **Affinity Head (`boltz/model/modules/affinity.py` and caller)**
   - Include template-conditioned `z` and any template-based potentials during affinity runs; ensure batching stays consistent.

8) **CLI / Config (`boltz/main.py`, docs)**
   - Add docs for ligand templates, new YAML fields, and the TS workflow. Add optional flag to convert `xyz` → `sdf` during preprocessing.

## Checkpoint Usage
- Reuse existing Boltz-2 structure (`boltz2_conf.ckpt`) and affinity (`boltz2_aff.ckpt`) checkpoints. Template branch changes should be non-breaking (feature default = dummy template as today).
- If ligand template conditioning materially changes representations, add a small finetune path gated behind a flag, but default inference should run without retraining.

## Validation Plan
- Unit/IO: schema parsing with ligand templates; atom_map resolution; xyz→sdf conversion routine.
- Featurizer: mixed protein+ligand template batches produce matching masks/shapes; dummy template fallback unchanged.
- Model: forward pass with ligand template (force on/off) covers structure + affinity; template potentials apply only to templated atoms.
- End-to-end: YAML example above runs through `boltz predict --use_templates` generating structures/affinity with TS template present; compare RMSD to template and check affinity stability across seeds.

## Special Considerations for Small Molecules / TS
- Preserve charge (-1) and multiplicity metadata; store as attributes for downstream scoring even if the network ignores spin.
- Atom limit: ensure TS atom count (27) stays within ligand atom cap (128; recommended ≤56 for affinity). Keep hydrogens consistent with Boltz’s RDKit sanitization.
- If TS SMILES is unavailable/unfaithful, use SDF-derived connectivity; set YAML `smiles` to canonicalized version of the converted mol to keep Boltz preprocessing deterministic.
- For forced templates, consider tighter thresholds (0.5–1.0 Å) and possibly atom-specific thresholds for labile atoms (e.g., protons).



Create repo and push

  - From /home/iska/amelie-ai/mora-stage-2:

    # optional: rename the working dir to match the new repo name
    mv models/boltz models/boltz-ts
    cd models/boltz-ts

    # keep current status; save work if needed
    git status -sb

    # create private repo with GitHub CLI (keeps current branch and pushes)
    gh repo create amelie-iska/boltz-ts --private --source . --remote origin --push
    If you prefer creating on the web UI, create a private repo amelie-iska/boltz-ts with no initializer, then set/push:

    git remote set-url origin git@github.com:amelie-iska/boltz-ts.git  # or https URL
    git push -u origin main  # adjust branch name if not main

  Verify install still works

  - Clone and install locally to confirm:

    cd /tmp
    git clone git@github.com:amelie-iska/boltz-ts.git
    cd boltz-ts
    pip install -e .[cuda]
    Since pyproject.toml still declares name = "boltz" and the package lives in src/boltz, the import path and CLI entry point remain unchanged.

  Optional cleanups

  - Update README clone examples to git clone git@github.com:amelie-iska/boltz-ts.git and pip install -e .[cuda].
  - If you keep the folder name as boltz, you can skip the mv; just run the gh repo create/remote set-url steps from the existing directory.