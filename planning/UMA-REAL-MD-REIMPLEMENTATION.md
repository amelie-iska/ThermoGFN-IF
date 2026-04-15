# Real UMA MD Removal and Reimplementation Plan

## Status

This plan replaces any catalytic UMA implementation that fabricates dynamics, morphs ligand chemistry by Cartesian interpolation, or assigns reaction-barrier rewards without a real FAIRChem/ASE molecular dynamics calculation. The default trainable generator for the current repository is LigandMPNN. ADFLIP is an ablation backend to revisit after the LigandMPNN path is stable and validated.

## 1. Removal Plan

### 1.1 Remove pseudo-reactive ligand path construction

Do not use any of the following as production catalytic labels:

- direct internal ligand coordinate morphs between reactant and product endpoints,
- rigid fallback paths for incomplete ligand atom mappings,
- bond-breaking or bond-forming spring schedules,
- PMFs seeded from topology-changing ligand morphs,
- log-rate proxies computed from diagnostic or unsupported sMD paths.

The runtime must treat topology-changing reactant/product pairs as unsupported for sMD/PMF barrier labels unless a separate chemistry-aware CV implementation is added and validated. The broad productive-pose UMA screen may still run because it is an actual equilibrium MD screen in the reactant basin.

### 1.2 Remove silent fallbacks

Unsupported states must be explicit:

- incomplete ligand mapping -> `unsupported_reactive_path`,
- ligand graph changes between endpoints -> `unsupported_reactive_path`,
- failed UMA calculator setup -> hard error,
- unavailable UMA model weights -> hard error,
- failed sMD quality gates -> non-OK catalytic status,
- PMF protocol ineligible -> PMF skipped with a recorded reason.

There should be no code path that silently replaces a failed or unsupported UMA calculation with a neutral, cached, random, heuristic, or pseudo-scored label.

### 1.3 Remove invalid reward labels

The fused catalytic reward must only treat UMA-cat as present when:

- the FAIRChem UMA calculator ran successfully,
- the broad screen succeeded,
- the sMD or PMF barrier came from a supported topology-preserving endpoint protocol,
- quality gates passed.

If the pathway is unsupported, the row may retain broad-screen telemetry, but `uma_cat_status` must not be `ok`, `uma_cat_barrier_source` must be `none`, and `uma_cat_log10_rate_proxy` must be a sentinel invalid value that cannot be mistaken for a usable kinetic label.

## 2. Reimplementation Plan

### 2.1 UMA engine contract

The only supported production UMA calculator path is:

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")
```

Supported model names are `uma-s-1p2`, `uma-s-1p1`, and `uma-m-1p1`. The default for this checkout is `uma-s-1p2` because that is the cached FAIR Chemistry checkpoint under `/home/iska/.cache/fairchem/models--facebook--UMA`. The task name for protein-ligand molecular systems is `omol`. The code should fail fast for unsupported model names rather than guessing.

The runtime must import the installed `fairchem` package from the active environment. Do not shadow it with the checked-in `./models/fairchem/src` tree for production runs unless that tree is upgraded to the same FairChem API version as the installed package. The cached `uma-s-1p2` checkpoint uses a newer checkpoint schema than the local source snapshot.

### 2.2 Physical preparation

For each packed LigandMPNN candidate:

1. Load reactant-bound and product-bound endpoint structures.
2. Add hydrogens at the configured pH where OpenMM tooling is available.
3. Add only explicitly marked first-shell waters for the current lightweight protocol.
4. Relax prepared endpoints with UMA/FIRE before any dynamics.
5. Record atom count, water count, preparation settings, model name, task name, and device in artifacts.

Full explicit solvent and ions are the long-term target for final scientific runs, but the current repository protocol must be honest about using a prepared endpoint plus limited first-shell waters.

### 2.3 Broad productive-pose screen

The broad screen remains valid because it is a real UMA Langevin trajectory in the reactant-bound basin. It should:

- use FAIRChem `omol` forces,
- run replicated ASE Langevin MD,
- save productive-pose frame telemetry,
- estimate `p_gnac`, gating free energy, uncertainty, visit counts, dwell time, and first-hit frame,
- mark severe clashes and instability as penalties or quality failures.

### 2.4 Supported sMD/PMF class

sMD and PMF are supported only for fully mapped, topology-preserving ligand endpoints:

- all ligand heavy atoms in the reactant endpoint map to product atoms,
- inferred reactant and product ligand bond graphs match,
- the endpoint change is a pose/conformation transition, not a chemical graph transition.

For this class, steering may use conservative endpoint pose restraints plus whole-protein/pocket elastic restraints. UMA remains the physical force field. Biases are recorded separately from the physical UMA energy.

### 2.5 Unsupported topology-changing class

When reactant/product endpoints imply ligand bond formation, bond breaking, multi-fragment association, dissociation, proton transfer, metal coordination changes, or incomplete atom mapping:

- do not run sMD as a reaction barrier calculation,
- do not run path umbrella PMF,
- do not compute `log10 k_proxy`,
- keep broad-screen telemetry,
- set `uma_cat_status = unsupported_reactive_path`,
- leave a precise protocol reason in artifacts.

Future support for these cases requires a separate validated CV layer, such as reaction-coordinate definitions from chemistry templates, QM/MM calibration, NEB/path refinement on a chemically valid state graph, or dataset-specific reaction families.

### 2.6 Quality gates

Every accepted sMD/PMF must report and gate on:

- endpoint/product RMSD,
- pocket RMSD,
- backbone RMSD,
- interpolated CA network deviation,
- close-contact counts,
- excess ligand bond counts,
- forward/reverse hysteresis when reverse pulls are enabled,
- finite energies and finite forces throughout the trajectory.

Any gate failure changes the status away from `ok` and prevents the row from contributing a positive UMA-cat reward.

### 2.7 Testing requirements

The unit suite must enforce:

- no topology-changing bond spring schedules are created,
- incomplete mappings do not create fallback paths,
- topology-changing endpoints do not create guided sMD paths,
- invalid sMD protocols produce `barrier_source = none`,
- invalid protocols do not receive a usable log-rate proxy,
- valid topology-preserving protocols still use physical log-rate uncertainty propagation.

A real UMA smoke test should be run in the `fairchem` environment before claiming a production trajectory:

```bash
conda run -n fairchem python scripts/prep/oracles/run_uma_unbiased_md.py \
  --structure-path path/to/reactant_complex.pdb \
  --protein-chain-id A \
  --ligand-chain-id B \
  --pocket-positions 10,25,40 \
  --output-pdb runs/tmp/uma_real_md_smoke.pdb \
  --model-name uma-s-1p2 \
  --device cuda:0 \
  --production-steps 100 \
  --record-every 10 \
  --temperature-k 300
```

The smoke output is acceptable only if energies and coordinates are finite, no severe clashes are introduced, and the generated multimodel PDB shows continuous MD frames rather than minimized endpoint jumps.

Current local smoke status:

- `uma-s-1p2` resolves from `/home/iska/.cache/fairchem/models--facebook--UMA/snapshots/be2896459a03fcde05e20d2fcefd11f450601fce/checkpoints/uma-s-1p2.pt` through the installed `fairchem` package.
- A direct water `omol` probe produced finite energy and forces and completed three Langevin steps.
- `scripts/prep/oracles/run_uma_unbiased_md.py` completed a three-step chignolin smoke with `smoke_quality_pass = true`, four written frames, finite energies, finite forces, and finite positions.
- This confirms runtime wiring only. Production acceptance still requires a representative protein-ligand panel with hydrogens, waters, quality gates, and trajectory inspection.
