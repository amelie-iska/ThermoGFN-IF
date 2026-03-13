# UMA Quality Plan

## Objective

Upgrade the current UMA-based catalytic dynamics stack so that:

- sMD trajectories remain structurally stable across the RF3 reactant/product dataset,
- the whole protein, not only the pocket, follows realistic restrained dynamics,
- pocket motion is flexible but not explosive,
- the ligand path is chemically plausible,
- unsupported multi-fragment systems use better CVs than naive endpoint morphing,
- PMFs are only computed on validated CVs and physically coherent path classes,
- the protocol scales across the dataset with a high-quality validation profile and a cheaper training profile derived from it.

This document is a remediation plan. It does not claim the present UMA sMD/PMF implementation is already high quality.

## Current Failure Modes

Observed from the exported trajectories and runtime diagnostics:

- The current pulls are too aggressive in wall-clock MD time; they are effectively short, forced deformations rather than high-quality restrained dynamics.
- The whole protein can distort, shear, or lose local secondary structure, especially outside the pocket.
- Distal regions that should move collectively instead behave either like a frozen scaffold or an underconstrained elastic blob.
- Domain/domain and secondary-structure relationships are not being preserved strongly enough during the pull.
- Pocket motion is either underconstrained or overconstrained.
- Unsupported systems with large graph edits still look like endpoint morphs, not realistic bound-state transitions.
- Long nonphysical bonds and stretched fragments appear in the ligand path.
- First-shell waters and hydrogenation are present but not yet integrated into a truly stable protocol.
- PMF eligibility logic is now better, but the actual CV methodology for unsupported systems is still not sufficiently physical.

## Core Diagnosis

The main problem is not only parameter choice. It is method choice and system scope.

The current stack still mixes three different regimes under one steering design:

1. small, chemically coherent reactive-center edits,
2. multi-fragment association / dissociation or large ligand graph edits,
3. mostly conformational endpoint changes where chemistry should not be represented by direct bond morphing.

These three regimes need different CVs and different uses of sMD/PMF.

Separately, the protocol still treats whole-protein stability as a side effect of local restraints. That is the wrong framing. Whole-system realism must be an explicit objective with explicit controls and explicit quality gates.

## Key Constraints from UMA / FAIRChem

From the current `fairchem` usage pattern:

- Use `FAIRChemCalculator` with the `omol` task for molecular and protein-ligand systems.
- Use conservative ASE Langevin MD settings.
- Prefer explicit minimization / relaxation before production.
- Treat MD stability as an empirical protocol-design problem.

Practical implications:

- The timestep must remain conservative.
- The thermostat should be gentle and physically interpretable.
- High-quality trajectories will require much longer pulls than the current short image schedules.
- For bond-making / bond-breaking, the correct object to steer is a chemistry-aware CV, not a Cartesian interpolation of endpoint coordinates.
- Whole-protein structural preservation must be enforced through soft physical priors and staged equilibration, not by hoping pocket restraints indirectly stabilize the rest of the fold.

## Whole-System Realism Requirements

The entire enzyme-ligand-water assembly must satisfy the following during sMD and PMF.

### 1. Whole-protein fold preservation

The protocol must preserve:

- local peptide geometry,
- secondary-structure continuity,
- domain-level relative arrangement,
- chain topology and compactness,
- realistic mobility gradients from active site to distal scaffold.

This means the protein cannot be modeled as "pocket + a few anchors". The full catalytic chain must be part of the restraint and validation design.

### 2. Realistic global motion

The protein should exhibit:

- smooth domain and loop motion,
- limited whole-body drift and rotation after alignment,
- no slab-like rigid freezing of the whole fold,
- no large-scale shear unrelated to the endpoint change.

### 3. Realistic ligand and fragment motion

The ligand path must avoid:

- stretched artificial bonds,
- component teleportation,
- impossible fragment overlap,
- pseudo-reactive bond morphing in systems that are not chemically representable that way.

### 4. Realistic water behavior

Waters should:

- remain chemically plausible near the active site,
- not drift into severe clashes,
- not be overconstrained into unrealistic geometries,
- provide optional relay / coordination support where justified.

### 5. Numerically stable MD

The protocol must produce:

- bounded energy drift,
- smooth work profiles,
- no frequent force spikes or clipped-force domination,
- trajectories that are reproducible across replicas up to expected stochastic variability.

## Required Protocol Split

The dataset should be routed into three protocol classes.

### Class A: Reactive-center systems

Criteria:

- small number of bond edits,
- modest reactive atom count,
- shared ligand mapping is trustworthy,
- chemistry can be represented by a local reactive-center CV set.

Method:

- use explicit reactive CVs,
- allow PMF,
- use sMD only as a path-seeding and near-TS harvesting method.

### Class B: Unsupported multi-fragment / large graph-edit systems

Criteria:

- many bond edits,
- large reactive fraction,
- fragment-level rearrangement dominates,
- Cartesian ligand morphing is not chemically credible.

Method:

- no direct bond-morph CV,
- no PMF by default,
- use conformational / assembly CVs:
  - fragment COM distances,
  - fragment orientation alignment,
  - fragment-pocket contact CVs,
  - coordination-number CVs,
  - catalytic-water / metal coordination CVs where relevant.

### Class C: Endpoint-conformational systems

Criteria:

- reactant/product differ mostly by bound pose and surrounding pocket state,
- chemistry is not representable from shared endpoint graph edits.

Method:

- treat these as bound-state transition paths, not chemical barriers,
- use pocket and protein path CVs plus fragment support CVs,
- PMF only if a separate chemically meaningful CV set is later defined.

## Preparation Protocol

Every endpoint pair should go through the same preparation pipeline.

### 1. Protonation and hydrogenation

- Standardize protonation at a fixed workflow pH, default `7.4`.
- Add hydrogens before any relaxation or steering.
- Record protonation assumptions in metadata.

### 2. First-shell waters

- Retain or place first-shell waters around polar ligand atoms, catalytic residues, and likely relay positions.
- Limit to a bounded number for stability and reproducibility.
- Reject waters that clash or are not supported by local geometry.

### 3. Endpoint relaxation

- After H/water preparation, run local UMA relaxation before any sMD.
- Use staged relaxation:
  - `FIRE` or `LBFGS` minimization,
  - short low-temperature Langevin,
  - restrained equilibration with the full-protein prior,
  - then equilibration at the target production temperature.

### 4. Endpoint equilibration

- Each endpoint should get a short restrained equilibrium MD segment before steering.
- Save the equilibrated endpoint coordinates used to initialize sMD and PMF.

### 5. Whole-system preflight checks

Before steering, each prepared endpoint should pass:

- no chain breaks in the selected catalytic chain,
- no severe backbone geometry outliers,
- no ligand-protein close-contact explosion,
- no large water clashes,
- bounded relaxation displacement outside the pocket.

## Protein Stability Model

The protein should neither be frozen nor allowed to unravel. This section is the primary blocker for current trajectory quality.

### Global structural prior

Use a soft, full-protein structural prior:

- interpolated `CA` elastic network over shared residues,
- stronger sequential / local chain restraints,
- weaker nonlocal tertiary restraints.

This prior should span the whole catalytic chain, not only residues near the pocket.

### Local pocket guidance

Use stronger local guidance on:

- pocket backbone atoms,
- catalytic side-chain anchor atoms,
- conserved binding-site heavy atoms.

### Distal anchors

Use only sparse distal anchors as a final stabilizer, not as the primary fold-preservation mechanism.

### Additional geometry prior

Add explicit backbone geometry restraints where needed:

- peptide-local distances,
- optional backbone angle / pseudo-dihedral restraints for unstable regions.

### Domain-level coherence prior

For multi-domain proteins or large folds, add an intermediate-scale structural prior:

- domain or subdomain centroid restraints,
- low-weight inter-domain orientation restraints,
- optional segment-wise elastic networks for domains that are known to move semi-rigidly.

The goal is to preserve realistic domain motion rather than either freezing or smearing the whole protein.

### Staged restraint release

Do not start full steering from an unrelaxed, fully free system.

Use a staged schedule:

1. strong global structural prior during endpoint minimization,
2. moderate structural prior during endpoint equilibration,
3. full steering with a still-active but weaker whole-protein prior,
4. PMF windows with the same prior retained consistently.

This should reduce both fold blow-up and the current overrigid behavior.

## Better CV Methodology

This is the main required upgrade.

### Reactive-center CVs for Class A

Primary CVs:

- forming bond coordination numbers,
- breaking bond coordination numbers,
- donor-acceptor distances for proton transfer surrogates,
- catalytic residue / ligand / water contact distances,
- optional path CVs `s` and `z` built from a relaxed reference string.

Do not steer only on ligand Cartesian coordinates.

### Fragment-aware CVs for Class B

For unsupported multi-fragment systems, use:

- fragment COM-to-pocket distances,
- fragment orientation alignment to endpoint reference poses,
- fragment-fragment separation / contact CVs,
- pocket-support distances from representative ligand atoms to stable pocket anchors,
- coordination numbers for key contacts instead of explicit bond interpolation,
- solvent coordination CVs where water bridges matter,
- metal-ligand coordination numbers where relevant.

These systems should be treated as assembly / reorganization paths unless a specific reactive-center model is available.

### Conformational endpoint CVs for Class C

For endpoint-conformational systems, use:

- global shared-backbone path progress,
- pocket RMSD / local path progress,
- ligand component pose progress,
- fragment-to-pocket support CVs,
- path distance `z` to prevent excursions away from the reference manifold.

### Path CV construction

For all classes, stop using pure endpoint coordinate morphing as the main progress definition.

Instead:

1. build a coarse reference path in CV space,
2. relax images under UMA with the structural prior,
3. reparameterize the path,
4. use path progress `s` and path distance `z` as monitoring variables,
5. use class-specific CV terms as the physical steering terms.

### Whole-protein path terms

In addition to ligand/pocket CVs, every protocol class should include whole-protein progress terms:

- global backbone path progress for the catalytic chain,
- local pocket path progress,
- optional domain-path progress for large proteins.

The whole protein must therefore have its own path representation, not just a static background restraint.

## sMD Protocol

### General principles

- lower aggressiveness substantially,
- steer more slowly,
- increase image count and actual MD per image,
- insert relaxation before each image advance,
- use weaker springs plus longer dynamics instead of stronger springs plus short forced motion.
- ensure the whole system remains close to a physically plausible manifold at all times, not only the ligand and pocket.

### Required staged schedule

Each sMD run should contain:

1. endpoint preparation,
2. endpoint minimization,
3. endpoint restrained equilibration,
4. image-to-image restrained relaxation,
5. short production MD segment at each image,
6. optional reverse pull with the same protocol.

This is slower, but the current direct short-pull workflow is what is producing broken trajectories.

### High-quality validation profile

Target defaults:

- temperature: `300 K`
- timestep: `0.1 fs`
- Langevin friction: `1 ps^-1`
- endpoint equilibration: `1-5 ps`
- sMD frames/images: `96`
- MD per image: `500-2500` steps depending on class and system size
- per-image minimization / relaxation before production steps
- multiple forward and reverse replicas
- explicit whole-protein structural prior active throughout the run
- no PMF unless the CV class is validated

For large proteins, the protocol should err toward:

- more images,
- smaller per-image target changes,
- stronger whole-protein prior during equilibration,
- weaker instantaneous steering forces,
- more total MD rather than more aggressive springs.

The current sub-picosecond total pull lengths are not acceptable for publication-quality endpoint-conditioned paths.

### Training profile

Derived from the validated protocol:

- keep the same CV design,
- reduce replicas,
- reduce MD per image,
- keep image count high enough to avoid coarse jumps,
- keep PMF off except on scheduled rounds or selected candidates.

The training profile must preserve the same whole-system controls as the validation profile. It may be cheaper, but it must not revert to a pocket-only stabilization model.

## PMF Protocol

PMF should only run where the CVs are physically meaningful.

### Eligibility

Run PMF only for:

- reactive-center systems with validated local chemistry CVs,
- optionally a smaller subset of assembly CV systems after separate validation.

### Window definition

- place windows in CV space, not just on naive ligand lambda,
- seed from the improved sMD path,
- preserve the same structural prior used in sMD,
- record overlap diagnostics explicitly.

For whole-system realism, PMF windows must also preserve:

- the full-protein structural prior,
- domain coherence terms where needed,
- pocket-support terms for unsupported systems if PMF is ever allowed there.

### PMF quality gates

- sufficient overlap between neighboring windows,
- no catastrophic fold drift,
- no exploding pocket contacts,
- stable CV histograms,
- MBAR / WHAM diagnostics recorded per run.

## Dataset-Level Validation

The protocol should not be promoted based on one or two trajectories.

### Validation panel

Build a representative panel covering:

- short / medium / long proteins,
- reactive-center systems,
- unsupported multi-fragment systems,
- conformational endpoint systems,
- water-rich and water-poor pockets,
- metal-free and metal-containing systems if present.

The validation panel must also intentionally include:

- small single-domain enzymes,
- larger multi-domain enzymes,
- flexible-loop active sites,
- buried active sites,
- proteins with long distal regions that are currently blowing up.

### Required metrics

For each trajectory:

- full-backbone RMSD,
- per-domain backbone RMSD where domains can be identified,
- backbone local-geometry deviation,
- secondary-structure preservation proxy,
- whole-protein radius of gyration / compactness drift,
- domain-centroid displacement and orientation drift,
- pocket RMSD,
- elastic-network deviation,
- maximum close contacts / clashes,
- maximum excess-bond count,
- forward/reverse hysteresis,
- work profile smoothness,
- path-distance `z`,
- protein COM and rotational drift after alignment,
- fraction of frames with unrealistic backbone distortion,
- water-retention and water-clash diagnostics,
- PMF overlap diagnostics where applicable.

Metrics should be summarized both:

- for the whole trajectory,
- and for the worst frame.

### Acceptance criteria

Promote defaults only if the validation panel shows:

- stable fold preservation,
- stable whole-protein geometry and compactness,
- realistic domain and loop motion,
- realistic pocket motion,
- no systematic long-bond artifacts,
- lower hysteresis,
- path classes consistent with the intended CV design,
- PMF only on systems that remain well-behaved under the validated CV set.

Explicit failure conditions:

- whole-protein RMSD explosion unrelated to endpoint difference,
- loss of secondary structure in multiple regions,
- repeated large chain-geometry violations,
- repeated severe long-bond artifacts,
- trajectories dominated by force clipping or violent work spikes.

## Implementation Phases

### Phase 1: Fix the methodology split

Files:

- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)
- [scripts/prep/oracles/uma_catalytic_score.py](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/uma_catalytic_score.py)

Tasks:

- finalize class routing,
- make class-specific CV bundles first-class,
- remove any remaining implicit dependence on endpoint coordinate morphing as the main CV.

### Phase 2: Stabilize protein dynamics

Files:

- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

Tasks:

- strengthen the fold-preserving structural prior,
- add local geometry priors for unstable backbone segments,
- add domain-level coherence restraints,
- add staged restraint release and endpoint equilibration,
- standardize endpoint relaxation and equilibration.

### Phase 3: Upgrade unsupported-system CVs

Files:

- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

Tasks:

- add fragment COM/orientation CVs,
- add coordination-number CVs for fragment-pocket interactions,
- add whole-protein path terms and domain progress terms to those protocols,
- add path-distance `z`,
- stop treating unsupported systems as pseudo-reactive bond morphs.

### Phase 4: Rebuild PMF around validated CVs

Files:

- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

Tasks:

- define window placement in CV space,
- add overlap diagnostics,
- preserve the full-protein structural prior inside PMF windows,
- add eligibility gates tied to class and quality metrics.

### Phase 5: Dataset validation and promotion

Files:

- [scripts/prep/oracles/validate_uma_smd_protocol.py](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/validate_uma_smd_protocol.py)
- [scripts/prep/oracles/profile_uma_protocol_dataset.py](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/profile_uma_protocol_dataset.py)

Tasks:

- run validation panel,
- summarize failure modes by class,
- define the final training-profile defaults from the validated profile.

## Immediate Next Steps

In order:

1. Replace the remaining ligand-lambda-centric steering with class-specific CV bundles.
2. Add a whole-protein structural prior that preserves secondary structure, domain coherence, and chain geometry.
3. Add explicit fragment orientation and coordination-number CVs for unsupported systems.
4. Increase total pull time by at least an order of magnitude for the validation profile.
5. Add endpoint equilibration and path-image relaxation as required stages, not optional extras.
6. Re-run validation on a representative RF3 panel before changing defaults again.

## What Success Looks Like

We should expect:

- supported systems to produce smooth, near-transition, low-hysteresis paths,
- unsupported systems to produce credible conformational / assembly trajectories rather than broken pseudo-chemistry,
- the whole protein to remain folded and dynamically plausible throughout the trajectory,
- loops and domains to move realistically rather than either freezing or blowing up,
- PMFs only where the CVs and path class justify them,
- no obvious pocket explosions, no spurious long bonds, and much better whole-system fold preservation across the RF3 dataset.
