# UMA sMD / PMF Rebuild Plan

## Status

The current UMA-based sMD / PMF stack is not acceptable for production, publication, or training-time catalytic labels.

The failure is not a single bad hyperparameter. It is a methodological failure:

- the system preparation is not physically adequate for whole-protein dynamics,
- the current "trajectory" generation is not truly continuous MD,
- the steering is defined mostly in Cartesian endpoint space rather than in physically meaningful CV space,
- unsupported chemistry is still being forced through a path construction that cannot represent it,
- PMFs are being contemplated from path classes that are not yet legitimate reaction coordinates.

This document replaces the previous ad hoc plan. It is a from-scratch rebuild plan for using UMA correctly and for obtaining high-quality, realistic, generalizable steered trajectories and PMFs across the RF3 enzyme dataset.

## Executive Diagnosis

### What UMA is

UMA is an ML interatomic potential exposed through `FAIRChemCalculator` in ASE. For our enzyme-ligand complexes, the current code uses the `omol` task:

- [models/fairchem/README.md](/home/ubuntu/amelie/ThermoGFN/models/fairchem/README.md)
- [models/fairchem/docs/uma_tutorials/uma_tutorial.md](/home/ubuntu/amelie/ThermoGFN/models/fairchem/docs/uma_tutorials/uma_tutorial.md)
- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

At the force level, the correct base dynamics are:

\[
F_i^{\mathrm{UMA}}(R) = - \nabla_{r_i} E_{\mathrm{UMA}}(R; \theta, q, s),
\]

where \(R\) is the full coordinate set, \(q\) and \(s\) denote charge / spin metadata where relevant, and \(\theta\) are the learned model parameters.

If we bias the system, the total potential should be:

\[
U_{\mathrm{tot}}(R, t) = E_{\mathrm{UMA}}(R) + U_{\mathrm{bias}}(R, t) + U_{\mathrm{struct}}(R),
\]

and the propagated forces should be:

\[
F_i^{\mathrm{tot}}(R,t) = -\nabla_{r_i} E_{\mathrm{UMA}}(R) - \nabla_{r_i} U_{\mathrm{bias}}(R,t) - \nabla_{r_i} U_{\mathrm{struct}}(R).
\]

This part is conceptually fine. The problem is the form of \(U_{\mathrm{bias}}\), the form of \(U_{\mathrm{struct}}\), and the physical environment in which we are applying them.

### What we are doing wrong now

The current runtime in [uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py) has several fundamental problems.

#### 1. We are not running a realistic whole-protein solvent environment

Right now the system is effectively a dry protein-ligand complex with:

- added hydrogens,
- at most a tiny hand-built first-shell water set,
- no full solvent bath,
- no ion atmosphere,
- no realistic long-timescale hydration support for fold stability.

For a whole enzyme, that is not a realistic MD environment. A large protein in near-vacuum or in a tiny local water shell will distort even if the MLIP itself is reasonable.

This is a primary cause of the observed whole-protein failure.

#### 2. The exported "trajectory" is not truly MD

In [run_steered_uma_dynamics(...)](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py), each image currently does:

1. advance the steering protocol,
2. run `FIRE`,
3. run a tiny number of Langevin steps.

This means the exported sequence is not a smooth finite-temperature trajectory. It is a sequence of locally re-minimized snapshots with a few MD steps inserted between them.

That is useful for path seeding, but it is not a realistic dynamical trajectory.

#### 3. The path is still fundamentally endpoint-Cartesian

For both the "supported" and "unsupported" classes, the current path generation is still dominated by endpoint coordinate interpolation:

- [build_guided_ligand_path_targets(...)](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)
- [build_ligand_restraint_model(...)](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)
- [build_component_pocket_support_model(...)](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

That is the root cause of:

- stretched bonds,
- fake product states that still look like translated reactants,
- unnatural fragment motion,
- pocket tearing,
- protein shearing.

#### 4. The "supported" class is still not genuinely reactive

Even when a pair is classified as `reactive_center`, the coordinate path is still largely a rigid/component pose path, with scheduled bond-length restraints layered on top.

That is not a chemistry-aware transition path. It is a Cartesian morph plus extra springs.

#### 5. The "unsupported" class is still being overinterpreted

For multi-fragment / large graph-edit systems, the current support CVs are still only a stabilization aid. They are not a sufficient physical description of:

- association,
- dissociation,
- fragment exchange,
- proton relay,
- metal coordination changes,
- water-mediated chemistry.

These systems need a fundamentally different CV design.

#### 6. The whole protein is only weakly and generically stabilized

The current structural prior is an improvement over nothing, but it is still too generic:

- CA elastic network,
- local backbone geometry,
- sparse anchors,
- pocket/global Cartesian guidance.

This is not enough to preserve realistic domain motion, secondary-structure persistence, and full-chain coherence while the active site is being driven.

#### 7. The time scales are far too short

The current defaults are still in the regime of:

- tiny endpoint equilibration,
- tiny per-image MD,
- total pulled path lengths on the order of tens to low hundreds of femtoseconds.

That is not enough for realistic loop, pocket, or domain adaptation in proteins.

## How UMA Should Be Used Here

The FAIRChem materials show three consistent patterns:

1. relax with a real optimizer first (`LBFGS`, `BFGS`, `FIRE`),
2. do actual MD with ASE integrators when running dynamics,
3. use NEB / DyNEB or other path methods to find a minimum-energy path rather than forcing a path by raw coordinate morphing.

Relevant references in-tree:

- [models/fairchem/docs/uma_tutorials/uma_tutorial.md](/home/ubuntu/amelie/ThermoGFN/models/fairchem/docs/uma_tutorials/uma_tutorial.md)
- [models/fairchem/docs/uma_tutorials/uma_catalysis_tutorial.md](/home/ubuntu/amelie/ThermoGFN/models/fairchem/docs/uma_tutorials/uma_catalysis_tutorial.md)
- [models/fairchem/README.md](/home/ubuntu/amelie/ThermoGFN/models/fairchem/README.md)

For our use case, UMA should be the base force field that generates:

- energies,
- forces,
- relaxed endpoint structures,
- restrained MD trajectories,
- NEB / path-refined initial guesses,
- umbrella-sampled windows.

The bias should live in low-dimensional CV space, not directly in all Cartesian ligand atoms except where absolutely necessary.

## Ground-Up Rebuild

## Phase 0. Stop Using the Current Exported Paths as Quality Targets

Immediate policy:

- Do not use the current multi-`MODEL` PDB exports as examples of "good" trajectories.
- Do not use current sMD outputs for production PMFs.
- Do not use the current pseudo-transition exports as training labels for "reactive path quality."

At best, the current code can be used as:

- a source of infrastructure,
- a source of preparation utilities,
- a source of diagnostics,
- a source of initial endpoint classification heuristics.

## Phase 1. Rebuild the Physical System Definition

### 1.1 Choose the correct system envelope

We need two supported operating modes.

#### Mode A: Whole-protein solvated dynamics

Use when the goal is realistic whole-protein gating and bound-state motion.

System contents:

- catalytic chain / chains,
- ligand or ligand fragments,
- crystallographic or prepared cofactors,
- explicit whole-protein solvent shell or box,
- neutralizing / background ions if supported by the representation.

This is the preferred mode for realistic conformational sMD.

#### Mode B: Active-site reaction subsystem embedded in a restrained protein

Use when the goal is higher-quality local chemistry in a tractable system.

System contents:

- full protein coordinates retained,
- explicit active-site region and nearby residues fully flexible,
- farther protein regions restrained by a strong but soft structural prior,
- explicit local water environment,
- optional cluster-style chemistry refinement for the reactive region.

This is the preferred mode for PMF and near-TS harvesting if full whole-protein PMF is too noisy or too expensive.

### 1.2 Solvation must become explicit and systematic

Current first-shell-only water placement is not enough.

Minimum acceptable upgrade:

- explicit protein solvation shell around the full catalytic protein, not only around the ligand,
- retained and validated first-shell waters at the active site,
- clash removal and local relaxation,
- metadata logging of all inserted waters.

Preferred target:

- whole-protein explicit solvent environment suitable for finite-temperature MD.

If a full solvent box is not feasible under current UMA throughput constraints, use:

- a large explicit solvent droplet or shell,
- a boundary restraint,
- stronger whole-protein structural priors,
- and clearly document that the method is shell-solvated, not fully periodic.

### 1.3 Protonation and chemical state

Before any MD:

- standardize protonation,
- document pH model,
- identify catalytic residues whose protonation state is uncertain,
- create alternate protonation-state branches for ambiguous active sites,
- keep endpoint chemistry internally consistent.

If proton transfer is part of the reaction, protonation-state ambiguity must be represented explicitly in protocol selection.

## Phase 2. Endpoint Preparation and Equilibration

Each endpoint must go through a real preparation workflow.

### 2.1 Endpoint minimization

Use UMA with ASE optimization, not image-wise steering, to relax each prepared endpoint:

\[
R^\star = \arg\min_R \left[E_{\mathrm{UMA}}(R) + U_{\mathrm{struct,prep}}(R)\right].
\]

Recommended tools:

- `LBFGS` or `BFGS` for endpoint relaxation,
- `FIRE` only as a rescue or preconditioner, not as the main dynamical engine.

### 2.2 Restrained endpoint equilibration

After minimization, run actual restrained Langevin MD:

\[
m_i \ddot r_i = F_i^{\mathrm{UMA}} - \gamma m_i \dot r_i + \eta_i(t) + F_i^{\mathrm{struct}}.
\]

This stage should be long enough to:

- settle solvent,
- settle hydrogens,
- relax local strain,
- equilibrate the pocket around the endpoint.

### 2.3 Time step and thermostat

Starting quality defaults:

- timestep: `0.1 fs` as the conservative baseline,
- temperature: `300 K`,
- friction: approximately `1 ps^{-1}` baseline,
- optional lower-temperature pre-equilibration for delicate systems before warming to target temperature.

Current `0.05 fs` can still be used for especially unstable systems, but it should not hide the fact that we need much longer physical trajectories.

### 2.4 Equilibration duration

The current tens-of-femtoseconds regime is not acceptable.

At minimum:

- endpoint restrained equilibration in the low-ps regime,
- protocol tuning on a validation panel to determine the shortest stable length,
- production-quality examples in the tens-of-ps to hundreds-of-ps regime for local path scouting.

For whole-protein conformational adaptation, the plan must assume longer time scales than current settings.

## Phase 3. Rebuild the Structural Prior for the Whole Protein

The whole protein must be stabilized explicitly.

### 3.1 Structural prior decomposition

Use:

\[
U_{\mathrm{struct}} = U_{\mathrm{local}} + U_{\mathrm{secondary}} + U_{\mathrm{domain}} + U_{\mathrm{global}} + U_{\mathrm{anchor}}.
\]

Where:

- \(U_{\mathrm{local}}\): peptide-local geometry restraints,
- \(U_{\mathrm{secondary}}\): soft helix / strand persistence terms,
- \(U_{\mathrm{domain}}\): domain coherence terms,
- \(U_{\mathrm{global}}\): weak whole-protein shape / contact-map preservation,
- \(U_{\mathrm{anchor}}\): very sparse distal anchors only as a last stabilizer.

### 3.2 Local geometry

Keep:

- peptide bond lengths,
- local backbone distances,
- optional pseudo-dihedral stabilization for unstable segments.

### 3.3 Secondary-structure preservation

The current CA network is not enough.

Add a secondary-structure aware term based on:

- local hydrogen-bond proxy distances,
- local backbone pseudo-dihedrals,
- segment-wise elastic templates.

These should be soft enough to allow real motion but strong enough to prevent helix/strand dissolution from steering noise.

### 3.4 Domain coherence

Introduce explicit domain or segment quasi-rigid coherence:

- partition the protein into domains / subdomains / long secondary-structure blocks,
- restrain centroid distances and coarse orientations,
- allow relative motion between domains consistent with endpoint change.

This is necessary because current "global backbone targets" are too crude.

### 3.5 Active-site flexibility

The pocket must be more flexible than the rest of the fold.

Therefore:

- strongest global restraints live away from the pocket,
- moderate local restraints live on pocket backbone,
- catalytic side chains retain the greatest freedom,
- ligand-contact residues can adapt but not explode.

## Phase 4. Replace the Current Steering with Proper CV-Based Biasing

This is the center of the rebuild.

### 4.1 General form

Steering should act on a reduced CV vector:

\[
\boldsymbol{\xi}(R) = \left(\xi_1(R), \xi_2(R), \dots, \xi_d(R)\right),
\]

with bias:

\[
U_{\mathrm{bias}}(R,t) = \frac{1}{2}\sum_{\alpha=1}^{d} k_\alpha \left[\xi_\alpha(R) - \xi_\alpha^\star(t)\right]^2.
\]

Not:

- raw Cartesian interpolation of the ligand,
- raw Cartesian interpolation of the whole pocket,
- direct atom-by-atom dragging of chemically changing fragments.

### 4.2 Protocol classes

The dataset must continue to be split, but with richer CV bundles.

#### Class A: Supported reactive-center systems

Use chemistry-aware CVs:

##### Coordination-number CVs

For bond forming / breaking:

\[
c_{ij}(r_{ij}) = \frac{1 - (r_{ij}/r_0)^n}{1 - (r_{ij}/r_0)^m},
\]

with \(m > n\), typically large enough to make a smooth switching function.

Build:

\[
\xi_{\mathrm{chem}}(R) = \sum_{(i,j)\in\mathcal{F}} w_{ij}^{\mathrm{form}} c_{ij}(r_{ij})
- \sum_{(i,j)\in\mathcal{B}} w_{ij}^{\mathrm{break}} c_{ij}(r_{ij}).
\]

##### Proton-transfer CVs

When relevant:

\[
\xi_{\mathrm{PT}}(R) = r_{\mathrm{donor-H}} - r_{\mathrm{acceptor-H}}.
\]

##### Geometry CVs

When relevant:

- attack angle,
- nucleophile-acceptor distance,
- catalytic-water coordination number,
- metal-ligand coordination number.

##### Path-CV formulation

For multi-CV path refinement, use path CVs \(s,z\) over a set of reference images \(\{\xi_i\}\):

\[
s(R) = \frac{\sum_{i=1}^{N} \frac{i-1}{N-1}\exp\left[-\lambda\|\xi(R)-\xi_i\|^2\right]}
{\sum_{i=1}^{N}\exp\left[-\lambda\|\xi(R)-\xi_i\|^2\right]},
\]

\[
z(R) = -\lambda^{-1}\ln\left(\sum_{i=1}^{N}\exp\left[-\lambda\|\xi(R)-\xi_i\|^2\right]\right).
\]

This should replace current Cartesian path forcing.

#### Class B: Unsupported multi-fragment / large graph-edit systems

Do not pretend these are simple bond-edit reactions.

Use conformational / assembly CVs:

- fragment-fragment COM distances,
- fragment orientation quaternions or principal-axis alignment,
- fragment-pocket contact coordination numbers,
- water-bridge occupancy CVs,
- metal / cofactor coordination numbers,
- pocket opening / loop closure distances,
- ligand burial and solvent exposure metrics.

These systems can still use sMD, but only as:

- bound-state transition sampling,
- assembly / dissociation path scouting,
- conformational gating diagnostics.

No chemical PMF by default.

#### Class C: Endpoint-conformational systems

Use:

- pocket RMS subspace CVs,
- domain-distance CVs,
- fragment support CVs,
- path CVs in these low-dimensional variables.

These are gating trajectories, not reactive barriers.

## Phase 5. Separate Path Scouting from Realistic Trajectory Generation

We need two distinct products:

1. path-initialization objects,
2. realistic finite-temperature trajectories.

### 5.1 Path scouting

Use:

- slow, CV-based sMD,
- restrained minimization only for endpoint preparation or initial image generation,
- optional NEB / string refinement for supported systems.

### 5.2 Realistic trajectory generation

Once a path exists in CV space, run actual restrained MD under a gently moving target:

\[
\xi^\star(t) = \xi_0 + v t,
\]

or equivalently a piecewise-smooth schedule, but without repeated per-image minimization.

If a snapshot sequence includes repeated `FIRE` minimizations, it must be labeled as a path-relaxed image sequence, not a dynamical trajectory.

## Phase 6. Introduce Proper Path Refinement

The FAIRChem catalysis tutorials point toward NEB / DyNEB for minimum-energy paths.

### 6.1 Supported systems

For genuinely reactive-center systems:

- build a CV-informed initial string,
- refine it with either:
  - NEB / DyNEB in a reduced or restrained coordinate set,
  - or a finite-temperature string / restrained path refinement.

The output should be:

- a smooth path,
- a set of window centers for umbrella sampling,
- near-TS candidate regions.

### 6.2 Unsupported systems

Use a path-string in assembly / conformational CV space, not a chemical NEB.

## Phase 7. PMF Rebuild

### 7.1 PMF is only meaningful on validated CVs

For a chosen CV \(\xi\),

\[
W(\xi) = -k_B T \ln P(\xi) + C.
\]

For multidimensional PMFs,

\[
W(\xi_1,\xi_2,\dots) = -k_B T \ln P(\xi_1,\xi_2,\dots) + C.
\]

We should reconstruct with MBAR or WHAM from umbrella windows:

\[
U_j(R) = E_{\mathrm{UMA}}(R) + U_{\mathrm{struct}}(R) + \frac{k_j}{2}\left[\xi(R)-\xi_j^\star\right]^2.
\]

### 7.2 PMF eligibility

PMF is permitted only if:

- CVs are chemically interpretable,
- steering remained stable,
- whole-protein structure stayed realistic,
- window overlap is acceptable,
- no extreme force clipping occurred,
- forward/reverse path scouting is qualitatively consistent.

### 7.3 PMF diagnostics

Must log:

- overlap matrix,
- effective sample size,
- block PMF stability,
- barrier variation across seed families,
- hysteresis between forward- and reverse-seeded runs,
- restraint work decomposition.

## Phase 8. Recommended MD / sMD Quality Profiles

### 8.1 Quality-first validation profile

Use for method validation and representative examples.

- full endpoint relaxation,
- long restrained endpoint equilibration,
- no per-image minimization inside exported trajectories,
- slower steering velocity,
- more replicas,
- more windows for PMF,
- more diagnostics.

### 8.2 Training profile

Use only after the validation profile is working.

Derived from the validated protocol by reducing:

- number of replicas,
- equilibration length,
- PMF frequency,
- and maybe system size,

while preserving the same CV definitions and the same physical preparation logic.

Training settings must be a downsampled version of a validated physical protocol, not an independently tuned cheap protocol.

## Phase 9. Validation Across the RF3 Dataset

### 9.1 Build a validation panel

Create a stratified panel across:

- sequence length,
- pocket size,
- ligand fragment count,
- metal / inorganic content,
- protocol class,
- supported vs unsupported chemistry,
- expected PMF eligibility.

### 9.2 Per-run acceptance metrics

Track at least:

- whole-backbone RMSD,
- non-pocket backbone RMSD,
- pocket RMSD,
- protein radius of gyration drift,
- domain-centroid drift,
- secondary-structure retention proxy,
- ligand internal geometry deviation,
- number of close contacts,
- number of extreme bond outliers,
- fraction of time under force clipping,
- work-profile smoothness,
- endpoint recovery quality.

### 9.3 Hard rejection criteria

Reject any trajectory or PMF seed with:

- obvious fold damage,
- persistent unrealistic bond lengths,
- severe ligand fragmentation artifact,
- exploding waters,
- domain shearing,
- unstable temperature / energy behavior,
- or bad CV monotonicity.

## Phase 10. Implementation Roadmap

## Milestone 1. Freeze the current production path

- keep existing code for reference only,
- prevent current pseudo-trajectory exports from being treated as final.

## Milestone 2. Rebuild preparation

Files:

- [train/thermogfn/uma_cat_runtime.py](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/uma_cat_runtime.py)

Tasks:

- explicit whole-system solvation workflow,
- rigorous protonation metadata,
- endpoint relaxation pipeline with `LBFGS` / `BFGS`,
- real restrained endpoint equilibration.

## Milestone 3. Rebuild structural prior

Tasks:

- add secondary-structure-aware prior,
- add domain-level coherence prior,
- separate distal-anchor role from true fold stabilization,
- add explicit validation metrics for global fold realism.

## Milestone 4. Replace Cartesian steering

Tasks:

- implement reactive-center coordination-number CVs,
- implement proton-transfer CVs where needed,
- implement fragment assembly CVs for unsupported systems,
- implement path-CV `s,z`,
- move biasing to CV space.

## Milestone 5. Separate path construction from trajectory generation

Tasks:

- path scouting with sMD / string / NEB,
- realistic restrained MD without per-image minimization for exportable trajectories.

## Milestone 6. PMF rebuild

Tasks:

- umbrella window generation from validated path,
- MBAR / WHAM diagnostics,
- block convergence,
- multi-seed agreement,
- PMF gating.

## Milestone 7. Dataset validation

Tasks:

- run a protocol panel,
- collect pass/fail statistics,
- derive training-profile defaults only after physical validation.

## Where the Previous Attempts Went Wrong

This needs to be explicit.

1. I treated endpoint-conditioned coordinate interpolation as if it could stand in for a realistic transition path.
2. I allowed repeated `FIRE` minimization inside the exported "trajectory," which made the output not genuinely dynamical.
3. I attempted to fix chemistry by layering more springs on top of a bad Cartesian path.
4. I underestimated how much the absence of real whole-system solvation would destabilize the protein.
5. I treated whole-protein stability as a local pocket-restraint problem instead of a full-system dynamical problem.
6. I did not separate supported reactive chemistry strongly enough from unsupported large graph-edit endpoint pairs.
7. I allowed PMF logic to advance further than the physical validity of the CVs justified.

Those errors are methodological, not cosmetic. The rebuild must correct them at the level of system definition, CVs, steering, and validation.

## Immediate Next Step

The first implementation step should be:

1. redesign the preparation / equilibration workflow around a realistic whole-system environment,
2. remove per-image minimization from anything exported as a "trajectory",
3. replace current ligand/pocket Cartesian steering with CV-space steering for at least one supported and one unsupported validation case,
4. validate against a small RF3 panel before touching PMF again.

Until that is done, trajectory aesthetics will continue to be poor because the current method is structurally incapable of producing the quality target.
