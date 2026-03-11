# ThermoGFN-IF Catalytic-Oracle Update Plan

## Goal

Update [`planning/ThermoGFN-IF.tex`](/home/ubuntu/amelie/ThermoGFN/planning/ThermoGFN-IF.tex) so the paper treats catalytic-activity and enzyme-kinetic oracles with the same rigor and prominence currently given to SPURS, BioEmu, and UMA. The revised manuscript should explicitly cover:

- catalytic oracles from [`models/KcatNet`](/home/ubuntu/amelie/ThermoGFN/models/KcatNet)
- catalytic oracles from [`models/GraphKcat`](/home/ubuntu/amelie/ThermoGFN/models/GraphKcat)
- catalytic oracles from [`models/MMKcat`](/home/ubuntu/amelie/ThermoGFN/models/MMKcat)
- GFlowNet-style RL / active-learning fine-tuning driven by catalytic oracle feedback
- theory, notation, reward design, acquisition logic, pseudo-code, implementation details, failure modes, validation, and hyperparameters

## Review Findings

### Current manuscript gaps

The current manuscript is strong on:

- thermostability via SPURS, BioEmu, UMA
- binding-aware extensions for complexes
- GFlowNet-style edit policies and one-shot active learning

But it is missing a comparable treatment of enzyme kinetics and catalysis:

- there is no dedicated technical review of KcatNet, GraphKcat, or MMKcat
- there is no formal notation for catalytic targets such as $k_{\rm cat}$, $K_M$, or $k_{\rm cat}/K_M$
- the reward section has no catalytic-oracle fusion equations
- the Method III section does not describe the repo’s kcat-specific active-learning loop
- there is no data/metadata section for reaction-conditioned design inputs such as substrate SMILES, products, pH, temperature, or organism labels
- there are no evaluation or failure-mode sections specific to kinetic-parameter prediction

### Structural issues already present in the manuscript

These are not the user’s primary request, but they affect how the update should be done:

- There is a duplicated UMA section:
  - `\section{UMA-MD protocol as the high-fidelity oracle}`
  - `\section{UMA-MD protocol for a defensible stability oracle}`
- There is no explicit `Method II` section header even though the paper uses `\methodII` and includes a Method II algorithm.

The duplicate UMA section can be repurposed into the new catalytic-oracle protocol section. That is the cleanest way to add a major new section without making the manuscript even more structurally redundant.

## Repo-Grounded Catalytic Oracle Map

### KcatNet

Primary local entrypoints:

- [`scripts/prep/oracles/kcatnet_score.py`](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/kcatnet_score.py)
- [`models/KcatNet/models/model_kcat.py`](/home/ubuntu/amelie/ThermoGFN/models/KcatNet/models/model_kcat.py)
- [`models/KcatNet/config_KcatNet.json`](/home/ubuntu/amelie/ThermoGFN/models/KcatNet/config_KcatNet.json)

Observed interface:

- inputs: protein sequence + substrate SMILES
- output: `kcatnet_log10`, `kcatnet_kcat`, `kcatnet_std`
- model components:
  - ProtT5 embedding
  - ESM embedding
  - substrate graph / molecular features
  - protein graph message passing
  - interaction graph layers
  - final scalar regression head for $\log_{10} k_{\rm cat}$

Paper role:

- wide, cheap-to-middle-cost catalytic oracle
- sequence/substrate-conditioned turnover oracle
- good default for dense or semi-dense catalytic screening

### GraphKcat

Primary local entrypoints:

- [`scripts/prep/oracles/graphkcat_score.py`](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/graphkcat_score.py)
- [`models/GraphKcat/predict.py`](/home/ubuntu/amelie/ThermoGFN/models/GraphKcat/predict.py)
- [`models/GraphKcat/config/TrainConfig_kcat_enz.json`](/home/ubuntu/amelie/ThermoGFN/models/GraphKcat/config/TrainConfig_kcat_enz.json)

Observed interface:

- inputs:
  - protein structure (`protein_path` or `cif_path`)
  - ligand SDF or substrate SMILES
  - optional organism, pH, temperature
- preprocessing:
  - pocket extraction around ligand at fixed 8 A cutoff
  - UniMol ligand embedding
  - ESM2 protein embedding
  - graph dataset construction on pocket / enzyme / ligand context
- outputs:
  - `graphkcat_log_kcat`
  - `graphkcat_log_km`
  - `graphkcat_log_kcat_km`
  - `graphkcat_std`

Paper role:

- structure- and pocket-aware catalytic oracle
- narrower and more expensive than KcatNet
- primary structural refinement oracle for kinetics

### MMKcat

Primary local entrypoints:

- [`scripts/prep/oracles/mmkcat_score.py`](/home/ubuntu/amelie/ThermoGFN/scripts/prep/oracles/mmkcat_score.py)
- [`models/MMKcat/model/basic_model_mm.py`](/home/ubuntu/amelie/ThermoGFN/models/MMKcat/model/basic_model_mm.py)

Observed interface:

- inputs:
  - protein sequence
  - substrate SMILES
  - optional product SMILES
  - enzyme structure graph built from ESMFold + `pdb2graph`
- outputs:
  - `mmkcat_log10`
  - `mmkcat_std`
  - `mmkcat_mask_predictions`
  - `mmkcat_num_masks`
- local wrapper averages multiple mask configurations such as `1111;1110;1101;1100`

Paper role:

- missing-modality catalytic oracle
- robust fallback when products or structure channels are partially unavailable
- explicit uncertainty source from mask disagreement

### Fusion and acquisition already present in repo

Primary local files:

- [`train/thermogfn/kcat_reward.py`](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/kcat_reward.py)
- [`train/thermogfn/acquisition.py`](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/acquisition.py)
- [`config/kcat_m3_default.yaml`](/home/ubuntu/amelie/ThermoGFN/config/kcat_m3_default.yaml)

Current implemented behavior:

- risk-adjusted mean minus kappa times std
- fused score currently uses KcatNet or MMKcat fallback plus GraphKcat and an agreement term
- Kcat-only Method III round orchestration is already implemented in:
  - [`scripts/orchestration/kcat_m3_run_round.py`](/home/ubuntu/amelie/ThermoGFN/scripts/orchestration/kcat_m3_run_round.py)
  - [`scripts/orchestration/kcat_m3_run_experiment.py`](/home/ubuntu/amelie/ThermoGFN/scripts/orchestration/kcat_m3_run_experiment.py)

Important honesty constraint:

- the repo currently has the most concrete catalytic implementation in the Kcat-only Method III loop
- the generic neural teacher/student GFlowNet exposition in the paper is still broader than the simplified current implementation in [`train/thermogfn/method3_core.py`](/home/ubuntu/amelie/ThermoGFN/train/thermogfn/method3_core.py)
- the manuscript should therefore distinguish cleanly between:
  - full methodology proposed in the paper
  - concrete Kcat-only active-learning path already wired in this repo

## Planned Manuscript Changes

## 1. Front matter and abstract

Update:

- `\hypersetup{...pdftitle=...}`
- `\title{...}`
- abstract

Changes:

- expand scope from thermostable and binding-aware design to thermostable, binding-aware, and catalysis-aware design
- mention the catalytic oracle trio explicitly
- add catalytic-target conditioning to the list of methodological contributions
- update the “tri-fidelity design stack” language so catalysis is not framed as an afterthought

## 2. Introduction

Add catalytic design framing:

- distinguish stability, binding, and catalytic turnover as distinct but coupled objectives
- introduce enzyme kinetic parameters:
  - $k_{\rm cat}$
  - $K_M$
  - $k_{\rm cat}/K_M$
- explain why catalytic optimization is different from pure thermostability optimization:
  - depends on substrate identity
  - may depend on products and assay context
  - often requires pocket-sensitive structure features
- explain why a multi-oracle catalytic stack is needed:
  - KcatNet for wide throughput
  - MMKcat for missing-modality robustness
  - GraphKcat for pocket- and structure-aware refinement

## 3. Technical review section

Insert a new subsection in `Technical review of the component models`:

- `\subsection{Why catalytic-oracle models are required, and how KcatNet, GraphKcat, and MMKcat differ}`

Content to add:

- KcatNet:
  - paired enzyme-sequence and substrate representation
  - geometric deep learning with protein and substrate graphs
  - strong candidate for dense $k_{\rm cat}$ screening
- GraphKcat:
  - pocket extraction around bound ligand
  - multi-scale pocket graph
  - joint prediction of $\log k_{\rm cat}$, $\log K_M$, and $\log (k_{\rm cat}/K_M)$
  - stronger structural inductive bias
- MMKcat:
  - multimodal prediction under missing modality
  - product-aware turnover modeling
  - mask ensemble as a built-in robustness/uncertainty mechanism
- make explicit that these models are not replacements for SPURS/BioEmu/UMA:
  - they answer a different axis of the design problem
  - they are catalytic oracles rather than thermostability or binding oracles

## 4. Problem formulation and notation

Extend notation to cover reaction-conditioned design.

### 4a. New quantities in sign-convention table

Add rows for:

- $\widehat \ell_{k_{\rm cat}} = \widehat{\log_{10} k_{\rm cat}}$
- $\widehat \ell_{K_M} = \widehat{\log_{10} K_M}$
- $\widehat \ell_{\mathrm{eff}} = \widehat{\log_{10}(k_{\rm cat}/K_M)}$

Directions:

- larger $\widehat \ell_{k_{\rm cat}}$ is better
- smaller $\widehat \ell_{K_M}$ is better
- larger $\widehat \ell_{\mathrm{eff}}$ is better

### 4b. Extend training targets / readouts table

Add catalytic entries under:

- cheap/middle/high-fidelity computational targets
- downstream experimental readouts

Examples:

- Michaelis-Menten assay estimates of $k_{\rm cat}$ and $K_M$
- catalytic efficiency $k_{\rm cat}/K_M$
- turnover retention after thermal challenge

### 4c. Add reaction-conditioned design object

Introduce something like:

$$
\mathcal{R} = (\mathcal{S}_{\mathrm{sub}}, \mathcal{S}_{\mathrm{prod}}, \eta),
$$

where:

- $\mathcal{S}_{\mathrm{sub}}$ is substrate specification
- $\mathcal{S}_{\mathrm{prod}}$ is optional product specification
- $\eta$ is assay/environment metadata such as pH, temperature, organism

Then define catalytic oracle channels as:

$$
\widehat y_{\mathrm{cat}}^O(x,\mathcal{R}) \in \{\widehat \ell_{k_{\rm cat}}, \widehat \ell_{K_M}, \widehat \ell_{\mathrm{eff}}\}.
$$

### 4d. Extend target vector

Expand the target vector to allow catalytic goals such as:

- target $k_{\rm cat}$
- upper bound on $K_M$
- lower bound on catalytic efficiency
- multi-objective combinations with thermostability and binding

## 5. New dedicated section for catalysis and kinetic-parameter design

Repurpose the duplicated second UMA section into:

- `\section{Catalytic-oracle protocol for enzyme kinetics and activity-aware design}`

This new major section should be comparable in depth to the existing BioEmu and UMA protocol sections.

Recommended subsection layout:

1. Why a dedicated catalytic-oracle protocol is required
2. Reaction-conditioned inputs, metadata, and admissibility
3. KcatNet protocol as the wide catalytic screen
4. MMKcat protocol as the missing-modality catalytic oracle
5. GraphKcat protocol as the pocket-aware structural catalytic oracle
6. Multi-oracle catalytic calibration and uncertainty
7. Catalytic-oracle routing by metadata availability and structure availability

### Core mathematical content for this section

#### KcatNet formalization

Define a simplified predictor:

$$
\widehat \ell_{k_{\rm cat}}^{\mathrm{KN}} = f_{\mathrm{KN}}\!\big(E_{\mathrm{ProtT5}}(s), E_{\mathrm{ESM}}(s), G_{\mathrm{sub}}(u)\big).
$$

Explain:

- protein language features
- substrate graph / molecular features
- interaction graph layers
- global pooling and scalar regression

#### MMKcat formalization

Define modality set:

$$
\mathcal{M} = \{\text{sub}, \text{enz-seq}, \text{enz-graph}, \text{prod}\}.
$$

Define mask-conditioned predictions:

$$
\widehat \ell_{k_{\rm cat}}^{\mathrm{MM}}(m) = f_{\mathrm{MM}}(z \odot m),
\qquad m \in \mathcal{M}_{\mathrm{mask}}.
$$

Then aggregate:

$$
\mu_{\mathrm{MM}}(x) = \frac{1}{|\mathcal{M}_{\mathrm{mask}}|}\sum_m \widehat \ell_{k_{\rm cat}}^{\mathrm{MM}}(m),
\qquad
\sigma_{\mathrm{MM}}^2(x) = \Var_m\big[\widehat \ell_{k_{\rm cat}}^{\mathrm{MM}}(m)\big].
$$

This matches the local wrapper behavior and gives a principled uncertainty channel.

#### GraphKcat formalization

Define a pocket-conditioned predictor:

$$
(\widehat \ell_{k_{\rm cat}}^{\mathrm{GK}}, \widehat \ell_{K_M}^{\mathrm{GK}}, \widehat \ell_{\mathrm{eff}}^{\mathrm{GK}})
=
f_{\mathrm{GK}}\!\big(
G_{\mathrm{pocket}}(x,u),
E_{\mathrm{ESM2}}(s),
E_{\mathrm{UniMol}}(u),
\eta
\big).
$$

Add a consistency identity:

$$
\widehat \ell_{\mathrm{eff}}^{\mathrm{GK}} \approx \widehat \ell_{k_{\rm cat}}^{\mathrm{GK}} - \widehat \ell_{K_M}^{\mathrm{GK}},
$$

and recommend either explicit post hoc checking or a consistency regularizer in learned fusion.

#### Catalytic fusion

Add a catalytic fused score such as:

$$
z_{\mathrm{cat}}(x,\mathcal{R})
=
\rho_{\mathrm{KN}} w_{\mathrm{KN}} z_{\mathrm{KN}}^{\mathrm{risk}}
\rho_{\mathrm{MM}} w_{\mathrm{MM}} z_{\mathrm{MM}}^{\mathrm{risk}}
\rho_{\mathrm{GK}} \Big(
w_{\mathrm{GK},k} z_{\mathrm{GK},k}^{\mathrm{risk}}
+ w_{\mathrm{GK},m} z_{\mathrm{GK},m}^{\mathrm{risk}}
+ w_{\mathrm{GK},e} z_{\mathrm{GK},e}^{\mathrm{risk}}
\Big)
+ w_{\mathrm{agree}} z_{\mathrm{agree}}.
$$

where:

- $z_{\mathrm{GK},m}^{\mathrm{risk}}$ uses the sign-flipped $\log K_M$ channel
- $z_{\mathrm{agree}}$ penalizes large disagreement between overlapping predictors

## 6. Reward section updates

Add new subsections inside `Reward design and tri-fidelity oracle fusion`:

- `\subsection{Catalytic branch: KcatNet, MMKcat, and GraphKcat}`
- `\subsection{Joint reward for thermostability, binding, and catalysis}`

Content:

- define risk-adjusted catalytic channels
- define catalytic uncertainty sources:
  - checkpoint or ensemble variance for KcatNet
  - mask disagreement for MMKcat
  - structure / conformer / assay-context sensitivity for GraphKcat
- define catalytic target-attainment terms
- update the final scalarized reward equation to include catalytic channels

The final score should become explicitly multi-axis:

- stability
- binding
- catalysis
- packing / plausibility
- OOD penalties

## 7. Method sections

### 7a. Method I

Add one subsection:

- `\subsection{Catalytic-oracle augmentation of \methodI{}}`

Content:

- dense KcatNet or MMKcat scoring on terminal candidates when reaction metadata exist
- GraphKcat screening on a smaller subset with structure and ligand geometry
- catalytic reward fused jointly with thermostability and binding where tasks require it
- explain that not every task activates every channel

### 7b. Method III

This is the most important update because the repo already contains a Kcat-only Method III loop.

Add:

- a new subsection describing catalytic active learning
- a dedicated pseudo-code algorithm for the Kcat round

Recommended subsection titles:

- `\subsection{Catalytic active learning with KcatNet, MMKcat, and GraphKcat}`
- `\subsection{Kcat-specific surrogate, teacher reward, and student filtering}`

Core content to add:

- round dataset contains reaction-conditioned examples
- KcatNet and MMKcat are the wide catalytic stage
- GraphKcat is the narrower pocket-aware stage
- fused catalytic reward is used to update the teacher and to label the replay

Pseudo-code should mirror the local orchestration order:

1. fit surrogate on catalytic reward
2. train teacher
3. distill student
4. generate candidate pool
5. validate substrate metadata
6. score wide pool with KcatNet and/or MMKcat
7. select a narrower GraphKcat batch
8. score GraphKcat
9. fuse catalytic labels
10. append to dataset

Need an explicit note that the current repo implementation is a Kcat-only specialization of the broader Method III framework.

## 8. Data curation updates

Add catalytic data requirements:

- substrate SMILES required for KcatNet and GraphKcat
- optional products for MMKcat
- protein structure or CIF for GraphKcat
- optional organism / pH / temperature metadata

Mention local validation step from:

- [`scripts/prep/02_validate_kcat_metadata.py`](/home/ubuntu/amelie/ThermoGFN/scripts/prep/02_validate_kcat_metadata.py)

State explicitly that Kcat-mode examples without substrate metadata are invalid.

## 9. Hyperparameters and compute

Add catalytic defaults to the hyperparameter section:

- KcatNet batch size
- GraphKcat batch size
- wide KcatNet/MMKcat budget
- narrow GraphKcat budget
- fusion weights and risk kappas
- metadata-dependent routing defaults

Also add a short catalytic compute paragraph:

- KcatNet wide screening
- MMKcat moderate-cost multimodal inference
- GraphKcat narrower pocket extraction + structure-aware inference

## 10. Validation section

Add catalytic validation subsections / paragraphs:

- validation of $k_{\rm cat}$ calibration
- validation of $K_M$ and $k_{\rm cat}/K_M$ when GraphKcat is used
- agreement / disagreement analysis across KcatNet, MMKcat, and GraphKcat
- structure-availability-stratified evaluation
- missing-modality evaluation for MMKcat
- target-conditioned catalytic success metrics

Also extend design metrics to include:

- best catalytic reward
- top-$k$ catalytic efficiency
- stability-vs-catalysis Pareto quality
- catalytic agreement rate across oracles

## 11. Failure modes

Add catalytic-specific failure modes:

- exploiting KcatNet without pocket realism
- GraphKcat sensitivity to poor ligand conformers or wrong protein structure
- MMKcat instability under systematically missing product channels
- catalytic/stability trade-off collapse
- assay-context mismatch across pH / temperature / organism metadata
- false catalytic improvements that destroy binding or foldability

## 12. Bibliography

Add references for:

- KcatNet
- GraphKcat
- MMKcat

Use actual paper metadata where available, not generic repo-only placeholders.

## Implementation Plan

## Step 1

Create this plan file.

## Step 2

Patch manuscript front matter:

- title
- pdf title
- abstract
- introduction

## Step 3

Patch technical review and notation:

- new catalytic technical-review subsection
- sign table updates
- targets/readouts table updates
- target-vector updates

## Step 4

Repurpose duplicated UMA section into catalytic protocol section.

## Step 5

Patch reward and methods:

- catalytic reward subsections
- Method I catalytic augmentation
- Method III catalytic active-learning workflow
- new catalytic pseudo-code algorithm

## Step 6

Patch downstream sections:

- data
- hyperparameters
- validation
- failure modes
- conclusion
- appendix table / checklist entries if needed

## Step 7

Patch bibliography with catalytic-oracle citations.

## Step 8

Run lightweight validation:

- `python` scan or `rg` for new citation keys
- optional `pdflatex` compile if LaTeX environment is available

## Quality bar for the final TeX update

The revised paper should:

- present catalysis as a first-class design axis, not an appendix note
- give KcatNet, GraphKcat, and MMKcat distinct roles rather than collapsing them into one generic “kcat oracle”
- separate proposed methodology from current repo implementation where they differ
- preserve scientific caution about calibration and assay-context dependence
- remain coherent with the existing GFlowNet framing rather than reading like a bolted-on kinetics survey
