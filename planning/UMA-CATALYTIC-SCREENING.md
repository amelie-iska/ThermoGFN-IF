# Whole-Enzyme UMA Screening Workflow with Optional Steered MD Between Reactant-Bound and Product-Bound States

## Overview

This document describes an updated two-stage screening methodology for enzyme libraries under the assumption that each candidate already has:

- a **reactant-bound enzyme complex**, and
- a **product-bound enzyme complex**.

The workflow uses **whole-enzyme dynamics under UMA** and combines:

1. a **broad equilibrium screen** based on the population of **generalized near-attack conformations** (gNACs) or productive poses,
2. a **narrow free-energy screen** based on **chemical-step PMFs** and the resulting activation free energy $\Delta G^\ddagger $, and
3. an optional but highly useful **steered MD (sMD) branch** that drives the system between the reactant-bound and product-bound basins in order to:
   - reveal likely transition pathways,
   - identify productive collective variables,
   - harvest near-transition-state-like geometries,
   - improve PMF initialization,
   - estimate nonequilibrium work statistics, and
   - expose hidden gating motions or mechanistic bottlenecks that equilibrium sampling may miss.

The basic philosophy is that catalytic competence can be decomposed into two broad requirements:

- the enzyme must frequently organize the bound reactants into catalytically competent arrangements, and
- once in such arrangements, the enzyme must present a low free-energy barrier to chemical transformation.

The sMD branch strengthens this methodology by adding a directed dynamical probe of the pathway connecting reactant-like and product-like basins.

---

## 1. Mathematical framing

Let the full solvated enzyme-ligand system for a candidate enzyme be represented by atomic coordinates

$$
R = (r_1, r_2, \dots, r_N),
$$

with whole-system UMA potential

$$
U_{\mathrm{UMA}}(R).
$$

The corresponding atomic forces are

$$
F_j(R) = -\nabla_{r_j} U_{\mathrm{UMA}}(R).
$$

At the level of a single elementary chemical step, a transition-state-theory-like rate expression is

$$
k_{\mathrm{chem}} \approx \kappa \frac{k_B T}{h} e^{-\beta \Delta G^\ddagger},
$$

where $ \kappa $ is the transmission coefficient, $ k_B $ is Boltzmann's constant, $ h $ is Planck's constant, and $ \beta = (k_B T)^{-1} $.

For screening purposes, catalytic competence is usefully decomposed into a **gating term** and a **chemical barrier term**:

$$
\Delta G_{\mathrm{eff}} \approx \Delta G_{\mathrm{gate}} + \Delta G^\ddagger_{\mathrm{chem}}.
$$

This motivates a screening proxy of the form

$$
k_{\mathrm{proxy}} \propto \exp\!\left[-\beta\left(\Delta G_{\mathrm{gate}} + \Delta G^\ddagger_{\mathrm{chem}}\right)\right].
$$

Stage 1 estimates $ \Delta G_{\mathrm{gate}} $. Stage 2 estimates $ \Delta G^\ddagger_{\mathrm{chem}} $. The sMD branch helps both stages by providing pathway information, reactive seeds, and nonequilibrium work observables.

---

## 2. Basins, reaction pathways, and why sMD is useful

In this updated framework, each design is assumed to have at least two known bound basins:

- a **reactant-bound basin** $ A $, and
- a **product-bound basin** $ B $.

In ordinary equilibrium MD, trajectories may remain trapped in one basin and fail to reveal the pathway connecting them on accessible timescales. Steered MD solves a different problem: rather than waiting for the system to discover a rare transition spontaneously, it **drives** the system along a chosen progress variable toward the target basin.

This is useful in several distinct ways.

First, sMD can reveal **which structural motions must accompany chemistry**. These may include loop closures, side-chain rotations, water insertion or expulsion, proton-relay alignment, fragment reorientation, or substrate compression along an attack coordinate.

Second, sMD can generate ensembles of **highly informative intermediate structures**, including geometries close to the barrier region. Those structures can then be recycled into umbrella windows, restrained equilibration seeds, or transition-region clustering analyses.

Third, sMD can identify whether the path between reactant and product basins is mechanically smooth or whether it involves metastable intermediates, steric bottlenecks, or hidden orthogonal coordinates.

Fourth, nonequilibrium work accumulated during pulling can be analyzed to estimate free-energy differences or at least to rank candidates by how difficult the forced conversion appears to be.

The key point is that sMD does **not** replace equilibrium screening or PMFs. It augments them.

---

## 3. Stage 1 broad screen: generalized NAC / productive-pose population

### 3.1 Definition of the generalized NAC manifold

For a given elementary step $ s $, define a mechanism-specific feature vector

$$
q_s(R) =
\left(
 d_{\mathrm{form}},
 d_{\mathrm{break}},
 \theta_{\mathrm{attack}},
 d_{\mathrm{PT},1},
 d_{\mathrm{PT},2},
 \chi_{\mathrm{cat}},
 n_{\mathrm{HB}},
 o_{\mathrm{wat}},
 \phi_{\mathrm{frag}},
 E_{\parallel}
\right).
$$

Typical components include:

- $ d_{\mathrm{form}} $: distance of the bond that will form,
- $ d_{\mathrm{break}} $: distance of the bond that will break,
- $ \theta_{\mathrm{attack}} $: nucleophilic or in-line attack angle,
- $ d_{\mathrm{PT},1}, d_{\mathrm{PT},2} $: proton-transfer donor-acceptor distances,
- $ \chi_{\mathrm{cat}} $: catalytic side-chain torsions,
- $ n_{\mathrm{HB}} $: occupancy of stabilizing hydrogen bonds,
- $ o_{\mathrm{wat}} $: catalytic-water or relay occupancy,
- $ \phi_{\mathrm{frag}} $: relative orientation and translation of substrate fragments,
- $ E_{\parallel} $: projected electric field along the reactive bond or reaction dipole axis.

Define the generalized productive region $ \mathcal{R}_s $ and the corresponding indicator

$$
I_{\mathrm{gNAC}}(R) =
\begin{cases}
1, & q_s(R) \in \mathcal{R}_s,\\
0, & \text{otherwise.}
\end{cases}
$$

Then the productive-pose population is

$$
p_{\mathrm{gNAC}} = \int I_{\mathrm{gNAC}}(R)\,\rho_A(R)\,dR,
$$

where $ \rho_A(R) $ is the equilibrium density in the reactant-bound basin.

The corresponding gating free energy is

$$
\Delta G_{\mathrm{gate}} = -RT \ln\!\left(\frac{p_{\mathrm{gNAC}}}{1-p_{\mathrm{gNAC}}}\right)
\approx -RT \ln p_{\mathrm{gNAC}}
\quad \text{when } p_{\mathrm{gNAC}} \ll 1.
$$

### 3.2 Soft productive-pose score

Because a hard cutoff is brittle, define a soft score

$$
S(R) = \sum_m w_m f_m\!\big(q_m(R)\big),
$$

with smooth feature rewards such as

$$
f_m(q_m) = \exp\!\left[-\frac{(q_m - q_m^\star)^2}{2\sigma_m^2}\right].
$$

Then define a soft productive probability

$$
p_{\mathrm{soft}} = \left\langle \sigma\!\big(\alpha(S(R)-S_0)\big) \right\rangle,
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### 3.3 Broad-screen estimators

If $ R_t^{(k)} $ denotes frame $ t $ from replicate $ k $, then

$$
\hat p_{\mathrm{gNAC}} =
\frac{1}{\sum_{k=1}^{K} T_k}
\sum_{k=1}^{K} \sum_{t=1}^{T_k}
I_{\mathrm{gNAC}}\!\left(R_t^{(k)}\right).
$$

The estimated gating free energy is

$$
\widehat{\Delta G}_{\mathrm{gate}} =
-RT \ln\!\left(\frac{\hat p_{\mathrm{gNAC}}}{1-\hat p_{\mathrm{gNAC}}}\right).
$$

Additional broad-stage observables should include:

- mean soft productive score $ \langle S \rangle $,
- number of productive-pose visits $ N_{\mathrm{visits}} $,
- mean dwell time $ \tau_{\mathrm{dwell}} $,
- first-passage time to the productive region $ t_{\mathrm{first\,hit}} $,
- replicate-to-replicate variance $ \mathrm{Var}_k(\hat p_{\mathrm{gNAC}}^{(k)}) $.

A practical ranking functional is

$$
J_1(E_i) =
a_1\big(-\widehat{\Delta G}_{\mathrm{gate}}\big)
+a_2 \langle S \rangle
+a_3 \log(1+N_{\mathrm{visits}})
+a_4 \log(1+\tau_{\mathrm{dwell}})
-a_5\,\mathrm{Var}_k(\hat p_{\mathrm{gNAC}}^{(k)})
-a_6 C_{\mathrm{instability}}.
$$

---

## 4. Adding steered MD between reactant-bound and product-bound states

## 4.1 Basic idea

In the updated procedure, one runs steered MD not only from equilibrium initial states, but also between the two known bound basins:

- **forward pulls** from the reactant-bound state toward the product-bound state,
- optionally **reverse pulls** from the product-bound state toward the reactant-bound state.

The key addition is a steering coordinate or progress variable $ \lambda(R) $ that measures progress between the two endpoint states. This can be defined in several ways.

A simple geometric choice is an RMSD-like or distance-based interpolation coordinate,

$$
\lambda(R) = \sum_n c_n \xi_n(R),
$$

where the $ \xi_n $ are chemically chosen descriptors. More often, one uses a spring restraint to a target path value:

$$
U_{\mathrm{bias}}(R,t) = \frac{k_{\mathrm{s}}}{2}\big(\lambda(R)-\lambda_0(t)\big)^2,
$$

where $ k_{\mathrm{s}} $ is the pulling spring constant and $ \lambda_0(t) $ is the moving steering center, often taken as

$$
\lambda_0(t) = \lambda_0(0) + vt,
$$

with pulling speed $ v $.

The total steered potential is therefore

$$
U_{\mathrm{tot}}(R,t) = U_{\mathrm{UMA}}(R) + U_{\mathrm{bias}}(R,t).
$$

The instantaneous pulling force along the steering coordinate is

$$
F_{\mathrm{pull}}(t) = -\frac{\partial U_{\mathrm{bias}}}{\partial \lambda} = -k_{\mathrm{s}}\big(\lambda(R)-\lambda_0(t)\big).
$$

The nonequilibrium work performed along a trajectory is

$$
W = \int_0^{\tau} \frac{\partial U_{\mathrm{bias}}(R,t)}{\partial t}\,dt.
$$

---

## 4.2 How sMD helps the existing methodology

### 4.2.1 Better collective variables

One of the hardest problems in PMF construction is choosing the right collective variables. If the coordinate is wrong or incomplete, the PMF can be misleading.

sMD helps because the forced trajectories reveal which coordinates actually move during conversion between the reactant-bound and product-bound ensembles. In practice, one can project the pulled ensemble onto candidate descriptors and identify which ones change monotonically, which ones show hysteresis, and which ones reveal metastable plateaus.

This lets you replace an ad hoc reaction coordinate with one that is empirically informed by pathway data.

### 4.2.2 Harvesting near-transition-state-like geometries

A major advantage of sMD is that it creates trajectories that pass through high-strain, high-work regions between endpoint basins. These regions are not guaranteed to be the true transition-state ensemble, but they are often enriched in **near-transition-state-like geometries**.

Operationally, one can identify candidate near-TS frames by combining several criteria:

- large instantaneous force $ |F_{\mathrm{pull}}| $,
- local maxima in accumulated work growth,
- reactive coordinates near equal-bond or bond-switching values,
- maximal structural frustration or inflection in pathway descriptors,
- convergence of forward and reverse pathways in configuration space.

For example, if the chemical coordinate is

$$
\xi = d_{\mathrm{break}} - d_{\mathrm{form}},
$$

then frames with $ \xi \approx 0 $ are natural candidates for near-TS seeding in a bond-exchange reaction. If proton transfer is involved, one may instead look for

$$
\xi_1 = d(\mathrm{H},A) - d(\mathrm{H},D) \approx 0,
\qquad
\xi_2 = d_{\mathrm{break}} - d_{\mathrm{form}} \approx 0.
$$

These candidate frames can then be clustered, restrained, and used as seeds for local equilibrium sampling or umbrella windows.

### 4.2.3 Better PMF initialization

One of the main numerical problems in umbrella sampling is getting windows initialized near the intended region of phase space without violent relaxation.

sMD solves that by providing a continuous ladder of configurations spanning the reactant-like and product-like sides. Umbrella windows can therefore be seeded directly from frames along the pulled path. This improves overlap between neighboring windows and reduces the amount of equilibration needed in each one.

### 4.2.4 Detecting hidden gating coordinates

A pulled trajectory often reveals that the apparent reaction coordinate is insufficient. The work profile may show plateaus or secondary peaks caused not by the main chemical coordinate itself, but by orthogonal motions such as loop closure, water penetration, side-chain packing, or substrate-fragment reorientation.

This is valuable because it tells you that the free-energy landscape has a coupled form, for example

$$
W = W(\xi_{\mathrm{chem}}, \xi_{\mathrm{gate}}),
$$

rather than a simple one-dimensional dependence on the chemical coordinate alone.

In such cases, sMD does not merely accelerate sampling. It diagnoses the need for a higher-dimensional PMF.

### 4.2.5 Nonequilibrium work as an additional ranking signal

For each steered trajectory one obtains a work value $ W $. Across multiple trajectories, one gets a work distribution $ P(W) $.

The mean work, dissipated work, width of the work distribution, and difference between forward and reverse work distributions can all be used as supplementary screening observables.

A candidate enzyme that requires very large work to be driven from reactant-like to product-like states may be kinetically or mechanically unfavorable, even before a full PMF is computed.

A simple derived quantity is the dissipated work,

$$
W_{\mathrm{diss}} = \langle W \rangle - \Delta F,
$$

where $ \Delta F $ is the equilibrium free-energy difference between endpoint basins. Large dissipation indicates substantial irreversibility or poor alignment of the pulling coordinate with the natural pathway.

---

## 5. Theoretical framework for extracting free-energy information from sMD

## 5.1 Jarzynski equality

If repeated nonequilibrium pulling trajectories connect the same endpoints, the free-energy difference between those endpoints can be estimated from the work distribution using Jarzynski's equality:

$$
e^{-\beta \Delta F} = \left\langle e^{-\beta W} \right\rangle.
$$

Equivalently,

$$
\Delta F = -\beta^{-1} \ln \left\langle e^{-\beta W} \right\rangle.
$$

This is exact in principle, but in practice convergence can be difficult because the exponential average is dominated by rare low-work trajectories.

## 5.2 Cumulant approximation

If the work distribution is roughly Gaussian and pulling is not too far from reversible, one may use a second-order cumulant approximation:

$$
\Delta F \approx \langle W \rangle - \frac{\beta}{2}\mathrm{Var}(W).
$$

This is often more stable numerically, though less formally general.

## 5.3 Hummer-Szabo reconstruction

Under appropriate pulling protocols, the nonequilibrium trajectories can be used not only to estimate endpoint free-energy differences but to reconstruct a PMF along the pulling coordinate. If $ z $ is the pulled coordinate, one can use reweighting formulas of the Hummer-Szabo type to estimate

$$
G(z) = -k_B T \ln P(z) + C
$$

from steered trajectories.

In practice, this means that sMD can act as an **initial PMF generator** or **pathway reconstructor**, not merely as a source of endpoint work values.

## 5.4 Crooks relation and forward-reverse pulling

If both forward and reverse steered trajectories are available, the Crooks fluctuation theorem provides an additional relationship:

$$
\frac{P_F(W)}{P_R(-W)} = e^{\beta(W-\Delta F)}.
$$

The crossing point of the forward and reverse work distributions occurs at

$$
W = \Delta F.
$$

This is useful because forward-reverse pulling can provide a more robust estimate of endpoint free-energy differences and can diagnose hysteresis. Large separation between forward and reverse work distributions suggests strong irreversibility, poor coordinate choice, or unresolved orthogonal barriers.

---

## 6. Using sMD to obtain near-transition-state-like geometries

## 6.1 Why it is only “near-TS” and not automatically the true TS

The true transition-state ensemble is defined dynamically, not just structurally. In strict terms it is the set of configurations with committor probability

$$
p_B(R) = \frac{1}{2},
$$

meaning that trajectories launched from such a configuration commit to reactants and products with equal probability.

A frame extracted from sMD is not automatically in this ensemble. However, sMD is extremely useful for enriching configurations near the barrier region because it forces the system through high-work, high-strain regions that equilibrium sampling may visit only rarely.

Thus, the practical goal is not to claim that pulled frames are exact transition states, but to use them as **candidate near-TS seeds**.

## 6.2 Practical near-TS harvesting criteria

A robust near-TS selection protocol may use a weighted score such as

$$
\Theta(R) =
b_1\,\tilde F_{\mathrm{pull}}(R)
+b_2\,\tilde W_{\mathrm{local}}(R)
+b_3\,\tilde Q_{\mathrm{sym}}(R)
+b_4\,\tilde C_{FR}(R)
-b_5\,\tilde D_{\mathrm{endpoint}}(R),
$$

where:

- $ \tilde F_{\mathrm{pull}} $ is a normalized instantaneous pulling force,
- $ \tilde W_{\mathrm{local}} $ is a local work-density or force-accumulation score,
- $ \tilde Q_{\mathrm{sym}} $ measures proximity to bond-switching or proton-sharing symmetry,
- $ \tilde C_{FR} $ measures how closely forward and reverse pathways overlap at that configuration class,
- $ \tilde D_{\mathrm{endpoint}} $ penalizes states too similar to either endpoint basin.

Frames with high $ \Theta(R) $ are then clustered to identify distinct near-TS families.

## 6.3 What to do with near-TS candidates

Near-TS candidates can be used in several ways:

1. **Seed umbrella windows** around the barrier region.
2. **Seed restrained equilibrium sampling** to estimate local fluctuations near the barrier.
3. **Run short unbiased or weakly biased launches** to test how rapidly they fall toward reactant or product basins.
4. **Construct string or path collective variables** using the clustered intermediates.
5. **Perform limited committor-like diagnostics** on the top candidates in the final screening stage.

This is one of the biggest practical advantages of integrating sMD into the workflow.

---

## 7. Updated Stage 2 PMF construction with sMD assistance

Without sMD, the PMF stage starts by choosing collective variables and manually initializing windows. With sMD, the process becomes much more data-driven.

### 7.1 Data-driven CV discovery

Candidate CVs can be ranked by how well they parameterize the pulled pathway. Useful diagnostics include:

- monotonicity along pulling time,
- low hysteresis between forward and reverse pulls,
- strong correlation with work accumulation,
- ability to separate metastable intermediates,
- ability to align different steered trajectories after reparameterization by path progress.

A path-CV formulation is often natural. Given a discrete set of reference images $ \{R_i^\ast\} $ along the pulled path, define a path-progress variable

$$
s(R) = \frac{\sum_i i\,e^{-\lambda \|R-R_i^\ast\|^2}}{\sum_i e^{-\lambda \|R-R_i^\ast\|^2}},
$$

and a path-distance variable

$$
z(R) = -\lambda^{-1} \ln \sum_i e^{-\lambda \|R-R_i^\ast\|^2}.
$$

Then one may construct PMFs in $ s $, or in $ (s,z) $, rather than in a crude single bond-distance coordinate.

### 7.2 sMD-seeded umbrella windows

Suppose $ \xi $ is the chosen PMF coordinate. After clustering the steered trajectories, select representative frames spanning the full coordinate range and use them to initialize umbrella windows with bias

$$
U_j(R) = U_{\mathrm{UMA}}(R) + \frac{k_j}{2}\big(\xi(R)-\xi_j^\star\big)^2.
$$

Because the seeds come from an already traversed path, neighboring windows are typically much better overlapped than in manually constructed initializations.

### 7.3 Multi-dimensional PMFs when sMD reveals hidden coupling

If the work profile indicates coupled gating, then define a multidimensional PMF,

$$
W(\xi_{\mathrm{chem}}, \xi_{\mathrm{gate}}) = -k_B T \ln P(\xi_{\mathrm{chem}}, \xi_{\mathrm{gate}}) + C.
$$

Examples of $ \xi_{\mathrm{gate}} $ include:

- loop-closure distance,
- active-site volume,
- catalytic-water coordination number,
- fragment assembly angle,
- side-chain rotamer state.

This is often the correct move when a nominally chemical transition is actually controlled by coupled structural rearrangement.

---

## 8. Additional optional uses of sMD in the workflow

## 8.1 Mechanism discrimination

If more than one chemical mechanism is plausible, one can define distinct steering protocols for each mechanistic hypothesis and compare:

- work distributions,
- pathway smoothness,
- metastable intermediates,
- near-TS cluster quality,
- PMF convergence quality after seeding.

Mechanisms that repeatedly yield pathological pulling behavior or extremely unfavorable work signatures may be deprioritized.

## 8.2 Screening catalytic resilience

By varying pulling speed $ v $, spring constant $ k_s $, or endpoint definitions, one can test whether the candidate enzyme supports a stable family of productive pathways or only a fragile single route. Robust catalysts should not depend on a vanishingly narrow steering protocol.

## 8.3 Identifying gating bottlenecks for redesign

If pulling repeatedly identifies the same obstructing loop, steric clash, water exclusion event, or fragment misalignment, that feature becomes a rational redesign target. In this sense, sMD is not only a scoring tool but a **diagnostic design-feedback tool**.

## 8.4 Building reactive path ensembles

Instead of using only one pulled path, generate many forward and reverse trajectories. The union of these forms a **reactive path ensemble** that can be clustered into common pathway classes. The population of these classes, together with their work statistics, can be used to characterize mechanistic diversity across enzyme candidates.

---

## 9. Updated ranking structure

The original methodology ranked candidates first by a broad productive-pose score and then by PMF-derived barrier. With sMD added, one can introduce a pathway quality score.

Define a pathway score

$$
J_{\mathrm{sMD}}(E_i) =
-c_1 \langle W_F \rangle
-c_2 \langle W_R \rangle
-c_3 \Delta W_{FR}
-c_4 W_{\mathrm{diss}}
+c_5 N_{\mathrm{nearTS}}
+c_6 O_{\mathrm{path}}
-c_7 C_{\mathrm{hyst}},
$$

where:

- $ \langle W_F \rangle $ is mean forward work,
- $ \langle W_R \rangle $ is mean reverse work,
- $ \Delta W_{FR} $ measures forward-reverse mismatch,
- $ W_{\mathrm{diss}} $ is dissipated work,
- $ N_{\mathrm{nearTS}} $ is the number or density of harvested near-TS candidates,
- $ O_{\mathrm{path}} $ measures overlap and continuity of pathway ensembles,
- $ C_{\mathrm{hyst}} $ penalizes severe hysteresis.

A combined ranking strategy might then be

$$
J_{\mathrm{final}}(E_i) =
\omega_1 J_1(E_i)
+ \omega_2 J_{\mathrm{sMD}}(E_i)
- \omega_3 \Delta G^\ddagger_{\mathrm{chem}}(E_i).
$$

Alternatively, if a purely free-energy-like quantity is preferred, one may keep the final physical score as

$$
J_2(E_i) = \Delta G_{\mathrm{gate}}(E_i) + \Delta G^\ddagger_{\mathrm{chem}}(E_i),
$$

and use $ J_{\mathrm{sMD}} $ only as a support score for prioritization, tie-breaking, or adaptive allocation of PMF resources.

---

## 10. Updated workflow

### Tier 1: equilibrium broad screen

For each candidate enzyme:

- run multiple whole-enzyme UMA trajectories from the reactant-bound complex,
- compute generalized NAC / productive-pose statistics,
- rank by $ J_1 $.

### Tier 2: optional sMD branch

For top candidates or ambiguous candidates:

- run forward sMD from reactant-bound to product-bound states,
- optionally run reverse sMD from product-bound to reactant-bound states,
- analyze work distributions, pathway intermediates, and near-TS candidate frames,
- update collective-variable definitions and pathway hypotheses.

### Tier 3: PMF stage

For finalists:

- use sMD-derived intermediates and near-TS candidates to seed umbrella windows,
- compute PMFs for the elementary chemical step or coupled chemical-gating process,
- extract $ \Delta G^\ddagger $.

### Tier 4: final diagnostics

For the very top designs:

- compare forward and reverse pulling consistency,
- repeat PMF calculations from independent sMD-derived seed families,
- optionally perform short commitment or splitting-probability tests on near-TS candidates.

---

## 11. Pseudocode

```python
# Inputs:
# designs: enzyme candidates
# reactant_complexes: whole-enzyme reactant-bound complexes
# product_complexes: whole-enzyme product-bound complexes
# mechanism_set: hypothesized elementary-step mechanisms

stage1_records = []

for design in designs:
    R_complex = reactant_complexes[design]

    # Stage 1: equilibrium reactant-basin sampling
    replicas = []
    for k in range(K_stage1):
        traj = run_uma_md(
            system=R_complex,
            mode="whole_enzyme",
            purpose="broad_screen",
            seed=k
        )
        replicas.append(traj)

    stats = compute_productive_pose_statistics(
        replicas,
        mechanism_templates=mechanism_set
    )

    J1 = aggregate_stage1_score(stats)

    stage1_records.append({
        "design": design,
        "J1": J1,
        "stats": stats
    })

# Select designs for optional sMD branch
smd_candidates = select_for_smd(stage1_records)

for rec in smd_candidates:
    design = rec["design"]
    A = reactant_complexes[design]
    B = product_complexes[design]

    forward_trajs = []
    reverse_trajs = []

    for k in range(K_smd):
        ftraj = run_steered_md(
            start=A,
            target=B,
            direction="forward",
            seed=k
        )
        forward_trajs.append(ftraj)

        rtraj = run_steered_md(
            start=B,
            target=A,
            direction="reverse",
            seed=k
        )
        reverse_trajs.append(rtraj)

    path_data = analyze_smd_ensemble(
        forward_trajs=forward_trajs,
        reverse_trajs=reverse_trajs,
        extract_work=True,
        extract_intermediates=True,
        detect_hidden_cvs=True,
        harvest_near_ts=True
    )

    rec["smd_data"] = path_data
    rec["JsMD"] = compute_smd_score(path_data)

# Finalists for PMF
finalists = select_top_fraction(smd_candidates, fraction=top_fraction)

for rec in finalists:
    pmf_results = []

    for mech in mechanism_set:
        cvs = define_collective_variables(
            mech,
            equilibrium_stats=rec["stats"],
            smd_data=rec.get("smd_data")
        )

        seeds = choose_umbrella_seeds(
            from_stage1=rec["stats"],
            from_smd=rec.get("smd_data")
        )

        windows = place_umbrella_windows(seeds=seeds, cv_def=cvs)

        biased_trajs = []
        for w in windows:
            traj = run_uma_umbrella_md(system=w.system, bias=w.bias)
            biased_trajs.append(traj)

        pmf = reconstruct_pmf(biased_trajs, method="MBAR")
        barrier = estimate_deltaG_dagger(pmf)

        pmf_results.append({
            "mechanism": mech.name,
            "pmf": pmf,
            "DeltaG_dagger": barrier
        })

    rec["pmf_results"] = pmf_results
    rec["J2"] = combine_gate_and_barrier(rec["stats"], pmf_results)
    rec["Jfinal"] = combine_all_scores(rec["J1"], rec.get("JsMD"), rec["J2"])

ranked_designs = sort_by_final_score(finalists, key="Jfinal")
```

---

## 12. Interpretation

The addition of sMD upgrades the methodology in three important ways.

First, it provides a **direct pathway probe** between the known reactant-bound and product-bound basins. This gives information that equilibrium sampling alone may fail to reveal.

Second, it provides an efficient route to **near-transition-state-like structures** and **pathway-informed collective variables**, both of which materially improve PMF construction.

Third, it introduces a family of **nonequilibrium work observables** that can be used for candidate ranking, mechanism discrimination, and redesign diagnostics.

The resulting methodology is therefore not merely equilibrium screening plus PMFs. It becomes a three-layer strategy:

1. **equilibrium preorganization screening**,
2. **nonequilibrium pathway probing**, and
3. **free-energy barrier estimation**.

That combination is substantially stronger than any one component alone.

---

## 13. Summary

The updated workflow is:

1. run whole-enzyme UMA equilibrium MD from the reactant-bound state,
2. estimate productive-pose population and $\Delta G_{\mathrm{gate}}$,
3. optionally run forward and reverse sMD between reactant-bound and product-bound states,
4. use the sMD branch to:

   * identify pathway coordinates,
   * discover hidden gating motions,
   * collect nonequilibrium work statistics,
   * harvest near-transition-state-like geometries,
   * seed umbrella windows more intelligently,
5. compute PMFs and extract ( \Delta G^\ddagger ),
6. rank candidates using productive-pose statistics, pathway quality, and free-energy barriers.

In physical terms, the method screens enzyme candidates by asking three linked questions:

* does the enzyme frequently organize the reactants into productive arrangements?
* is there a mechanically and chemically plausible route connecting reactant-like and product-like basins?
* once the system reaches the barrier region, how high is the remaining free-energy barrier?

That is the most useful role of steered MD in this workflow: it bridges equilibrium catalytic preorganization and quantitative barrier estimation. 