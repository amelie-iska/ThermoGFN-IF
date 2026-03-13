import math
import unittest

import numpy as np

from train.thermogfn.uma_cat_runtime import (
    StructureData,
    _build_smd_record_steps,
    _gating_delta_g_std_kcal_mol,
    _jarzynski_profile,
    _project_reaction_progress_lambda,
    analyze_endpoint_protocol,
    build_guided_ligand_path_targets,
    build_interpolated_ca_elastic_network,
    build_ligand_graph_model,
    build_ligand_restraint_model,
    build_rigid_ligand_path_targets,
    map_ligand_atoms_between_endpoints,
    match_backbone_ca_indices,
    match_backbone_heavy_indices,
    radius_of_gyration,
    summarize_catalytic_screen,
)
from train.thermogfn.uma_cat_reward import KB_KCAL_MOL_K


class TestUmaCatRuntime(unittest.TestCase):
    def test_build_smd_record_steps_matches_image_grid(self):
        steps = _build_smd_record_steps(96, 48)
        self.assertEqual(int(steps[0]), 0)
        self.assertEqual(int(steps[-1]), 96 * 48)
        self.assertEqual(len(steps), 97)
        self.assertTrue(np.all(np.diff(steps) == 48))

    def test_rigid_ligand_path_preserves_internal_geometry(self):
        ligand = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        matched_start = ligand[[0, 1, 2]]
        rot_z_90 = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        matched_end = matched_start @ rot_z_90.T + np.asarray([5.0, 2.0, 0.0], dtype=float)
        lambdas = np.linspace(0.0, 1.0, 5, dtype=float)
        path = build_rigid_ligand_path_targets(
            ligand_positions=ligand,
            matched_start_positions=matched_start,
            matched_end_positions=matched_end,
            lambdas=lambdas,
        )
        ref_dist = np.linalg.norm(ligand[0] - ligand[3])
        for frame in path:
            self.assertAlmostEqual(np.linalg.norm(frame[0] - frame[3]), ref_dist, places=6)
        np.testing.assert_allclose(path[0], ligand, atol=1e-6)

    def test_ligand_mapping_prefers_atom_name_identity(self):
        react = StructureData(
            path="reactant",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [7.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C", "C", "O", "P"],
            atom_names=["CA", "CB", "CG", "CD", "C0", "O0", "P0"],
            residue_names=["ALA", "ALA", "ALA", "ALA", "LIG", "LIG", "LIG"],
            chain_ids=["A", "A", "A", "A", "B", "B", "B"],
            residue_ids=[10, 10, 11, 11, 0, 0, 0],
            group_pdb=["ATOM", "ATOM", "ATOM", "ATOM", "HETATM", "HETATM", "HETATM"],
        )
        product = StructureData(
            path="product",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C", "O", "C", "P"],
            atom_names=["CA", "CB", "CG", "CD", "O0", "C0", "P0"],
            residue_names=["ALA", "ALA", "ALA", "ALA", "LIG", "LIG", "LIG"],
            chain_ids=["A", "A", "A", "A", "B", "B", "B"],
            residue_ids=[10, 10, 11, 11, 0, 0, 0],
            group_pdb=["ATOM", "ATOM", "ATOM", "ATOM", "HETATM", "HETATM", "HETATM"],
        )
        out = map_ligand_atoms_between_endpoints(
            react,
            product,
            protein_chain_id="A",
            ligand_chain_id="B",
            pocket_positions=[10, 11],
        )
        react_names = [react.atom_names[i] for i in out["reactant_indices"]]
        prod_names = [product.atom_names[i] for i in out["product_indices"]]
        self.assertEqual(react_names, prod_names)
        self.assertEqual(out["exact_name_matches"], 3)
        self.assertEqual(out["alignment_mode"], "backbone_ca")

    def test_match_backbone_ca_indices(self):
        left = StructureData(
            path="left",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C"],
            atom_names=["CA", "CA", "CA"],
            residue_names=["ALA", "GLY", "SER"],
            chain_ids=["A", "A", "A"],
            residue_ids=[10, 11, 12],
            group_pdb=["ATOM", "ATOM", "ATOM"],
        )
        right = StructureData(
            path="right",
            positions=np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C"],
            atom_names=["CA", "CA", "CA"],
            residue_names=["ALA", "GLY", "SER"],
            chain_ids=["A", "A", "A"],
            residue_ids=[10, 11, 12],
            group_pdb=["ATOM", "ATOM", "ATOM"],
        )
        left_idx, right_idx = match_backbone_ca_indices(left, right, chain_id="A")
        np.testing.assert_array_equal(left_idx, np.asarray([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(right_idx, np.asarray([0, 1, 2], dtype=np.int64))

    def test_match_backbone_heavy_indices(self):
        left = StructureData(
            path="left",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["N", "C", "C", "O"],
            atom_names=["N", "CA", "C", "O"],
            residue_names=["ALA", "ALA", "ALA", "ALA"],
            chain_ids=["A", "A", "A", "A"],
            residue_ids=[10, 10, 10, 10],
            group_pdb=["ATOM", "ATOM", "ATOM", "ATOM"],
        )
        right = StructureData(
            path="right",
            positions=np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["N", "C", "C", "O"],
            atom_names=["N", "CA", "C", "O"],
            residue_names=["ALA", "ALA", "ALA", "ALA"],
            chain_ids=["A", "A", "A", "A"],
            residue_ids=[10, 10, 10, 10],
            group_pdb=["ATOM", "ATOM", "ATOM", "ATOM"],
        )
        left_idx, right_idx = match_backbone_heavy_indices(left, right, chain_id="A")
        np.testing.assert_array_equal(left_idx, np.asarray([0, 1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(right_idx, np.asarray([0, 1, 2, 3], dtype=np.int64))

    def test_build_ligand_restraint_model_turns_broken_bond_off(self):
        react = StructureData(
            path="react",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.45, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C"],
            atom_names=["A1", "A2", "CA", "CA"],
            residue_names=["LIG", "LIG", "ALA", "GLY"],
            chain_ids=["B", "B", "A", "A"],
            residue_ids=[1, 1, 10, 11],
            group_pdb=["HETATM", "HETATM", "ATOM", "ATOM"],
        )
        mapping = {
            "reactant_indices": np.asarray([0, 1], dtype=np.int64),
            "product_indices": np.asarray([0, 1], dtype=np.int64),
            "product_aligned_positions": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        }
        model = build_ligand_restraint_model(
            reactant=react,
            ligand_indices=np.asarray([0, 1], dtype=np.int64),
            mapping=mapping,
        )
        self.assertEqual(model["bond_pairs"].shape[0], 1)
        self.assertAlmostEqual(float(model["bond_start_force_constants"][0]), 6.0, places=6)
        self.assertAlmostEqual(float(model["bond_end_force_constants"][0]), 0.0, places=6)
        self.assertEqual(int(model["reactive_atom_count"]), 2)
        np.testing.assert_allclose(model["steer_weights"], np.asarray([1.0, 1.0], dtype=float), atol=1e-8)
        self.assertEqual(model["steering_mode"], "reactive_center")

    def test_build_guided_ligand_path_targets_uses_internal_morph_for_reactive_center(self):
        react = StructureData(
            path="react",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C"],
            atom_names=["C1", "C2"],
            residue_names=["LIG", "LIG"],
            chain_ids=["B", "B"],
            residue_ids=[1, 1],
            group_pdb=["HETATM", "HETATM"],
        )
        mapping = {
            "reactant_indices": np.asarray([0, 1], dtype=np.int64),
            "product_indices": np.asarray([0, 1], dtype=np.int64),
            "product_aligned_positions": np.asarray(
                [
                    [0.2, 0.5, 0.0],
                    [1.7, 0.5, 0.0],
                ],
                dtype=float,
            ),
        }
        graph_model = {
            "steering_mode": "reactive_center",
            "symbols": ["C", "C"],
        }
        lambdas = np.asarray([0.0, 0.5, 1.0], dtype=float)
        path, mode, _ = build_guided_ligand_path_targets(
            reactant=react,
            ligand_indices=np.asarray([0, 1], dtype=np.int64),
            mapping=mapping,
            lambdas=lambdas,
            graph_model=graph_model,
        )
        self.assertEqual(mode, "reactive_center_internal_morph")
        np.testing.assert_allclose(path[0], react.positions[[0, 1]], atol=1e-8)
        np.testing.assert_allclose(path[-1], mapping["product_aligned_positions"][[0, 1]], atol=1e-8)

    def test_build_ligand_graph_model_gates_large_reactive_center(self):
        ligand_positions = np.asarray([[1.4 * i, 0.0, 0.0] for i in range(14)], dtype=float)
        product_positions = np.asarray([[4.0 * i, 0.0, 0.0] for i in range(14)], dtype=float)
        react = StructureData(
            path="react",
            positions=np.asarray(
                ligand_positions.tolist()
                + [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C"] * 16,
            atom_names=[f"A{i+1}" for i in range(14)] + ["CA", "CA"],
            residue_names=["LIG"] * 14 + ["ALA", "GLY"],
            chain_ids=["B"] * 14 + ["A", "A"],
            residue_ids=[1] * 14 + [10, 11],
            group_pdb=["HETATM"] * 14 + ["ATOM", "ATOM"],
        )
        mapping = {
            "reactant_indices": np.asarray(list(range(14)), dtype=np.int64),
            "product_indices": np.asarray(list(range(14)), dtype=np.int64),
            "product_aligned_positions": np.asarray(
                product_positions.tolist()
                + [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        }
        graph = build_ligand_graph_model(
            reactant=react,
            ligand_indices=np.asarray(list(range(14)), dtype=np.int64),
            mapping=mapping,
        )
        self.assertEqual(graph["steering_mode"], "component_pose_only")
        self.assertFalse(graph["steering_confident"])
        model = build_ligand_restraint_model(
            reactant=react,
            ligand_indices=np.asarray(list(range(14)), dtype=np.int64),
            mapping=mapping,
            graph_model=graph,
        )
        self.assertEqual(model["bond_pairs"].shape[0], 0)
        self.assertEqual(model["steering_mode"], "component_pose_only")

    def test_build_ligand_restraint_model_tracks_reactive_center(self):
        react = StructureData(
            path="react",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [5.4, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C", "C", "C"],
            atom_names=["A1", "A2", "B1", "B2", "CA", "CA"],
            residue_names=["LIG", "LIG", "LIG", "LIG", "ALA", "GLY"],
            chain_ids=["B", "B", "B", "B", "A", "A"],
            residue_ids=[1, 1, 2, 2, 10, 11],
            group_pdb=["HETATM", "HETATM", "HETATM", "HETATM", "ATOM", "ATOM"],
        )
        mapping = {
            "reactant_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "product_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "product_aligned_positions": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                    [1.8, 0.0, 0.0],
                    [3.2, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        }
        graph = build_ligand_graph_model(
            reactant=react,
            ligand_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            mapping=mapping,
        )
        self.assertEqual(graph["steering_mode"], "reactive_center")
        self.assertGreaterEqual(len(graph["formed_bonds"]), 1)
        model = build_ligand_restraint_model(
            reactant=react,
            ligand_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            mapping=mapping,
            graph_model=graph,
        )
        self.assertGreaterEqual(int(model["bond_pairs"].shape[0]), 1)
        self.assertEqual(model["steering_mode"], "reactive_center")

    def test_analyze_endpoint_protocol_respects_threshold_override(self):
        react = StructureData(
            path="react",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C"],
            atom_names=["CA", "CA", "L1", "L2"],
            residue_names=["ALA", "GLY", "LIG", "LIG"],
            chain_ids=["A", "A", "B", "B"],
            residue_ids=[10, 11, 1, 1],
            group_pdb=["ATOM", "ATOM", "HETATM", "HETATM"],
        )
        prod = StructureData(
            path="prod",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [4.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C"],
            atom_names=["CA", "CA", "L1", "L2"],
            residue_names=["ALA", "GLY", "LIG", "LIG"],
            chain_ids=["A", "A", "B", "B"],
            residue_ids=[10, 11, 1, 1],
            group_pdb=["ATOM", "ATOM", "HETATM", "HETATM"],
        )
        default_bundle = analyze_endpoint_protocol(
            reactant=react,
            product=prod,
            protein_chain_id="A",
            ligand_chain_id="B",
            pocket_positions=[10, 11],
        )
        self.assertEqual(default_bundle["protocol_meta"]["protocol_mode"], "reactive_center")
        self.assertTrue(default_bundle["protocol_meta"]["pmf_eligible"])
        strict_bundle = analyze_endpoint_protocol(
            reactant=react,
            product=prod,
            protein_chain_id="A",
            ligand_chain_id="B",
            pocket_positions=[10, 11],
            max_reactive_bonds=0,
            max_reactive_atoms=1,
            max_reactive_fraction=0.1,
        )
        self.assertEqual(strict_bundle["protocol_meta"]["protocol_mode"], "conformational_endpoint")
        self.assertFalse(strict_bundle["protocol_meta"]["pmf_eligible"])

    def test_bond_pairs_contribute_to_reaction_progress_lambda(self):
        lam = _project_reaction_progress_lambda(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                ],
                dtype=float,
            ),
            steered_indices=np.asarray([0, 1], dtype=np.int64),
            start_targets=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            end_targets=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            bond_pairs=np.asarray([[0, 2]], dtype=np.int64),
            bond_start_distances=np.asarray([4.0], dtype=float),
            bond_end_distances=np.asarray([2.0], dtype=float),
            bond_force_constants=np.asarray([1.0], dtype=float),
        )
        self.assertAlmostEqual(lam, 0.5, places=6)

    def test_jarzynski_profile_is_numerically_stable(self):
        traces = [
            np.asarray([0.0, 1000.0, 2000.0], dtype=float),
            np.asarray([0.0, 1001.0, 1999.0], dtype=float),
        ]
        lambdas, free = _jarzynski_profile(traces, temperature_k=300.0)
        self.assertEqual(lambdas.shape, (3,))
        self.assertTrue(np.all(np.isfinite(free)))

    def test_build_interpolated_ca_elastic_network(self):
        react = StructureData(
            path="react",
            positions=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [3.8, 0.0, 0.0],
                    [7.6, 0.0, 0.0],
                    [11.4, 0.0, 0.0],
                    [15.2, 0.0, 0.0],
                    [19.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            symbols=["C", "C", "C", "C", "C", "C"],
            atom_names=["CA", "CA", "CA", "CA", "CA", "CA"],
            residue_names=["ALA", "GLY", "SER", "THR", "LEU", "VAL"],
            chain_ids=["A", "A", "A", "A", "A", "A"],
            residue_ids=[1, 2, 3, 4, 5, 6],
            group_pdb=["ATOM", "ATOM", "ATOM", "ATOM", "ATOM", "ATOM"],
        )
        product_aligned = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.5, 0.0],
                [7.6, 0.5, 0.0],
                [11.4, 0.0, 0.0],
                [15.2, -0.5, 0.0],
                [19.0, -0.5, 0.0],
            ],
            dtype=float,
        )
        network = build_interpolated_ca_elastic_network(
            reactant=react,
            product_aligned_positions=product_aligned,
            reactant_ca_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            product_ca_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            sequential_k_eva2=6.0,
            midrange_k_eva2=2.0,
            contact_k_eva2=0.35,
            contact_cutoff_a=8.0,
        )
        self.assertGreaterEqual(len(network["pairs"]), 5)
        pair_to_k = {
            tuple(pair.tolist()): float(k)
            for pair, k in zip(network["pairs"], network["force_constants"], strict=False)
        }
        self.assertAlmostEqual(pair_to_k[(0, 1)], 6.0, places=6)
        self.assertAlmostEqual(pair_to_k[(0, 2)], 3.0, places=6)
        self.assertAlmostEqual(pair_to_k[(0, 4)], 2.0, places=6)

    def test_radius_of_gyration(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        rg = radius_of_gyration(positions, indices=np.asarray([0, 1, 2], dtype=np.int64))
        self.assertAlmostEqual(rg, math.sqrt(8.0 / 3.0), places=6)

    def test_gating_delta_g_std_positive(self):
        sigma = _gating_delta_g_std_kcal_mol(0.1, 0.02, 300.0)
        self.assertGreater(sigma, 0.0)

    def test_summarize_uses_physical_log_rate_uncertainty(self):
        broad = {
            "status": "ok",
            "p_gnac": 0.1,
            "p_soft": 0.2,
            "delta_g_gate_kcal_mol": 1.5,
            "delta_g_gate_std_kcal_mol": 0.5,
            "p_gnac_lcb": 0.08,
            "productive_visit_count": 3,
            "productive_dwell_frames": 5.0,
            "first_hit_frame": 10.0,
        }
        smd = {
            "status": "ok",
            "delta_g_smd_barrier_kcal_mol": 4.0,
            "delta_g_smd_barrier_std_kcal_mol": 1.0,
            "delta_g_react_to_prod_kcal_mol": 1.2,
            "delta_g_react_to_prod_std_kcal_mol": 0.3,
            "mean_final_work_kcal_mol": 2.0,
            "std_final_work_kcal_mol": 0.8,
            "forward_reverse_gap_kcal_mol": 4.0,
            "near_ts_candidates": [{}, {}],
        }

        out = summarize_catalytic_screen(
            broad=broad,
            smd=smd,
            pmf=None,
            temperature_k=300.0,
        )
        expected_barrier_std = math.sqrt(1.0**2 + 2.0**2)
        expected_log_std = math.sqrt(0.5**2 + expected_barrier_std**2) / (
            KB_KCAL_MOL_K * 300.0 * math.log(10.0)
        )
        self.assertAlmostEqual(out["uma_cat_delta_g_barrier_std_kcal_mol"], expected_barrier_std, places=6)
        self.assertAlmostEqual(out["uma_cat_log10_rate_std"], expected_log_std, places=6)
        self.assertEqual(out["uma_cat_barrier_source"], "smd")

    def test_summarize_prefers_pmf_uncertainty_when_available(self):
        broad = {
            "status": "ok",
            "p_gnac": 0.1,
            "p_soft": 0.2,
            "delta_g_gate_kcal_mol": 1.5,
            "delta_g_gate_std_kcal_mol": 0.5,
            "p_gnac_lcb": 0.08,
            "productive_visit_count": 3,
            "productive_dwell_frames": 5.0,
            "first_hit_frame": 10.0,
        }
        smd = {
            "status": "ok",
            "delta_g_smd_barrier_kcal_mol": 4.0,
            "delta_g_smd_barrier_std_kcal_mol": 1.0,
            "mean_final_work_kcal_mol": 2.0,
            "std_final_work_kcal_mol": 0.8,
            "forward_reverse_gap_kcal_mol": 8.0,
            "near_ts_candidates": [],
        }
        pmf = {
            "status": "ok",
            "delta_g_pmf_barrier_kcal_mol": 3.0,
            "delta_g_pmf_barrier_std_kcal_mol": 0.4,
            "delta_g_pmf_react_to_prod_kcal_mol": 0.9,
            "delta_g_pmf_react_to_prod_std_kcal_mol": 0.2,
        }

        out = summarize_catalytic_screen(
            broad=broad,
            smd=smd,
            pmf=pmf,
            temperature_k=300.0,
        )
        self.assertEqual(out["uma_cat_barrier_source"], "pmf")
        self.assertAlmostEqual(out["uma_cat_delta_g_barrier_std_kcal_mol"], 0.4, places=6)

    def test_summarize_marks_nonreactive_protocol_as_diagnostic(self):
        broad = {
            "status": "ok",
            "p_gnac": 0.1,
            "p_soft": 0.2,
            "delta_g_gate_kcal_mol": 1.5,
            "delta_g_gate_std_kcal_mol": 0.5,
            "p_gnac_lcb": 0.08,
            "productive_visit_count": 3,
            "productive_dwell_frames": 5.0,
            "first_hit_frame": 10.0,
        }
        smd = {
            "status": "ok",
            "mapping": {
                "protocol_mode": "conformational_endpoint",
                "protocol_reason": "too_many_graph_edits",
                "reactive_barrier_valid": False,
                "pmf_eligible": False,
            },
            "delta_g_smd_barrier_kcal_mol": 4.0,
            "delta_g_smd_barrier_std_kcal_mol": 1.0,
            "mean_final_work_kcal_mol": 2.0,
            "std_final_work_kcal_mol": 0.8,
            "forward_reverse_gap_kcal_mol": 0.0,
            "near_ts_candidates": [],
        }

        out = summarize_catalytic_screen(
            broad=broad,
            smd=smd,
            pmf=None,
            temperature_k=300.0,
        )
        self.assertEqual(out["uma_cat_protocol_mode"], "conformational_endpoint")
        self.assertEqual(out["uma_cat_protocol_reason"], "too_many_graph_edits")
        self.assertFalse(out["uma_cat_reactive_barrier_valid"])
        self.assertFalse(out["uma_cat_pmf_eligible"])
        self.assertEqual(out["uma_cat_barrier_source"], "diagnostic_smd")


if __name__ == "__main__":
    unittest.main()
