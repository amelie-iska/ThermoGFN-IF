import unittest

from train.thermogfn.method3_core import (
    distill_student_from_teacher,
    fit_surrogate_ensemble,
    generate_student_candidates,
    teacher_student_kl,
    train_teacher_policy,
)


class TestMethod3Core(unittest.TestCase):
    def test_surrogate_teacher_student(self):
        records = [
            {
                "candidate_id": f"seed_{i}",
                "run_id": "test",
                "round_id": 0,
                "task_type": "ligand",
                "backbone_id": f"b{i}",
                "seed_id": f"b{i}",
                "sequence": "ACDEFGHIK",
                "mutations": [],
                "K": 0,
                "prepared_atom_count": 100 + i,
                "eligibility": {"bioemu": False, "uma_whole": True, "uma_local": False},
                "source": "baseline",
                "schema_version": "v1",
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "protein_chain_id": "A",
                "reactant_complex_path": "/tmp/reactant.cif",
                "product_complex_path": "/tmp/product.cif",
                "pocket_positions": [1, 3, 5],
            }
            for i in range(2)
        ]
        records.extend(
            [
                {
                    **records[0],
                    "candidate_id": "m1",
                    "sequence": "YCDEFGHIK",
                    "mutations": ["A1Y"],
                    "K": 1,
                    "source": "student",
                    "reward": 1.25,
                },
                {
                    **records[0],
                    "candidate_id": "m2",
                    "sequence": "YCDEFGHVK",
                    "mutations": ["A1Y", "I8V"],
                    "K": 2,
                    "source": "student",
                    "reward": 1.55,
                },
                {
                    **records[1],
                    "candidate_id": "m3",
                    "sequence": "ACDEYGHIK",
                    "mutations": ["F5Y"],
                    "K": 1,
                    "source": "student",
                    "reward": 1.10,
                },
            ]
        )
        surrogate = fit_surrogate_ensemble(records, ensemble_size=4, seed=13)
        self.assertEqual(len(surrogate["models"]), 4)

        teacher = train_teacher_policy(records, seed=13, steps=64, gamma_off=0.7, surrogate=surrogate)
        student = distill_student_from_teacher(teacher, records, seed=13, steps=256)
        kl = teacher_student_kl(teacher, student)
        self.assertGreaterEqual(kl, 0.0)
        self.assertEqual(teacher["teacher_mode"], "trajectory_balance_gflownet")
        self.assertTrue(teacher["is_true_gflownet"])
        self.assertEqual(student["teacher_mode"], "trajectory_balance_gflownet")
        self.assertTrue(student["is_true_gflownet"])
        self.assertIn("params", teacher)
        self.assertAlmostEqual(sum(teacher["k_probs"].values()), 1.0, places=4)
        self.assertIn("position_probs_by_seed", student)

        pool = generate_student_candidates(
            student=student,
            seeds=[r for r in records if r["source"] == "baseline"],
            pool_size=6,
            run_id="test",
            round_id=0,
            seed=13,
        )
        self.assertGreaterEqual(len(pool), 1)
        self.assertTrue(all("candidate_id" in rec for rec in pool))
        self.assertTrue(all(rec["source"] == "student" for rec in pool))


if __name__ == "__main__":
    unittest.main()
