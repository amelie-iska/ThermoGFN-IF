import unittest

from train.thermogfn.method3_core import fit_surrogate_ensemble, train_teacher_policy, distill_student_from_teacher, teacher_student_kl


class TestMethod3Core(unittest.TestCase):
    def test_surrogate_teacher_student(self):
        records = [
            {
                "candidate_id": f"c{i}",
                "sequence": "ACDEFGHIKLMNPQ"[: (5 + (i % 5))],
                "K": i % 4,
                "prepared_atom_count": 100 + i,
                "reward": 1.0 + 0.1 * i,
            }
            for i in range(20)
        ]
        surrogate = fit_surrogate_ensemble(records, ensemble_size=4, seed=13)
        self.assertEqual(len(surrogate["models"]), 4)

        teacher = train_teacher_policy(records, seed=13)
        student = distill_student_from_teacher(teacher, records, seed=13)
        kl = teacher_student_kl(teacher, student)
        self.assertGreaterEqual(kl, 0.0)


if __name__ == "__main__":
    unittest.main()
