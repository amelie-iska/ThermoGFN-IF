import unittest

from train.thermogfn.method3_core import distill_student_from_teacher, train_teacher_policy


class TestMethod3PolicySupport(unittest.TestCase):
    def test_baseline_only_records_keep_nonzero_mutation_support(self):
        records = [
            {
                "candidate_id": f"c{i}",
                "sequence": "ACDEFGHIK",
                "K": 0,
                "prepared_atom_count": 500,
                "reward": 1.0,
            }
            for i in range(32)
        ]
        teacher = train_teacher_policy(records, seed=7)
        t_probs = {int(k): float(v) for k, v in teacher["k_probs"].items()}
        self.assertGreater(sum(t_probs.get(k, 0.0) for k in [1, 2, 3, 4]), 0.0)

        student = distill_student_from_teacher(teacher, records, seed=7)
        s_probs = {int(k): float(v) for k, v in student["k_probs"].items()}
        self.assertGreater(sum(s_probs.get(k, 0.0) for k in [1, 2, 3, 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
