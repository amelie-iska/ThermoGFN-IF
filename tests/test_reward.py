import unittest

from train.thermogfn.reward import compute_fused_score, compute_reliability_gates


class TestReward(unittest.TestCase):
    def test_reward_positive(self):
        rec = {
            "task_type": "monomer",
            "prepared_atom_count": 500,
            "K": 2,
            "spurs_mean": 0.2,
            "spurs_std": 0.1,
            "bioemu_calibrated": 0.3,
            "bioemu_std": 0.1,
            "uma_calibrated": 0.4,
            "uma_std": 0.1,
            "rho_B": 1.0,
            "rho_U": 1.0,
            "pack_unc": 0.1,
        }
        out = compute_fused_score(rec)
        self.assertGreater(out["reward"], 0.0)

    def test_gates(self):
        rec = {"task_type": "monomer", "prepared_atom_count": 9500}
        rho_b, rho_u = compute_reliability_gates(rec)
        self.assertEqual(rho_b, 1.0)
        self.assertEqual(rho_u, 0.5)


if __name__ == "__main__":
    unittest.main()
