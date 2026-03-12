import math
import unittest

from train.thermogfn.uma_cat_runtime import (
    _gating_delta_g_std_kcal_mol,
    summarize_catalytic_screen,
)
from train.thermogfn.uma_cat_reward import KB_KCAL_MOL_K


class TestUmaCatRuntime(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
