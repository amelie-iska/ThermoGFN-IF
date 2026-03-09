import unittest

from train.thermogfn.schemas import validate_candidate, validate_oracle_score, SchemaError


class TestSchemas(unittest.TestCase):
    def test_candidate_valid(self):
        rec = {
            "candidate_id": "c1",
            "run_id": "r1",
            "round_id": 0,
            "task_type": "monomer",
            "backbone_id": "b1",
            "seed_id": "s1",
            "sequence": "ACDE",
            "mutations": [],
            "K": 0,
            "prepared_atom_count": 100,
            "eligibility": {"bioemu": True, "uma_whole": True, "uma_local": False},
            "source": "baseline",
            "schema_version": "v1",
        }
        validate_candidate(rec)

    def test_candidate_invalid_k(self):
        rec = {
            "candidate_id": "c1",
            "run_id": "r1",
            "round_id": 0,
            "task_type": "monomer",
            "backbone_id": "b1",
            "seed_id": "s1",
            "sequence": "ACDE",
            "mutations": ["A1C"],
            "K": 0,
            "prepared_atom_count": 100,
            "eligibility": {"bioemu": True, "uma_whole": True, "uma_local": False},
            "source": "baseline",
            "schema_version": "v1",
        }
        with self.assertRaises(SchemaError):
            validate_candidate(rec)

    def test_oracle_valid(self):
        rec = {
            "candidate_id": "c1",
            "rho_B": 1.0,
            "rho_U": 0.5,
            "reward": 1.2,
            "spurs_mode": "single",
        }
        validate_oracle_score(rec)

    def test_candidate_ppi_valid_with_decomposition(self):
        rec = {
            "candidate_id": "c2",
            "run_id": "r1",
            "round_id": 0,
            "task_type": "ppi",
            "backbone_id": "b1",
            "seed_id": "s1",
            "sequence": "ACDE",
            "mutations": [],
            "K": 0,
            "prepared_atom_count": 1000,
            "eligibility": {"bioemu": False, "uma_whole": True, "uma_local": False},
            "source": "baseline",
            "schema_version": "v1",
            "decomposition": {"bound": "complex", "components": ["a", "b"], "stoichiometry": [1, 1]},
        }
        validate_candidate(rec)


if __name__ == "__main__":
    unittest.main()
