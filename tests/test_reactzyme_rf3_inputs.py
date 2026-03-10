import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "rf3" / "build_reactzyme_rf3_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_reactzyme_rf3_inputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestReactzymeRf3Inputs(unittest.TestCase):
    def test_extract_pocket_positions_supports_points_and_ranges(self):
        features = [
            {"type": "Binding site", "location": {"position": {"value": 8}}},
            {
                "type": "Active site",
                "location": {"start": {"value": 11}, "end": {"value": 13}},
            },
            {"type": "Site", "location": {"position": {"value": 99}}},
        ]

        positions = MODULE._extract_pocket_positions_from_features(features)

        self.assertEqual(positions, [8, 11, 12, 13])

    def test_build_rf3_example_includes_constraint_and_template_fields(self):
        example = MODULE._build_rf3_example(
            pair_id="train__row__RHEA_1",
            state="reactant",
            uniprot_id="P12345",
            rhea_id="RHEA:1",
            row_id="train_0",
            protein_sequence="MPEPTIDE",
            ligand_smiles="CCO.O",
            ligand_sdf=Path("/tmp/test_reactant.sdf"),
            pocket_positions=[7, 11],
            protein_chain_id="A",
            ligand_chain_id="B",
            pocket_distance_threshold=8.0,
            template_threshold=0.5,
        )

        self.assertEqual(example["version"], 1)
        self.assertEqual(example["components"][0]["chain_id"], "A")
        self.assertEqual(example["components"][1]["chain_id"], "B")
        self.assertEqual(example["template_selection"], ["B"])
        self.assertEqual(example["ground_truth_conformer_selection"], ["B"])
        self.assertEqual(example["constraints"][0]["pocket"]["contacts"], [["A", 7], ["A", 11]])
        self.assertEqual(example["constraints"][0]["pocket"]["max_distance"], 8.0)
        self.assertEqual(example["templates"][0]["sdf"], "/tmp/test_reactant.sdf")
        self.assertEqual(example["metadata"]["fragment_count"], 2)


if __name__ == "__main__":
    unittest.main()
