import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class BuildUmaCatDatasetFromSplitTest(unittest.TestCase):
    def test_build_from_rf3_pair_split_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_root = root / "rf3_split"
            (split_root / "train").mkdir(parents=True)
            (split_root / "test").mkdir(parents=True)
            (split_root / "metadata").mkdir(parents=True)
            reactant_cif = root / "reactant_model.cif"
            product_cif = root / "product_model.cif"
            reactant_cif.write_text("data_test\n")
            product_cif.write_text("data_test\n")
            spec = {
                "pair_id": "pair_001",
                "sequence": "ACDE",
                "sequence_length": 4,
                "protein_chain_id": "A",
                "ligand_chain_id": "B",
                "rhea_id": "RHEA:1",
                "uniprot_id": "P00001",
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "ligand_smiles": "CCO",
                "pocket_positions": [1, 2],
                "reactant_complex_path": str(reactant_cif),
                "product_complex_path": str(product_cif),
                "reactant_protein_path": str(reactant_cif),
                "product_protein_path": str(product_cif),
                "representative_structure_path": str(reactant_cif),
                "prepared_atom_count": 10,
                "specification": {"extra": {"task_name": "rf3_reactzyme_protrek_pair_split", "example_id": "pair_001"}},
            }
            (split_root / "train" / "pair_001.json").write_text(json.dumps(spec))
            (split_root / "metadata" / "split_summary.json").write_text(json.dumps({"n_train": 1, "n_test": 0}))
            out_path = root / "dataset.jsonl"
            cmd = [
                "python",
                str(REPO_ROOT / "scripts/rf3/build_uma_cat_dataset.py"),
                "--prepared-input-root",
                str(root),
                "--reactant-root",
                str(root),
                "--product-root",
                str(root),
                "--split-root",
                str(split_root),
                "--output-path",
                str(out_path),
                "--run-id",
                "test_run",
                "--split",
                "train",
                "--round-id",
                "0",
                "--no-progress",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["pair_id"], "pair_001")
            self.assertEqual(row["sequence"], "ACDE")
            self.assertEqual(row["reactant_complex_path"], str(reactant_cif))
            self.assertEqual(row["product_complex_path"], str(product_cif))


if __name__ == "__main__":
    unittest.main()
