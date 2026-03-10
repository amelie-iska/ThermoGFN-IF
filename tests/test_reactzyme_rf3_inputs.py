import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "rf3" / "build_reactzyme_rf3_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_reactzyme_rf3_inputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestReactzymeRf3Inputs(unittest.TestCase):
    def test_count_sdf_atoms_reads_v2000_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "ligand.sdf"
            sdf_path.write_text(
                "\n".join(
                    [
                        "ligand",
                        "  RDKit          3D",
                        "",
                        " 10  9  0  0  0  0  0  0  0  0  0 V2000",
                        "M  END",
                        "$$$$",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(MODULE._count_sdf_atoms(sdf_path), 10)

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
            ligand_atom_count=42,
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
        self.assertEqual(example["metadata"]["ligand_atom_count"], 42)

    def test_main_applies_atom_limit_and_sequence_pair_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_root = tmp / "generate-constraints_0"
            output_root = tmp / "out"
            pocket_cache = source_root / "pocket_cache"
            output_templates = source_root / "output_sdf_templates" / "train"
            pocket_cache.mkdir(parents=True)
            output_templates.mkdir(parents=True)

            seq_tsv = source_root / "cleaned_uniprot_rhea.tsv"
            seq_tsv.write_text(
                "\n".join(
                    [
                        "Entry\tSequence",
                        "P1\tMAAA",
                        "P2\tMAAA",
                        "P3\tMAAA",
                        "P4\tMBBB",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            for uniprot_id in ("P1", "P2", "P3", "P4"):
                (pocket_cache / f"{uniprot_id}.json").write_text(
                    json.dumps(
                        {
                            "features": [
                                {
                                    "type": "Binding site",
                                    "location": {"position": {"value": 7}},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            def write_sdf(path: Path, atom_count: int) -> None:
                path.write_text(
                    "\n".join(
                        [
                            path.stem,
                            "  RDKit          3D",
                            "",
                            f"{atom_count:>3}  0  0  0  0  0  0  0  0  0  0 V2000",
                            "M  END",
                            "$$$$",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            manifest_path = output_templates / "manifest.csv"
            rows = []
            for pair_id, row_id, rhea_id, uniprot_id, atom_count in (
                ("pair_1", "row_1", "RHEA:1", "P1", 20),
                ("pair_2", "row_2", "RHEA:2", "P2", 30),
                ("pair_3", "row_3", "RHEA:3", "P3", 40),
                ("pair_4", "row_4", "RHEA:4", "P4", 300),
            ):
                reactant_sdf = output_templates / f"{pair_id}__reactant.sdf"
                product_sdf = output_templates / f"{pair_id}__product.sdf"
                write_sdf(reactant_sdf, atom_count)
                write_sdf(product_sdf, atom_count)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "row_id": row_id,
                        "rhea_id": rhea_id,
                        "uniprot_id": uniprot_id,
                        "status": "reactant:ok|product:ok",
                        "reactant_smiles": "CC.O",
                        "product_smiles": "C.CC",
                        "reactant_sdf": str(reactant_sdf),
                        "product_sdf": str(product_sdf),
                    }
                )

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            argv = sys.argv[:]
            try:
                sys.argv = [
                    str(MODULE_PATH),
                    "--source-root",
                    str(source_root),
                    "--manifest",
                    str(manifest_path),
                    "--sequence-tsv",
                    str(seq_tsv),
                    "--pocket-cache",
                    str(pocket_cache),
                    "--output-root",
                    str(output_root),
                    "--max-ligand-atoms",
                    "256",
                    "--max-pairs-per-sequence",
                    "2",
                    "--log-level",
                    "ERROR",
                ]
                rc = MODULE.main()
            finally:
                sys.argv = argv

            self.assertEqual(rc, 0)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["accepted_rows"], 2)
            self.assertEqual(summary["counts"]["emitted_examples"], 4)
            self.assertEqual(summary["counts"]["skipped_sequence_pair_cap"], 1)
            self.assertEqual(summary["counts"]["skipped_ligand_atom_limit"], 1)

            reactant_examples = json.loads((output_root / "reactant.json").read_text(encoding="utf-8"))
            self.assertEqual(len(reactant_examples), 2)
            self.assertTrue(all(ex["metadata"]["ligand_atom_count"] <= 256 for ex in reactant_examples))
            self.assertEqual(
                [row["pair_id"] for row in map(json.loads, (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) if row["state"] == "reactant"],
                ["pair_1", "pair_2"],
            )


if __name__ == "__main__":
    unittest.main()
