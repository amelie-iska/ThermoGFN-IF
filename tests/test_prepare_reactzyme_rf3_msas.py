import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "rf3" / "prepare_reactzyme_rf3_msas.py"
SPEC = importlib.util.spec_from_file_location("prepare_reactzyme_rf3_msas", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestPrepareReactzymeRf3Msas(unittest.TestCase):
    def test_validate_smiles_for_rf3_rejects_dummy_atoms(self):
        ok, reason = MODULE._validate_smiles_for_rf3(
            "*C",
            validation_cache={},
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "dummy_atom")

    def test_validate_pair_ligands_checks_both_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reactant_path = tmp / "reactant.json"
            product_path = tmp / "product.json"

            reactant_path.write_text(
                '[{"name":"pair_1__reactant","components":[{"seq":"MAAA","chain_id":"A"},{"smiles":"CCO","chain_id":"B"}],"metadata":{"pair_id":"pair_1","ligand_smiles":"CCO"}}]',
                encoding="utf-8",
            )
            product_path.write_text(
                '[{"name":"pair_1__product","components":[{"seq":"MAAA","chain_id":"A"},{"smiles":"*CCO","chain_id":"B"}],"metadata":{"pair_id":"pair_1","ligand_smiles":"*CCO"}}]',
                encoding="utf-8",
            )

            invalid_pair_reason, counts = MODULE._validate_pair_ligands(
                {
                    "reactant": ("state_json", [reactant_path]),
                    "product": ("state_json", [product_path]),
                },
                selected_pair_ids={"pair_1"},
            )

            self.assertEqual(invalid_pair_reason, {"pair_1": "dummy_atom"})
            self.assertEqual(counts["invalid_pair_total"], 1)
            self.assertEqual(counts["invalid_pair_dummy_atom"], 1)

    def test_trim_a3m_depth_keeps_first_n_sequences(self):
        a3m = (
            ">query\n"
            "AAAA\n"
            ">hit1\n"
            "BBBB\n"
            ">hit2\n"
            "CCCC\n"
            ">hit3\n"
            "DDDD\n"
        )

        trimmed = MODULE._trim_a3m_depth(a3m, 2)

        self.assertEqual(MODULE._count_a3m_sequences(trimmed), 2)
        self.assertIn(">query\nAAAA\n", trimmed)
        self.assertIn(">hit1\nBBBB\n", trimmed)
        self.assertNotIn(">hit2\nCCCC\n", trimmed)

    def test_trim_a3m_depth_zero_disables_trimming(self):
        a3m = ">query\nAAAA\n>hit1\nBBBB\n>hit2\nCCCC\n"
        trimmed = MODULE._trim_a3m_depth(a3m, 0)
        self.assertEqual(trimmed, a3m)

    def test_map_local_a3m_texts_to_sequences_uses_query_sequence(self):
        chunk = ["SEQAAAA", "SEQBBBB"]
        mapping = MODULE._map_local_a3m_texts_to_sequences(
            chunk,
            {
                "0.a3m": ">q1\nSEQBBBB\n>hit\nSEQBBBB\n",
                "1.a3m": ">q0\nSEQAAAA\n>hit\nSEQAAAA\n",
            },
        )

        self.assertEqual(set(mapping.keys()), set(chunk))
        self.assertIn("SEQAAAA", mapping["SEQAAAA"])
        self.assertIn("SEQBBBB", mapping["SEQBBBB"])

    def test_a3m_matches_sequence_validates_query_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.a3m"
            path.write_text(">query\nSEQAAAA\n>hit\nSEQAAAA\n", encoding="utf-8")

            self.assertTrue(MODULE._a3m_matches_sequence(path, "SEQAAAA"))
            self.assertFalse(MODULE._a3m_matches_sequence(path, "SEQBBBB"))

    def test_generate_msas_serial_writes_trimmed_a3ms(self):
        mmseqs2_mod = types.ModuleType("boltz.data.msa.mmseqs2")
        mmseqs2_mod.logger = types.SimpleNamespace(disabled=False)
        mmseqs2_mod.tqdm = object()

        def fake_run_mmseqs2(seqs, **kwargs):
            lines = []
            for idx, _seq in enumerate(seqs):
                lines.append(
                    (
                        f">query_{idx}\nAAAA\n"
                        f">hit1_{idx}\nBBBB\n"
                        f">hit2_{idx}\nCCCC\n"
                    )
                )
            return lines

        mmseqs2_mod.run_mmseqs2 = fake_run_mmseqs2

        boltz_mod = types.ModuleType("boltz")
        data_mod = types.ModuleType("boltz.data")
        msa_mod = types.ModuleType("boltz.data.msa")
        boltz_mod.data = data_mod
        data_mod.msa = msa_mod
        msa_mod.mmseqs2 = mmseqs2_mod

        fake_modules = {
            "boltz": boltz_mod,
            "boltz.data": data_mod,
            "boltz.data.msa": msa_mod,
            "boltz.data.msa.mmseqs2": mmseqs2_mod,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with mock.patch.dict(sys.modules, fake_modules, clear=False):
                seq_to_path = MODULE._generate_msas_via_server(
                    ["SEQ1", "SEQ2", "SEQ1"],
                    boltz_src_path=REPO_ROOT,
                    msa_cache_dir=cache_dir,
                    host_url="http://127.0.0.1:8080/api",
                    use_env=False,
                    use_filter=False,
                    pairing_strategy="greedy",
                    reuse_cache=False,
                    msa_batch_size=1,
                    msa_concurrency=1,
                    msa_retries=1,
                    msa_depth=2,
                )

            self.assertEqual(set(seq_to_path.keys()), {"SEQ1", "SEQ2"})
            for msa_path in seq_to_path.values():
                self.assertTrue(msa_path.exists())
                a3m = msa_path.read_text(encoding="utf-8")
                self.assertEqual(MODULE._count_a3m_sequences(a3m), 2)

    def test_generate_msas_local_direct_writes_trimmed_a3ms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            def fake_run_local_chunk(chunk, **kwargs):
                return {
                    sequence: (
                        f">query_{idx}\nAAAA\n"
                        f">hit1_{idx}\nBBBB\n"
                        f">hit2_{idx}\nCCCC\n"
                    )
                    for idx, sequence in enumerate(chunk)
                }

            with mock.patch.object(
                MODULE,
                "_infer_local_mmseqs_assets",
                return_value={
                    "mmseqs_bin": cache_dir / "mmseqs",
                    "db_prefix": cache_dir / "uniref30_2302_db",
                    "db_seq": cache_dir / "uniref30_2302_db_seq",
                    "db_aln": cache_dir / "uniref30_2302_db_aln",
                },
            ), mock.patch.object(MODULE, "_run_local_mmseqs_chunk", side_effect=fake_run_local_chunk):
                seq_to_path = MODULE._generate_msas_local_direct(
                    ["SEQ1", "SEQ2", "SEQ3", "SEQ1"],
                    local_msa_root=cache_dir,
                    msa_cache_dir=cache_dir,
                    reuse_cache=False,
                    msa_batch_size=2,
                    msa_concurrency=2,
                    msa_retries=1,
                    msa_depth=2,
                    cuda_devices=["0", "1"],
                    use_filter=False,
                    max_seqs=4096,
                    num_iterations=3,
                    msa_threads_per_job=1,
                )

            self.assertEqual(set(seq_to_path.keys()), {"SEQ1", "SEQ2", "SEQ3"})
            for msa_path in seq_to_path.values():
                self.assertTrue(msa_path.exists())
                a3m = msa_path.read_text(encoding="utf-8")
                self.assertEqual(MODULE._count_a3m_sequences(a3m), 2)

    def test_generate_msas_local_direct_rebuilds_invalid_cached_a3m(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            bad_cache = MODULE._msa_cache_path_for_sequence(cache_dir, "SEQ1")
            bad_cache.write_text(">query\nWRONGSEQ\n", encoding="utf-8")

            def fake_run_local_chunk(chunk, **kwargs):
                return {sequence: f">query\n{sequence}\n>hit\n{sequence}\n" for sequence in chunk}

            with mock.patch.object(
                MODULE,
                "_infer_local_mmseqs_assets",
                return_value={
                    "mmseqs_bin": cache_dir / "mmseqs",
                    "db_prefix": cache_dir / "uniref30_2302_db",
                    "db_seq": cache_dir / "uniref30_2302_db_seq",
                    "db_aln": cache_dir / "uniref30_2302_db_aln",
                },
            ), mock.patch.object(MODULE, "_run_local_mmseqs_chunk", side_effect=fake_run_local_chunk):
                seq_to_path = MODULE._generate_msas_local_direct(
                    ["SEQ1"],
                    local_msa_root=cache_dir,
                    msa_cache_dir=cache_dir,
                    reuse_cache=True,
                    msa_batch_size=1,
                    msa_concurrency=1,
                    msa_retries=1,
                    msa_depth=2048,
                    cuda_devices=["0"],
                    use_filter=False,
                    max_seqs=4096,
                    num_iterations=3,
                    msa_threads_per_job=1,
                )

            rewritten = seq_to_path["SEQ1"].read_text(encoding="utf-8")
            self.assertIn("SEQ1", rewritten)
            self.assertNotIn("WRONGSEQ", rewritten)


if __name__ == "__main__":
    unittest.main()
