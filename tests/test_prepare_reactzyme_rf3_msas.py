import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "rf3" / "prepare_reactzyme_rf3_msas.py"
SPEC = importlib.util.spec_from_file_location("prepare_reactzyme_rf3_msas", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestPrepareReactzymeRf3Msas(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
