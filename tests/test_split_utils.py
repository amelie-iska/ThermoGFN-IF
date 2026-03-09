import unittest
from pathlib import Path

from train.thermogfn.split_utils import discover_split, build_design_index


class TestSplitUtils(unittest.TestCase):
    def test_discover_and_index(self):
        root = Path("data/rfd3_splits/unconditional_monomer_protrek35m")
        paths = discover_split(root)
        rows = build_design_index(paths)
        self.assertGreater(len(rows), 0)
        self.assertIn("split", rows[0])
        self.assertIn("spec_path", rows[0])


if __name__ == "__main__":
    unittest.main()
