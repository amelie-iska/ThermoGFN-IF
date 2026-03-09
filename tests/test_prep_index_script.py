import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPrepIndexScript(unittest.TestCase):
    def test_build_training_index_script_supports_multi_root(self):
        with tempfile.TemporaryDirectory(prefix="thermogfn_test_") as tmp:
            out = Path(tmp) / "index.jsonl"
            cmd = [
                "python",
                "scripts/prep/02_build_training_index.py",
                "--split-root",
                "data/rfd3_splits/unconditional_monomer_protrek35m",
                "--split-root",
                "data/rfd3_splits/does_not_exist_yet",
                "--allow-missing",
                "--output",
                str(out),
            ]
            rc = subprocess.run(cmd, check=False).returncode
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            first = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(first.get("task_type"), "monomer")
            self.assertIn("split_root", first)
            self.assertIn("split_name", first)


if __name__ == "__main__":
    unittest.main()
