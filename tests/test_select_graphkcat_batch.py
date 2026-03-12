import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSelectGraphKcatBatch(unittest.TestCase):
    def test_require_graphkcat_ok_filters_errors(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_path = td_path / "in.jsonl"
            output_path = td_path / "out.jsonl"
            rows = [
                {"candidate_id": "good_hi", "graphkcat_status": "ok", "graphkcat_log_kcat": 4.0, "graphkcat_std": 0.2},
                {"candidate_id": "bad_err", "graphkcat_status": "error", "graphkcat_log_kcat": 10.0, "graphkcat_std": 0.1},
                {"candidate_id": "good_lo", "graphkcat_status": "ok", "graphkcat_log_kcat": 3.0, "graphkcat_std": 0.1},
            ]
            with input_path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row))
                    fh.write("\n")

            subprocess.run(
                [
                    "python",
                    "scripts/train/m3_select_graphkcat_batch.py",
                    "--input-path",
                    str(input_path),
                    "--output-path",
                    str(output_path),
                    "--budget",
                    "3",
                    "--require-graphkcat-ok",
                    "--no-progress",
                ],
                check=True,
                cwd="/home/ubuntu/amelie/ThermoGFN",
            )

            with output_path.open("r", encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh]
            self.assertEqual([row["candidate_id"] for row in out_rows], ["good_hi", "good_lo"])


if __name__ == "__main__":
    unittest.main()
