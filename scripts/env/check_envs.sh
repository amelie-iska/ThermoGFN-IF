#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_JSON="${1:-${REPO_ROOT}/runs/env_status.json}"
RUN_HEALTH="${RUN_HEALTH_CHECKS:-0}"

mkdir -p "$(dirname "$OUT_JSON")"

python - <<'PY' "$REPO_ROOT" "$OUT_JSON" "$RUN_HEALTH"
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
out_json = Path(sys.argv[2])
run_health = sys.argv[3] == "1"

sys.path.insert(0, str(repo_root))
from train.thermogfn.env_utils import check_envs

report = check_envs(run_health_checks=run_health)
out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

required = report.get("required", {})
missing = [k for k, v in required.items() if v.get("status") == "missing"]
if missing:
    print(f"Missing required envs: {', '.join(missing)}")
    raise SystemExit(1)
if run_health:
    unhealthy = [k for k, v in required.items() if v.get("status") != "ready"]
    if unhealthy:
        print(f"Required env health checks failed: {', '.join(unhealthy)}")
        raise SystemExit(1)
print(f"Wrote environment status to {out_json}")
PY
