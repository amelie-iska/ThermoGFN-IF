#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_JSON="${1:-${REPO_ROOT}/runs/env_status_kcat.json}"
shift || true
RUN_HEALTH="${RUN_HEALTH_CHECKS:-0}"

if [[ $# -gt 0 ]]; then
  REQUIRED_ENVS=("$@")
else
  REQUIRED_ENVS=("ligandmpnn_env" "KcatNet" "apodock")
fi

mkdir -p "$(dirname "$OUT_JSON")"

python - <<'PY' "$REPO_ROOT" "$OUT_JSON" "$RUN_HEALTH" "${REQUIRED_ENVS[@]}"
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
out_json = Path(sys.argv[2])
run_health = sys.argv[3] == "1"
required_envs = list(sys.argv[4:])

sys.path.insert(0, str(repo_root))
from train.thermogfn.env_utils import list_conda_envs, run_health_check

envs = list_conda_envs()
report = {"required": {}, "optional": {}}

for name in required_envs:
    status = "ready" if name in envs else "missing"
    report["required"][name] = {
        "status": status,
        "path": envs.get(name),
    }

if run_health:
    for name in required_envs:
        if name not in envs:
            continue
        ok, details = run_health_check(name)
        report["required"][name]["health_ok"] = ok
        report["required"][name]["health_details"] = details
        report["required"][name]["status"] = "ready" if ok else "exists_unchecked"

out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

missing = [k for k, v in report["required"].items() if v.get("status") == "missing"]
if missing:
    print(f"Missing required envs: {', '.join(missing)}")
    raise SystemExit(1)

if run_health:
    unhealthy = [k for k, v in report["required"].items() if v.get("status") != "ready"]
    if unhealthy:
        print(f"Required env health checks failed: {', '.join(unhealthy)}")
        raise SystemExit(1)

print(f"Wrote environment status to {out_json}")
PY
