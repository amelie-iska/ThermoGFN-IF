#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${THERMOGFN_UMA_CAT_CONFIG:-config/uma_cat_catalyst_gt_graphkcat_8round.yaml}"
RUNNER_ENV="${THERMOGFN_RUNNER_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REBUILD_DATASET=0
SKIP_ENV_CHECK=0
SKIP_ENV_HEALTH=0
DRY_RUN=0
HAS_OUTPUT_ROOT_OVERRIDE=0
PASSTHROUGH=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/orchestration/run_uma_cat_catalyst_gt_8round.sh [options]

Options:
  --config PATH       YAML config to run.
                      Default: config/uma_cat_catalyst_gt_graphkcat_8round.yaml
  --runner-env NAME   Conda env used to run orchestration and UMA dataset build.
                      Default: oracles.envs.uma_cat from config, then fairchem.
  --rebuild-dataset   Rebuild data.dataset_path from data.split_root before training.
  --skip-env-check    Do not preflight required conda env presence.
  --skip-env-health   Do not run import/model health checks before training.
  --dry-run           Forward to uma_cat_m3_run_experiment.py without launching stages.
  --no-progress       Disable tqdm progress bars.
  -h, --help          Show this help.

Any other arguments are passed through to uma_cat_m3_run_experiment.py.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --runner-env)
      RUNNER_ENV="$2"; shift 2 ;;
    --rebuild-dataset)
      REBUILD_DATASET=1; shift ;;
    --skip-env-check)
      SKIP_ENV_CHECK=1; shift ;;
    --skip-env-health)
      SKIP_ENV_HEALTH=1; shift ;;
    --dry-run)
      DRY_RUN=1
      PASSTHROUGH+=("$1"); shift ;;
    --no-progress)
      PASSTHROUGH+=("$1"); shift ;;
    --output-root)
      HAS_OUTPUT_ROOT_OVERRIDE=1
      PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      PASSTHROUGH+=("$1"); shift ;;
  esac
done

cfg_get() {
  local key="$1"
  local default_value="${2:-}"
  "$PYTHON_BIN" - "$REPO_ROOT" "$CONFIG_PATH" "$key" "$default_value" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
key = sys.argv[3]
default = sys.argv[4]
if not cfg_path.is_absolute():
    cfg_path = repo_root / cfg_path

try:
    import yaml
except Exception:
    print(default)
    raise SystemExit(0)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
cur = cfg
for part in key.split("."):
    if not isinstance(cur, dict) or part not in cur:
        print(default)
        raise SystemExit(0)
    cur = cur[part]
if cur is None:
    print(default)
elif isinstance(cur, bool):
    print("1" if cur else "0")
else:
    print(cur)
PY
}

cfg_required_envs() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$CONFIG_PATH" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
if not cfg_path.is_absolute():
    cfg_path = repo_root / cfg_path

try:
    import yaml
except Exception:
    raise SystemExit(0)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
required = cfg.get("env", {}).get("required", [])
if isinstance(required, list):
    for item in required:
        if item:
            print(str(item))
PY
}

check_envs_present() {
  local missing=()
  local env_name
  mapfile -t required_envs < <(cfg_required_envs)
  if [[ ${#required_envs[@]} -eq 0 ]]; then
    return 0
  fi
  for env_name in "${required_envs[@]}"; do
    if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
      missing+=("$env_name")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[uma-cat-8round] error: missing required conda env(s): ${missing[*]}" >&2
    echo "[uma-cat-8round] create GraphKcat env with: bash scripts/env/create_graphkcat_env.sh --env-name apodock --solver classic" >&2
    echo "[uma-cat-8round] or disable GraphKcat in config before running." >&2
    exit 2
  fi
}

check_envs_health() {
  local status_json="runs/env_status_uma_cat_catalyst_gt_graphkcat.json"
  mapfile -t required_envs < <(cfg_required_envs)
  if [[ ${#required_envs[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[uma-cat-8round] health-checking required envs: ${required_envs[*]}"
  RUN_HEALTH_CHECKS=1 bash scripts/env/check_kcat_envs.sh "$status_json" "${required_envs[@]}"
}

if [[ -z "$RUNNER_ENV" ]]; then
  RUNNER_ENV="$(cfg_get oracles.envs.uma_cat fairchem)"
fi

if [[ "$SKIP_ENV_CHECK" -eq 0 ]]; then
  check_envs_present
fi

if [[ "$SKIP_ENV_HEALTH" -eq 0 ]]; then
  check_envs_health
fi

RUN_ID="$(cfg_get run.run_id uma_cat_catalyst_gt_graphkcat_8round)"
DATASET_PATH="$(cfg_get data.dataset_path "")"
SPLIT_ROOT="$(cfg_get data.split_root "")"
SPLIT_NAME="$(cfg_get data.split train)"
BUILD_LIMIT="$(cfg_get data.build_limit "")"

if [[ "$DRY_RUN" -eq 1 && "$HAS_OUTPUT_ROOT_OVERRIDE" -eq 0 ]]; then
  DRY_OUTPUT_ROOT="runs/dryrun/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
  PASSTHROUGH+=(--output-root "$DRY_OUTPUT_ROOT")
  echo "[uma-cat-8round] dry-run output root: $DRY_OUTPUT_ROOT"
fi

if [[ -z "$DATASET_PATH" ]]; then
  echo "[uma-cat-8round] error: config must set data.dataset_path" >&2
  exit 2
fi

if [[ ! -f "$DATASET_PATH" || "$REBUILD_DATASET" -eq 1 ]]; then
  if [[ -z "$SPLIT_ROOT" ]]; then
    echo "[uma-cat-8round] error: dataset is missing and config does not set data.split_root" >&2
    echo "[uma-cat-8round] missing dataset: $DATASET_PATH" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$DATASET_PATH")"
  BUILD_CMD=(
    conda run --no-capture-output -n "$RUNNER_ENV"
    python scripts/rf3/build_uma_cat_dataset.py
    --split-root "$SPLIT_ROOT"
    --split "$SPLIT_NAME"
    --output-path "$DATASET_PATH"
    --run-id "$RUN_ID"
    --round-id 0
    --no-progress
  )
  if [[ -n "$BUILD_LIMIT" && "$BUILD_LIMIT" != "0" ]]; then
    BUILD_CMD+=(--limit "$BUILD_LIMIT")
  fi
  echo "[uma-cat-8round] building dataset: $DATASET_PATH"
  "${BUILD_CMD[@]}"
fi

echo "[uma-cat-8round] launching 8-round UMA-cat experiment from config: $CONFIG_PATH"
echo "[uma-cat-8round] runner env: $RUNNER_ENV"
exec conda run --no-capture-output -n "$RUNNER_ENV" \
  python scripts/orchestration/uma_cat_m3_run_experiment.py \
  --config "$CONFIG_PATH" \
  "${PASSTHROUGH[@]}"
