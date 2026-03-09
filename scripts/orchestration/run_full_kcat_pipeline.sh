#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="thermogfn_kcat"
CONFIG_PATH="config/kcat_m3_default.yaml"
OUTPUT_ROOT=""
NUM_ROUNDS=8
POOL_SIZE=50000
KCATNET_BUDGET=1024
GRAPHKCAT_BUDGET=256
MAX_CHECKPOINTS=5
SEED=13
STRICT_GATES=1
STEP_HEARTBEAT_SEC=30
EARLY_STOP_OVERFIT_GAP_TOP8=""
EARLY_STOP_OVERFIT_GAP_BEST=""
EARLY_STOP_PATIENCE=1
SPLIT_ROOTS=()
METADATA_OVERLAYS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/orchestration/run_full_kcat_pipeline.sh [options]

Options:
  --config PATH                    YAML config path (default: config/kcat_m3_default.yaml)
  --run-id ID                      Run identifier (default: thermogfn_kcat)
  --output-root PATH               Output root (default: runs/<run-id>)
  --split-root PATH                Add split root (repeatable; must carry substrate metadata
                                   for Kcat training unless paired with --metadata-overlay)
  --metadata-overlay PATH          JSONL/CSV overlay keyed by spec_path/stem/backbone_id/
                                   example_id to inject substrate metadata (repeatable)
  --rounds N                       Number of Kcat Method III rounds (default: 8)
  --pool-size N                    Candidate pool size per round (default: 50000)
  --kcatnet-budget N                KcatNet budget per round (default: 1024)
  --graphkcat-budget N             GraphKcat budget per round (default: 256)
  --max-checkpoints N              Max checkpoints per model family (default: 5)
  --seed N                         Random seed (default: 13)
  --step-heartbeat-sec N           Heartbeat interval in seconds (default: 30)
  --early-stop-overfit-gap-top8 X  Early-stop threshold for top8 train-test reward gap
  --early-stop-overfit-gap-best X  Early-stop threshold for best train-test reward gap
  --early-stop-patience N          Consecutive breach rounds before stop (default: 1)
  --no-strict-gates                Disable strict round gates
  -h, --help                       Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --split-root)
      SPLIT_ROOTS+=("$2"); shift 2 ;;
    --metadata-overlay)
      METADATA_OVERLAYS+=("$2"); shift 2 ;;
    --rounds)
      NUM_ROUNDS="$2"; shift 2 ;;
    --pool-size)
      POOL_SIZE="$2"; shift 2 ;;
    --kcatnet-budget)
      KCATNET_BUDGET="$2"; shift 2 ;;
    --graphkcat-budget)
      GRAPHKCAT_BUDGET="$2"; shift 2 ;;
    --max-checkpoints)
      MAX_CHECKPOINTS="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --step-heartbeat-sec)
      STEP_HEARTBEAT_SEC="$2"; shift 2 ;;
    --early-stop-overfit-gap-top8)
      EARLY_STOP_OVERFIT_GAP_TOP8="$2"; shift 2 ;;
    --early-stop-overfit-gap-best)
      EARLY_STOP_OVERFIT_GAP_BEST="$2"; shift 2 ;;
    --early-stop-patience)
      EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --no-strict-gates)
      STRICT_GATES=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ${#SPLIT_ROOTS[@]} -eq 0 ]]; then
  echo "[kcat-pipeline] error: at least one --split-root is required" >&2
  exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="runs/${RUN_ID}"
fi

BOOTSTRAP_DIR="${OUTPUT_ROOT}/bootstrap"
KCAT_DIR="${OUTPUT_ROOT}/kcat_m3"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$BOOTSTRAP_DIR" "$KCAT_DIR" "$LOG_DIR"

ENV_STATUS="${OUTPUT_ROOT}/env_status.json"
ENV_STATUS_HEALTH="${OUTPUT_ROOT}/env_status_health.json"

echo "[kcat-pipeline] checking conda environments for Kcat stage"
./scripts/env/check_kcat_envs.sh "$ENV_STATUS" ligandmpnn_env KcatNet apodock
RUN_HEALTH_CHECKS=1 ./scripts/env/check_kcat_envs.sh "$ENV_STATUS_HEALTH" ligandmpnn_env KcatNet apodock

echo "[kcat-pipeline] validating split roots"
VALIDATE_ARGS=()
for split_root in "${SPLIT_ROOTS[@]}"; do
  split_tag="$(basename "$split_root")"
  validate_out="${BOOTSTRAP_DIR}/validate_${split_tag}.json"
  python scripts/prep/01_validate_monomer_split.py \
    --split-root "$split_root" \
    --output "$validate_out"
  VALIDATE_ARGS+=("$validate_out")
done

echo "[kcat-pipeline] building design index"
BUILD_ARGS=(python scripts/prep/02_build_training_index.py --output "${BOOTSTRAP_DIR}/design_index.jsonl")
for split_root in "${SPLIT_ROOTS[@]}"; do
  BUILD_ARGS+=(--split-root "$split_root")
done
for overlay_path in "${METADATA_OVERLAYS[@]}"; do
  BUILD_ARGS+=(--metadata-overlay "$overlay_path")
done
"${BUILD_ARGS[@]}"

echo "[kcat-pipeline] validating Kcat substrate metadata on design index"
python scripts/prep/02_validate_kcat_metadata.py \
  --input "${BOOTSTRAP_DIR}/design_index.jsonl" \
  --stage design_index \
  --output "${BOOTSTRAP_DIR}/validate_kcat_design_index.json"

echo "[kcat-pipeline] computing baseline sequences"
python scripts/prep/03_compute_baselines.py \
  --config "$CONFIG_PATH" \
  --index-path "${BOOTSTRAP_DIR}/design_index.jsonl" \
  --output "${BOOTSTRAP_DIR}/baselines.jsonl" \
  --run-id "$RUN_ID"

echo "[kcat-pipeline] materializing D0 train/test"
python scripts/prep/05_materialize_round0_dataset.py \
  --baselines "${BOOTSTRAP_DIR}/baselines.jsonl" \
  --output-train "${BOOTSTRAP_DIR}/D_0_train.jsonl" \
  --output-test "${BOOTSTRAP_DIR}/D_0_test.jsonl" \
  --output-all "${BOOTSTRAP_DIR}/D_0_all.jsonl"

echo "[kcat-pipeline] validating Kcat substrate metadata on D0 train/test"
python scripts/prep/02_validate_kcat_metadata.py \
  --input "${BOOTSTRAP_DIR}/D_0_train.jsonl" \
  --stage d0_train \
  --output "${BOOTSTRAP_DIR}/validate_kcat_d0_train.json"
python scripts/prep/02_validate_kcat_metadata.py \
  --input "${BOOTSTRAP_DIR}/D_0_test.jsonl" \
  --stage d0_test \
  --output "${BOOTSTRAP_DIR}/validate_kcat_d0_test.json"

echo "[kcat-pipeline] running Kcat Method III experiment"
EXPERIMENT_CMD=(
  python scripts/orchestration/kcat_m3_run_experiment.py
  --config "$CONFIG_PATH"
  --run-id "$RUN_ID"
  --dataset-path "${BOOTSTRAP_DIR}/D_0_train.jsonl"
  --dataset-test-path "${BOOTSTRAP_DIR}/D_0_test.jsonl"
  --output-root "$KCAT_DIR"
  --num-rounds "$NUM_ROUNDS"
  --pool-size "$POOL_SIZE"
  --kcatnet-budget "$KCATNET_BUDGET"
  --graphkcat-budget "$GRAPHKCAT_BUDGET"
  --max-checkpoints "$MAX_CHECKPOINTS"
  --seed "$SEED"
  --step-heartbeat-sec "$STEP_HEARTBEAT_SEC"
  --env-status-json "$ENV_STATUS_HEALTH"
  --require-ready
)
if [[ "$STRICT_GATES" -eq 1 ]]; then
  EXPERIMENT_CMD+=(--strict-gates)
fi
if [[ -n "$EARLY_STOP_OVERFIT_GAP_TOP8" ]]; then
  EXPERIMENT_CMD+=(--early-stop-overfit-gap-top8 "$EARLY_STOP_OVERFIT_GAP_TOP8")
fi
if [[ -n "$EARLY_STOP_OVERFIT_GAP_BEST" ]]; then
  EXPERIMENT_CMD+=(--early-stop-overfit-gap-best "$EARLY_STOP_OVERFIT_GAP_BEST")
fi
EXPERIMENT_CMD+=(--early-stop-patience "$EARLY_STOP_PATIENCE")

"${EXPERIMENT_CMD[@]}"

echo "[kcat-pipeline] complete"
