#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="thermogfn_production"
OUTPUT_ROOT=""
CONFIG_PATH="config/m3_default.yaml"
NUM_ROUNDS=8
POOL_SIZE=50000
BIOEMU_BUDGET=512
UMA_BUDGET=64
MAX_CHECKPOINTS=5
SEED=13
GENERATOR_BACKEND="ligandmpnn"
LIGANDMPNN_ENV="ligandmpnn_env"
LIGANDMPNN_MODEL_TYPE="ligand_mpnn"
LIGANDMPNN_CHECKPOINT=""
LIGANDMPNN_USE_ATOM_CONTEXT=1
LIGANDMPNN_PARSE_ZERO_OCC=0
LIGANDMPNN_RUN_HEARTBEAT_SEC=5
LIGANDMPNN_BATCH_SIZE=1
LIGANDMPNN_NUMBER_OF_BATCHES=1
LIGANDMPNN_TEMPERATURE=0.1
STRICT_GATES=1
SKIP_PREFETCH=0
STEP_HEARTBEAT_SEC=30
EARLY_STOP_OVERFIT_GAP_TOP8=""
EARLY_STOP_OVERFIT_GAP_BEST=""
EARLY_STOP_PATIENCE=1
FINAL_NUM_CANDIDATES=512
FINAL_TOP_K=64

SPLIT_ROOTS=("data/rfd3_splits/unconditional_monomer_protrek35m")

usage() {
  cat <<'USAGE'
Usage:
  scripts/orchestration/run_full_production_pipeline.sh [options]

Options:
  --config PATH                 YAML config path (default: config/m3_default.yaml)
  --run-id ID                    Run identifier (default: thermogfn_production)
  --output-root PATH             Output root (default: runs/<run-id>)
  --split-root PATH              Add split root (repeatable)
  --rounds N                     Number of Method III rounds (default: 8)
  --pool-size N                  Candidates per round (default: 50000)
  --bioemu-budget N              BioEmu batch per round (default: 512)
  --uma-budget N                 UMA batch per round (default: 64)
  --max-checkpoints N            Max checkpoint files kept per model family (default: 5)
  --seed N                       Random seed (default: 13)
  --generator-backend NAME       Baseline generator backend: ligandmpnn|adflip (default: ligandmpnn)
  --ligandmpnn-env NAME          Conda env name for LigandMPNN (default: ligandmpnn_env)
  --ligandmpnn-model-type NAME   LigandMPNN model_type (default: ligand_mpnn)
  --ligandmpnn-checkpoint PATH   Optional LigandMPNN checkpoint path override
  --ligandmpnn-use-atom-context N  LigandMPNN atom context flag (default: 1)
  --ligandmpnn-parse-zero-occ N  Parse zero-occupancy atoms (default: 0)
  --ligandmpnn-run-heartbeat-sec N  Baseline LigandMPNN run heartbeat seconds (default: 5)
  --ligandmpnn-batch-size N        LigandMPNN run.py batch_size (default: 1)
  --ligandmpnn-number-of-batches N LigandMPNN run.py number_of_batches (default: 1)
  --ligandmpnn-temperature X       LigandMPNN sampling temperature (default: 0.1)
  --adflip-device DEVICE         Baseline ADFLIP device (default: cuda:0)
  --adflip-steps N               Baseline ADFLIP steps (default: 32)
  --adflip-threshold X           Baseline ADFLIP threshold (default: 0.9)
  --step-heartbeat-sec N         Step heartbeat for orchestration scripts (default: 30)
  --early-stop-overfit-gap-top8 X  Early-stop threshold for train-test top8 reward gap
  --early-stop-overfit-gap-best X  Early-stop threshold for train-test best reward gap
  --early-stop-patience N        Consecutive breach rounds before stop (default: 1)
  --final-num-candidates N       Final inference candidates on test seeds (default: 512)
  --final-top-k N                Final top-k selected candidates (default: 64)
  --no-strict-gates              Disable strict round gates
  --skip-prefetch                Skip oracle artifact prefetch
  -h, --help                     Show help
USAGE
}

_extract_cfg_defaults() {
  python - "$REPO_ROOT" "$CONFIG_PATH" <<'PY'
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
if not cfg_path.is_absolute():
    cfg_path = repo_root / cfg_path

try:
    import yaml
except Exception:
    print("{}")
    raise SystemExit(0)

if not cfg_path.exists():
    print("{}")
    raise SystemExit(0)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
if not isinstance(cfg, dict):
    print("{}")
    raise SystemExit(0)

def g(path, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

out = {
    "RUN_ID": g("run.run_id", "thermogfn_production"),
    "SEED": g("run.seed", 13),
    "STRICT_GATES": 1 if bool(g("run.strict_gates", True)) else 0,
    "NUM_ROUNDS": g("method3.rounds", 8),
    "POOL_SIZE": g("round.pool_size", 50000),
    "BIOEMU_BUDGET": g("round.bioemu_budget", 512),
    "UMA_BUDGET": g("round.uma_budget", 64),
    "MAX_CHECKPOINTS": g("round.max_checkpoints", 5),
    "STEP_HEARTBEAT_SEC": g("round.step_heartbeat_sec", 30),
    "EARLY_STOP_OVERFIT_GAP_TOP8": g("method3.early_stop.overfit_gap_top8_mean_reward", ""),
    "EARLY_STOP_OVERFIT_GAP_BEST": g("method3.early_stop.overfit_gap_best_reward", ""),
    "EARLY_STOP_PATIENCE": g("method3.early_stop.patience_rounds", 1),
    "GENERATOR_BACKEND": g("generator.backend", "ligandmpnn"),
    "LIGANDMPNN_ENV": g("generator.ligandmpnn.env_name", "ligandmpnn_env"),
    "LIGANDMPNN_MODEL_TYPE": g("generator.ligandmpnn.model_type", "ligand_mpnn"),
    "LIGANDMPNN_CHECKPOINT": g("generator.ligandmpnn.checkpoint", ""),
    "LIGANDMPNN_USE_ATOM_CONTEXT": g("generator.ligandmpnn.use_atom_context", 1),
    "LIGANDMPNN_PARSE_ZERO_OCC": g("generator.ligandmpnn.parse_atoms_with_zero_occupancy", 0),
    "LIGANDMPNN_RUN_HEARTBEAT_SEC": g("generator.ligandmpnn.run_heartbeat_sec", 5),
    "LIGANDMPNN_BATCH_SIZE": g("generator.ligandmpnn.batch_size", 1),
    "LIGANDMPNN_NUMBER_OF_BATCHES": g("generator.ligandmpnn.number_of_batches", 1),
    "LIGANDMPNN_TEMPERATURE": g("generator.ligandmpnn.temperature", 0.1),
    "FINAL_NUM_CANDIDATES": g("inference.final.num_candidates", 512),
    "FINAL_TOP_K": g("inference.final.top_k", 64),
}
print(json.dumps(out))
PY
}

_extract_cfg_split_roots() {
  python - "$REPO_ROOT" "$CONFIG_PATH" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
if not cfg_path.is_absolute():
    cfg_path = repo_root / cfg_path
if not cfg_path.exists():
    raise SystemExit(0)
try:
    import yaml
except Exception:
    raise SystemExit(0)
cfg = yaml.safe_load(cfg_path.read_text()) or {}
roots = cfg.get("data", {}).get("split_roots", [])
if isinstance(roots, list):
    for item in roots:
        if item is None:
            continue
        print(str(item))
PY
}

# Pre-scan CLI for --config so defaults can come from the selected config file.
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--config" ]]; then
    j=$((i+1))
    CONFIG_PATH="${!j}"
    break
  fi
done

CFG_DEFAULTS_JSON="$(_extract_cfg_defaults)"
if [[ "$CFG_DEFAULTS_JSON" != "{}" ]]; then
  eval "$(
    python - <<'PY' "$CFG_DEFAULTS_JSON"
import json
import shlex
import sys
d = json.loads(sys.argv[1])
for k, v in d.items():
    if v is None:
        v = ""
    print(f"{k}={shlex.quote(str(v))}")
PY
  )"
fi
CFG_SPLIT_ROOTS="$(_extract_cfg_split_roots || true)"
if [[ -n "$CFG_SPLIT_ROOTS" ]]; then
  mapfile -t SPLIT_ROOTS <<<"$CFG_SPLIT_ROOTS"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --split-root)
      if [[ "${SPLIT_ROOTS[*]}" == "data/rfd3_splits/unconditional_monomer_protrek35m" && ${#SPLIT_ROOTS[@]} -eq 1 ]]; then
        SPLIT_ROOTS=()
      fi
      SPLIT_ROOTS+=("$2")
      shift 2
      ;;
    --rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --pool-size)
      POOL_SIZE="$2"
      shift 2
      ;;
    --bioemu-budget)
      BIOEMU_BUDGET="$2"
      shift 2
      ;;
    --uma-budget)
      UMA_BUDGET="$2"
      shift 2
      ;;
    --max-checkpoints)
      MAX_CHECKPOINTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --generator-backend)
      GENERATOR_BACKEND="$2"
      shift 2
      ;;
    --ligandmpnn-env)
      LIGANDMPNN_ENV="$2"
      shift 2
      ;;
    --ligandmpnn-model-type)
      LIGANDMPNN_MODEL_TYPE="$2"
      shift 2
      ;;
    --ligandmpnn-checkpoint)
      LIGANDMPNN_CHECKPOINT="$2"
      shift 2
      ;;
    --ligandmpnn-use-atom-context)
      LIGANDMPNN_USE_ATOM_CONTEXT="$2"
      shift 2
      ;;
    --ligandmpnn-parse-zero-occ)
      LIGANDMPNN_PARSE_ZERO_OCC="$2"
      shift 2
      ;;
    --ligandmpnn-run-heartbeat-sec)
      LIGANDMPNN_RUN_HEARTBEAT_SEC="$2"
      shift 2
      ;;
    --ligandmpnn-batch-size)
      LIGANDMPNN_BATCH_SIZE="$2"
      shift 2
      ;;
    --ligandmpnn-number-of-batches)
      LIGANDMPNN_NUMBER_OF_BATCHES="$2"
      shift 2
      ;;
    --ligandmpnn-temperature)
      LIGANDMPNN_TEMPERATURE="$2"
      shift 2
      ;;
    --adflip-device)
      ADFLIP_DEVICE="$2"
      shift 2
      ;;
    --adflip-steps)
      ADFLIP_STEPS="$2"
      shift 2
      ;;
    --adflip-threshold)
      ADFLIP_THRESHOLD="$2"
      shift 2
      ;;
    --step-heartbeat-sec)
      STEP_HEARTBEAT_SEC="$2"
      shift 2
      ;;
    --early-stop-overfit-gap-top8)
      EARLY_STOP_OVERFIT_GAP_TOP8="$2"
      shift 2
      ;;
    --early-stop-overfit-gap-best)
      EARLY_STOP_OVERFIT_GAP_BEST="$2"
      shift 2
      ;;
    --early-stop-patience)
      EARLY_STOP_PATIENCE="$2"
      shift 2
      ;;
    --final-num-candidates)
      FINAL_NUM_CANDIDATES="$2"
      shift 2
      ;;
    --final-top-k)
      FINAL_TOP_K="$2"
      shift 2
      ;;
    --no-strict-gates)
      STRICT_GATES=0
      shift
      ;;
    --skip-prefetch)
      SKIP_PREFETCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ADFLIP_DEVICE="${ADFLIP_DEVICE:-cuda:0}"
ADFLIP_STEPS="${ADFLIP_STEPS:-32}"
ADFLIP_THRESHOLD="${ADFLIP_THRESHOLD:-0.9}"

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="runs/${RUN_ID}"
fi

BOOTSTRAP_DIR="${OUTPUT_ROOT}/bootstrap"
M3_DIR="${OUTPUT_ROOT}/m3"
INFER_DIR="${OUTPUT_ROOT}/infer"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$BOOTSTRAP_DIR" "$M3_DIR" "$INFER_DIR" "$LOG_DIR"

ENV_STATUS="${OUTPUT_ROOT}/env_status.json"
ENV_STATUS_HEALTH="${OUTPUT_ROOT}/env_status_health.json"

echo "[pipeline] checking conda environments"
./scripts/env/check_envs.sh "$ENV_STATUS"
RUN_HEALTH_CHECKS=1 ./scripts/env/check_envs.sh "$ENV_STATUS_HEALTH"

if [[ "$SKIP_PREFETCH" -eq 0 ]]; then
  echo "[pipeline] prefetching oracle artifacts"
  ./scripts/env/prefetch_production_oracles.sh 2>&1 | tee "${LOG_DIR}/prefetch_production_oracles.log"
fi

echo "[pipeline] validating split roots"
for split_root in "${SPLIT_ROOTS[@]}"; do
  split_name="$(basename "$split_root")"
  python scripts/prep/01_validate_monomer_split.py \
    --split-root "$split_root" \
    --output "${BOOTSTRAP_DIR}/validate_${split_name}.json"
done

echo "[pipeline] building design index"
INDEX_PATH="${BOOTSTRAP_DIR}/design_index.jsonl"
index_cmd=(python scripts/prep/02_build_training_index.py --output "$INDEX_PATH" --allow-missing)
for split_root in "${SPLIT_ROOTS[@]}"; do
  index_cmd+=(--split-root "$split_root")
done
"${index_cmd[@]}"

echo "[pipeline] computing baseline sequences with generator backend=${GENERATOR_BACKEND}"
BASELINES_PATH="${BOOTSTRAP_DIR}/baselines.jsonl"
baseline_cmd=(python scripts/prep/03_compute_baselines.py \
  --config "$CONFIG_PATH" \
  --index-path "$INDEX_PATH" \
  --output "$BASELINES_PATH" \
  --run-id "$RUN_ID" \
  --generator-backend "$GENERATOR_BACKEND" \
  --ligandmpnn-env "$LIGANDMPNN_ENV" \
  --ligandmpnn-model-type "$LIGANDMPNN_MODEL_TYPE" \
  --ligandmpnn-use-atom-context "$LIGANDMPNN_USE_ATOM_CONTEXT" \
  --ligandmpnn-parse-atoms-with-zero-occupancy "$LIGANDMPNN_PARSE_ZERO_OCC" \
  --ligandmpnn-batch-size "$LIGANDMPNN_BATCH_SIZE" \
  --ligandmpnn-number-of-batches "$LIGANDMPNN_NUMBER_OF_BATCHES" \
  --ligandmpnn-temperature "$LIGANDMPNN_TEMPERATURE" \
  --ligandmpnn-run-heartbeat-sec "$LIGANDMPNN_RUN_HEARTBEAT_SEC" \
  --device "$ADFLIP_DEVICE" \
  --steps "$ADFLIP_STEPS" \
  --threshold "$ADFLIP_THRESHOLD")
if [[ -n "$LIGANDMPNN_CHECKPOINT" ]]; then
  baseline_cmd+=(--ligandmpnn-checkpoint "$LIGANDMPNN_CHECKPOINT")
fi
"${baseline_cmd[@]}"

echo "[pipeline] materializing D0 train/test datasets"
D0_TRAIN="${BOOTSTRAP_DIR}/D_0_train.jsonl"
D0_TEST="${BOOTSTRAP_DIR}/D_0_test.jsonl"
D0_ALL="${BOOTSTRAP_DIR}/D_0_all.jsonl"
python scripts/prep/05_materialize_round0_dataset.py \
  --baselines "$BASELINES_PATH" \
  --output-train "$D0_TRAIN" \
  --output-test "$D0_TEST" \
  --output-all "$D0_ALL"

echo "[pipeline] running Method III training"
m3_cmd=(
  python scripts/orchestration/m3_run_experiment.py
  --config "$CONFIG_PATH"
  --run-id "$RUN_ID"
  --dataset-path "$D0_TRAIN"
  --dataset-test-path "$D0_TEST"
  --output-root "$M3_DIR"
  --num-rounds "$NUM_ROUNDS"
  --pool-size "$POOL_SIZE"
  --bioemu-budget "$BIOEMU_BUDGET"
  --uma-budget "$UMA_BUDGET"
  --max-checkpoints "$MAX_CHECKPOINTS"
  --seed "$SEED"
  --step-heartbeat-sec "$STEP_HEARTBEAT_SEC"
  --env-status-json "$ENV_STATUS_HEALTH"
  --require-ready
)
if [[ -n "$EARLY_STOP_OVERFIT_GAP_TOP8" ]]; then
  m3_cmd+=(--early-stop-overfit-gap-top8 "$EARLY_STOP_OVERFIT_GAP_TOP8")
fi
if [[ -n "$EARLY_STOP_OVERFIT_GAP_BEST" ]]; then
  m3_cmd+=(--early-stop-overfit-gap-best "$EARLY_STOP_OVERFIT_GAP_BEST")
fi
m3_cmd+=(--early-stop-patience "$EARLY_STOP_PATIENCE")
if [[ "$STRICT_GATES" -eq 1 ]]; then
  m3_cmd+=(--strict-gates)
fi
"${m3_cmd[@]}"

echo "[pipeline] collecting final round artifacts"
LAST_ROUND="$(python - <<'PY' "$M3_DIR/experiment_history.json"
import json,sys
h=json.load(open(sys.argv[1]))
ok=[r["round"] for r in h if r.get("returncode")==0]
if not ok:
    raise SystemExit(1)
print(max(ok))
PY
)"
ROUND_TAG="$(printf "%03d" "$LAST_ROUND")"
STUDENT_CKPT="${M3_DIR}/round_${ROUND_TAG}/models/student_round_${LAST_ROUND}.ckpt"
FUSED_LAST="${M3_DIR}/round_${ROUND_TAG}/data/fused_scored_round_${LAST_ROUND}.jsonl"

echo "[pipeline] running post-training inference and ranking"
CANDIDATES="${INFER_DIR}/candidates.jsonl"
SCORED_DIR="${INFER_DIR}/scored"
TOP_PATH="${INFER_DIR}/top${FINAL_TOP_K}.jsonl"
CARD_PATH="${INFER_DIR}/top_candidate_card.md"

python scripts/infer/generate_unconditioned.py \
  --student-ckpt "$STUDENT_CKPT" \
  --seed-dataset "$D0_TEST" \
  --output-path "$CANDIDATES" \
  --run-id "${RUN_ID}_infer" \
  --round-id "$LAST_ROUND" \
  --num-candidates "$FINAL_NUM_CANDIDATES" \
  --seed "$SEED"

python scripts/train/m1_train_simul_mf.py \
  --input-path "$CANDIDATES" \
  --output-dir "$SCORED_DIR"

python scripts/infer/rescore_and_select.py \
  --input-path "${SCORED_DIR}/m1_fused.jsonl" \
  --output-path "$TOP_PATH" \
  --top-k "$FINAL_TOP_K"

TOP_CANDIDATE_ID="$(python - <<'PY' "$TOP_PATH"
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
if not rows:
    raise SystemExit(1)
print(rows[0]["candidate_id"])
PY
)"

python scripts/infer/report_candidate_card.py \
  --input-path "$TOP_PATH" \
  --candidate-id "$TOP_CANDIDATE_ID" \
  --output "$CARD_PATH"

python scripts/eval/eval_large_protein_breakdown.py \
  --input-path "$FUSED_LAST" \
  --output "${M3_DIR}/round_${ROUND_TAG}/metrics/large_protein_breakdown.json"

SUMMARY_PATH="${OUTPUT_ROOT}/pipeline_summary.json"
python - <<'PY' "$SUMMARY_PATH" "$RUN_ID" "$CONFIG_PATH" "$OUTPUT_ROOT" "$D0_TRAIN" "$D0_TEST" "$M3_DIR/experiment_history.json" "$STUDENT_CKPT" "$TOP_PATH" "$CARD_PATH" "$FINAL_NUM_CANDIDATES" "$FINAL_TOP_K"
import json,sys
out,run_id,config_path,root,d0_train,d0_test,hist,student,top,card,final_n,final_k=sys.argv[1:]
payload={
  "run_id": run_id,
  "config_path": config_path,
  "output_root": root,
  "d0_train": d0_train,
  "d0_test": d0_test,
  "experiment_history": hist,
  "student_checkpoint": student,
  "top_candidates": top,
  "top_candidate_card": card,
  "final_num_candidates": int(final_n),
  "final_top_k": int(final_k),
}
json.dump(payload, open(out,"w"), indent=2, sort_keys=True)
print(out)
PY

echo "[pipeline] complete"
echo "[pipeline] summary: ${SUMMARY_PATH}"
