#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/.venvs/foundry-rf3"
INPUT_ROOT=""
PREPARED_ROOT=""
OUT_ROOT=""
STATES="both"
CKPT_PATH="rf3"
CHECKPOINT_DIRS=""
LOCAL_MSA_ROOT="${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa"
BOLTZ_SRC_PATH=""
MSA_SERVER_URL="http://127.0.0.1:8080/api"
MSA_CACHE_DIR=""
REUSE_CACHE=0
USE_ENV_DB=0
USE_FILTER=0
PAIRING_STRATEGY="greedy"
MSA_BATCH_SIZE=64
MSA_CONCURRENCY=4
MSA_RETRIES=2
MSA_DEPTH=2048
SHARDS=1
MAX_DOCKED_PAIRS=2000
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RF3_GPUS=4
N_RECYCLES=10
DIFFUSION_BATCH_SIZE=5
NUM_STEPS=50
HYDRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/rf3/run_foundry_rf3_local_msa.sh [options]

Required:
  --input-root PATH      RF3 input directory from build_reactzyme_rf3_inputs.py
  --prepared-root PATH   Output directory for JSONs with msa_path
  --out-root PATH        RF3 inference output directory

Options:
  --env-dir PATH         Repo-local virtualenv directory (default: .venvs/foundry-rf3)
  --states VALUE         both|reactant|product (default: both)
  --ckpt-path VALUE      RF3 checkpoint path or registered name (default: rf3)
  --checkpoint-dirs PATH Colon-separated checkpoint search dirs
  --local-msa-root PATH  Shared MMSeqs2 local workspace
  --boltz-src-path PATH  Explicit Boltz client src path for MSA generation
  --msa-server-url URL   Local MMSeqs2 server URL (default: http://127.0.0.1:8080/api)
  --msa-cache-dir PATH   Directory for generated .a3m files
  --cuda-devices LIST    Export CUDA_VISIBLE_DEVICES for RF3 inference (default: 0,1,2,3)
  --rf3-gpus N           Number of GPUs for RF3 inference (default: 4)
  --reuse-cache          Reuse cached .a3m files
  --use-env-db           Request environmental DB
  --use-filter           Enable MMSeqs filtering
  --pairing-strategy X   Pairing strategy for run_mmseqs2 (default: greedy)
  --msa-batch-size N     Sequences per MMSeqs2 batch request (default: 64)
  --msa-concurrency N    Parallel MMSeqs2 batch requests (default: 4)
  --msa-retries N        Retries per MMSeqs2 batch request (default: 2)
  --msa-depth N          Max sequences retained per written A3M (default: 2048; 0 disables)
  --shards N             Output shard count for prepared JSONs (default: 1)
  --max-docked-pairs N   Max docking pairs to prepare/predict (default: 2000; 0 disables)
  --max-examples N       Backward-compatible alias for --max-docked-pairs
  --n-recycles N         RF3 recycle count (default: 10)
  --diffusion-batch-size N
                         RF3 diffusion batch size (default: 5)
  --num-steps N          RF3 diffusion steps (default: 50)
  --hydra-override ARG   Extra Hydra override passed through to RF3, repeatable
  -h, --help             Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"; shift 2 ;;
    --prepared-root)
      PREPARED_ROOT="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --env-dir)
      ENV_DIR="$2"; shift 2 ;;
    --states)
      STATES="$2"; shift 2 ;;
    --ckpt-path)
      CKPT_PATH="$2"; shift 2 ;;
    --checkpoint-dirs)
      CHECKPOINT_DIRS="$2"; shift 2 ;;
    --local-msa-root)
      LOCAL_MSA_ROOT="$2"; shift 2 ;;
    --boltz-src-path)
      BOLTZ_SRC_PATH="$2"; shift 2 ;;
    --msa-server-url)
      MSA_SERVER_URL="$2"; shift 2 ;;
    --msa-cache-dir)
      MSA_CACHE_DIR="$2"; shift 2 ;;
    --cuda-devices)
      CUDA_DEVICES="$2"; shift 2 ;;
    --rf3-gpus)
      RF3_GPUS="$2"; shift 2 ;;
    --reuse-cache)
      REUSE_CACHE=1; shift ;;
    --use-env-db)
      USE_ENV_DB=1; shift ;;
    --use-filter)
      USE_FILTER=1; shift ;;
    --pairing-strategy)
      PAIRING_STRATEGY="$2"; shift 2 ;;
    --msa-batch-size)
      MSA_BATCH_SIZE="$2"; shift 2 ;;
    --msa-concurrency)
      MSA_CONCURRENCY="$2"; shift 2 ;;
    --msa-retries)
      MSA_RETRIES="$2"; shift 2 ;;
    --msa-depth)
      MSA_DEPTH="$2"; shift 2 ;;
    --shards)
      SHARDS="$2"; shift 2 ;;
    --max-docked-pairs)
      MAX_DOCKED_PAIRS="$2"; shift 2 ;;
    --max-examples)
      MAX_DOCKED_PAIRS="$2"; shift 2 ;;
    --n-recycles)
      N_RECYCLES="$2"; shift 2 ;;
    --diffusion-batch-size)
      DIFFUSION_BATCH_SIZE="$2"; shift 2 ;;
    --num-steps)
      NUM_STEPS="$2"; shift 2 ;;
    --hydra-override)
      HYDRA_OVERRIDES+=("$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${INPUT_ROOT}" || -z "${PREPARED_ROOT}" || -z "${OUT_ROOT}" ]]; then
  usage
  exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing python executable: ${ENV_DIR}/bin/python" >&2
  exit 1
fi

if ! [[ "${RF3_GPUS}" =~ ^[0-9]+$ ]] || (( RF3_GPUS <= 0 )); then
  echo "Invalid --rf3-gpus value: ${RF3_GPUS}" >&2
  exit 2
fi

count_csv_items() {
  local csv="$1"
  local IFS=','
  local items=()
  read -r -a items <<< "${csv}"
  echo "${#items[@]}"
}

detect_system_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || true)"
    echo "${count}"
    return 0
  fi
  echo ""
}

if [[ -n "${CUDA_DEVICES}" ]]; then
  visible_gpu_count="$(count_csv_items "${CUDA_DEVICES}")"
  if (( visible_gpu_count != RF3_GPUS )); then
    echo "RF3 requested ${RF3_GPUS} GPUs but CUDA_VISIBLE_DEVICES would expose ${visible_gpu_count}: ${CUDA_DEVICES}" >&2
    echo "Pass matching --rf3-gpus and --cuda-devices values." >&2
    exit 2
  fi
  system_gpu_count="$(detect_system_gpu_count)"
  if [[ -n "${system_gpu_count}" && "${system_gpu_count}" != "0" && "${visible_gpu_count}" -gt "${system_gpu_count}" ]]; then
    echo "Requested CUDA devices ${CUDA_DEVICES}, but only ${system_gpu_count} system GPU(s) are visible." >&2
    exit 1
  fi
fi

PYTHON_BIN="${ENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}/models/foundry/src:${REPO_ROOT}/models/foundry/models/rf3/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${CHECKPOINT_DIRS}" ]]; then
  export FOUNDRY_CHECKPOINT_DIRS="${CHECKPOINT_DIRS}${FOUNDRY_CHECKPOINT_DIRS:+:${FOUNDRY_CHECKPOINT_DIRS}}"
fi

PREP_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/rf3/prepare_reactzyme_rf3_msas.py"
  --input-root "${INPUT_ROOT}"
  --output-root "${PREPARED_ROOT}"
  --states "${STATES}"
  --local-msa-root "${LOCAL_MSA_ROOT}"
  --msa-server-url "${MSA_SERVER_URL}"
  --pairing-strategy "${PAIRING_STRATEGY}"
  --msa-batch-size "${MSA_BATCH_SIZE}"
  --msa-concurrency "${MSA_CONCURRENCY}"
  --msa-retries "${MSA_RETRIES}"
  --msa-depth "${MSA_DEPTH}"
  --shards "${SHARDS}"
  --max-docked-pairs "${MAX_DOCKED_PAIRS}"
)
if [[ -n "${BOLTZ_SRC_PATH}" ]]; then
  PREP_ARGS+=(--boltz-src-path "${BOLTZ_SRC_PATH}")
fi
if [[ -n "${MSA_CACHE_DIR}" ]]; then
  PREP_ARGS+=(--msa-cache-dir "${MSA_CACHE_DIR}")
fi
if [[ "${REUSE_CACHE}" -eq 1 ]]; then
  PREP_ARGS+=(--reuse-cache)
fi
if [[ "${USE_ENV_DB}" -eq 1 ]]; then
  PREP_ARGS+=(--use-env-db)
fi
if [[ "${USE_FILTER}" -eq 1 ]]; then
  PREP_ARGS+=(--use-filter)
fi

echo "[foundry-rf3-run] preparing local MSAs"
"${PREP_ARGS[@]}"

run_state() {
  local state="$1"
  local input_json="${PREPARED_ROOT}/${state}.json"
  local shard_dir="${PREPARED_ROOT}/shards/${state}"
  local state_out="${OUT_ROOT}/${state}"
  local override

  if [[ -d "${shard_dir}" ]] && compgen -G "${shard_dir}/*.json" >/dev/null; then
    local shard_json
    for shard_json in "${shard_dir}"/*.json; do
      local shard_name
      shard_name="$(basename "${shard_json}" .json)"
      local shard_out="${state_out}/${shard_name}"
      local -a shard_cmd=(
        "${PYTHON_BIN}" "${REPO_ROOT}/models/foundry/models/rf3/src/rf3/inference.py"
        "inputs=${shard_json}"
        "out_dir=${shard_out}"
        "ckpt_path=${CKPT_PATH}"
        "devices_per_node=${RF3_GPUS}"
        "num_nodes=1"
        "n_recycles=${N_RECYCLES}"
        "diffusion_batch_size=${DIFFUSION_BATCH_SIZE}"
        "num_steps=${NUM_STEPS}"
      )
      for override in "${HYDRA_OVERRIDES[@]}"; do
        shard_cmd+=("${override}")
      done
      echo "[foundry-rf3-run] running state=${state} shard=${shard_name} inputs=${shard_json} out_dir=${shard_out} gpus=${RF3_GPUS} devices=${CUDA_DEVICES}"
      if [[ -n "${CUDA_DEVICES}" ]]; then
        env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${shard_cmd[@]}"
      else
        "${shard_cmd[@]}"
      fi
    done
    return
  fi

  if [[ ! -f "${input_json}" ]]; then
    echo "Prepared input JSON missing: ${input_json}" >&2
    exit 1
  fi

  local -a cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/models/foundry/models/rf3/src/rf3/inference.py"
    "inputs=${input_json}"
    "out_dir=${state_out}"
    "ckpt_path=${CKPT_PATH}"
    "devices_per_node=${RF3_GPUS}"
    "num_nodes=1"
    "n_recycles=${N_RECYCLES}"
    "diffusion_batch_size=${DIFFUSION_BATCH_SIZE}"
    "num_steps=${NUM_STEPS}"
  )
  for override in "${HYDRA_OVERRIDES[@]}"; do
    cmd+=("${override}")
  done

  echo "[foundry-rf3-run] running state=${state} inputs=${input_json} out_dir=${state_out} gpus=${RF3_GPUS} devices=${CUDA_DEVICES}"
  if [[ -n "${CUDA_DEVICES}" ]]; then
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

case "${STATES}" in
  both)
    run_state reactant
    run_state product
    ;;
  reactant|product)
    run_state "${STATES}"
    ;;
  *)
    echo "Unsupported states value: ${STATES}" >&2
    exit 2
    ;;
esac
