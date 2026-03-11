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
MSA_BACKEND="local_direct"
MSA_SERVER_URL="http://127.0.0.1:8080/api"
MSA_CACHE_DIR=""
REUSE_CACHE=0
SKIP_MSA_PREP=0
USE_ENV_DB=0
USE_FILTER=1
PAIRING_STRATEGY="greedy"
MSA_BATCH_SIZE=64
MSA_CONCURRENCY=8
MSA_RETRIES=2
MSA_DEPTH=2048
MSA_THREADS_PER_JOB=0
MMSEQS_MAX_SEQS=4096
MMSEQS_NUM_ITERATIONS=3
SHARDS=1
MAX_DOCKED_PAIRS=2000
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RF3_GPUS=4
RF3_LAUNCH_MODE="auto"
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
  --msa-backend VALUE    local_direct|server (default: local_direct)
  --boltz-src-path PATH  Explicit Boltz client src path for MSA generation
  --msa-server-url URL   Local MMSeqs2 server URL (default: http://127.0.0.1:8080/api)
  --msa-cache-dir PATH   Directory for generated .a3m files
  --skip-msa-prep        Reuse an existing prepared-root and skip MSA generation
  --cuda-devices LIST    Export CUDA_VISIBLE_DEVICES for RF3 inference (default: 0,1,2,3)
  --rf3-gpus N           Number of GPUs for RF3 inference (default: 4)
  --rf3-launch-mode X    auto|sharded_single|ddp (default: auto)
  --reuse-cache          Reuse cached .a3m files
  --use-env-db           Request environmental DB
  --use-filter           Enable MMSeqs filtering (default)
  --no-use-filter        Disable MMSeqs filtering
  --pairing-strategy X   Pairing strategy for run_mmseqs2 (default: greedy)
  --msa-batch-size N     Sequences per MMSeqs2 batch request (default: 64)
  --msa-concurrency N    Parallel MMSeqs2 batch requests (default: 8)
  --msa-retries N        Retries per MMSeqs2 batch request (default: 2)
  --msa-depth N          Max sequences retained per written A3M (default: 2048; 0 disables)
  --msa-threads-per-job N
                         Threads per local_direct MMSeqs chunk (default: auto)
  --mmseqs-max-seqs N    Tune MMSeqs search/gpuserver max-seqs (default: 4096)
  --mmseqs-num-iterations N
                         Tune MMSeqs search num-iterations (default: 3)
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
    --msa-backend)
      MSA_BACKEND="$2"; shift 2 ;;
    --boltz-src-path)
      BOLTZ_SRC_PATH="$2"; shift 2 ;;
    --msa-server-url)
      MSA_SERVER_URL="$2"; shift 2 ;;
    --msa-cache-dir)
      MSA_CACHE_DIR="$2"; shift 2 ;;
    --skip-msa-prep)
      SKIP_MSA_PREP=1; shift ;;
    --cuda-devices)
      CUDA_DEVICES="$2"; shift 2 ;;
    --rf3-gpus)
      RF3_GPUS="$2"; shift 2 ;;
    --rf3-launch-mode)
      RF3_LAUNCH_MODE="$2"; shift 2 ;;
    --reuse-cache)
      REUSE_CACHE=1; shift ;;
    --use-env-db)
      USE_ENV_DB=1; shift ;;
    --use-filter)
      USE_FILTER=1; shift ;;
    --no-use-filter)
      USE_FILTER=0; shift ;;
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
    --msa-threads-per-job)
      MSA_THREADS_PER_JOB="$2"; shift 2 ;;
    --mmseqs-max-seqs)
      MMSEQS_MAX_SEQS="$2"; shift 2 ;;
    --mmseqs-num-iterations)
      MMSEQS_NUM_ITERATIONS="$2"; shift 2 ;;
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

case "${RF3_LAUNCH_MODE}" in
  auto|sharded_single|ddp) ;;
  *)
    echo "Unsupported --rf3-launch-mode value: ${RF3_LAUNCH_MODE}" >&2
    exit 2 ;;
esac

if [[ "${RF3_LAUNCH_MODE}" == "auto" ]]; then
  if (( RF3_GPUS > 1 )); then
    RF3_LAUNCH_MODE="sharded_single"
  else
    RF3_LAUNCH_MODE="ddp"
  fi
fi

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_DEVICES}"
if (( RF3_GPUS > 1 )) && [[ "${RF3_LAUNCH_MODE}" == "sharded_single" ]] && (( SHARDS < RF3_GPUS )); then
  SHARDS="${RF3_GPUS}"
  echo "[foundry-rf3-run] increasing --shards to ${SHARDS} to match sharded single-GPU RF3 launch"
fi
if [[ "${RF3_LAUNCH_MODE}" == "sharded_single" ]] && (( ${#HYDRA_OVERRIDES[@]} > 0 )); then
  echo "Hydra overrides are not supported with --rf3-launch-mode sharded_single." >&2
  echo "Use --rf3-launch-mode ddp if you need raw Hydra overrides." >&2
  exit 2
fi

PYTHON_BIN="${ENV_DIR}/bin/python"
INPUT_ROOT="$(realpath "${INPUT_ROOT}")"
PREPARED_ROOT="$(realpath -m "${PREPARED_ROOT}")"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
LOCAL_MSA_ROOT="$(realpath -m "${LOCAL_MSA_ROOT}")"
if [[ -n "${MSA_CACHE_DIR}" ]]; then
  MSA_CACHE_DIR="$(realpath -m "${MSA_CACHE_DIR}")"
fi

EFFECTIVE_MSA_DIR="${PREPARED_ROOT}/msas"
if [[ -n "${MSA_CACHE_DIR}" ]]; then
  EFFECTIVE_MSA_DIR="${MSA_CACHE_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}/models/foundry/src:${REPO_ROOT}/models/foundry/models/rf3/src${PYTHONPATH:+:${PYTHONPATH}}"
DEFAULT_RF3_CKPT="${REPO_ROOT}/weights/rf3_foundry_01_24_latest_remapped.ckpt"
if [[ "${CKPT_PATH}" == "rf3" && -f "${DEFAULT_RF3_CKPT}" ]]; then
  echo "[foundry-rf3-run] using repo-local RF3 checkpoint ${DEFAULT_RF3_CKPT}"
  CKPT_PATH="${DEFAULT_RF3_CKPT}"
fi
if [[ -n "${CHECKPOINT_DIRS}" ]]; then
  export FOUNDRY_CHECKPOINT_DIRS="${CHECKPOINT_DIRS}${FOUNDRY_CHECKPOINT_DIRS:+:${FOUNDRY_CHECKPOINT_DIRS}}"
elif [[ -d "${REPO_ROOT}/weights" ]]; then
  export FOUNDRY_CHECKPOINT_DIRS="${REPO_ROOT}/weights${FOUNDRY_CHECKPOINT_DIRS:+:${FOUNDRY_CHECKPOINT_DIRS}}"
fi

case "${MSA_BACKEND}" in
  local_direct|server) ;;
  *)
    echo "Unsupported --msa-backend value: ${MSA_BACKEND}" >&2
    exit 2 ;;
esac

if [[ "${MSA_BACKEND}" == "server" ]]; then
  MMSEQS_SERVER_RESTART_CMD=(
    bash "${REPO_ROOT}/scripts/env/start_local_mmseqs2_uniref30_server.sh"
    --msa-root "${LOCAL_MSA_ROOT}"
    --uniref30-root "/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302"
    --cleanup-uniref100
    --cuda-devices "${CUDA_DEVICES}"
    --mmseqs-max-seqs "${MMSEQS_MAX_SEQS}"
    --mmseqs-num-iterations "${MMSEQS_NUM_ITERATIONS}"
  )
  echo "[foundry-rf3-run] ensuring tuned MMSeqs2 server config (backend=server max_seqs=${MMSEQS_MAX_SEQS} num_iterations=${MMSEQS_NUM_ITERATIONS})"
  if [[ -f "${LOCAL_MSA_ROOT}/run/mmseqs-server.pid" ]]; then
    old_pid="$(cat "${LOCAL_MSA_ROOT}/run/mmseqs-server.pid" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      kill "${old_pid}" || true
      sleep 2
    fi
    rm -f "${LOCAL_MSA_ROOT}/run/mmseqs-server.pid"
  fi
  "${MMSEQS_SERVER_RESTART_CMD[@]}"
else
  echo "[foundry-rf3-run] using direct local MMSeqs2 GPU backend (max_seqs=${MMSEQS_MAX_SEQS} num_iterations=${MMSEQS_NUM_ITERATIONS})"
fi

PREP_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/rf3/prepare_reactzyme_rf3_msas.py"
  --input-root "${INPUT_ROOT}"
  --output-root "${PREPARED_ROOT}"
  --states "${STATES}"
  --local-msa-root "${LOCAL_MSA_ROOT}"
  --msa-backend "${MSA_BACKEND}"
  --msa-server-url "${MSA_SERVER_URL}"
  --pairing-strategy "${PAIRING_STRATEGY}"
  --msa-batch-size "${MSA_BATCH_SIZE}"
  --msa-concurrency "${MSA_CONCURRENCY}"
  --msa-retries "${MSA_RETRIES}"
  --msa-depth "${MSA_DEPTH}"
  --msa-threads-per-job "${MSA_THREADS_PER_JOB}"
  --cuda-devices "${CUDA_DEVICES}"
  --mmseqs-max-seqs "${MMSEQS_MAX_SEQS}"
  --mmseqs-num-iterations "${MMSEQS_NUM_ITERATIONS}"
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
if [[ "${SKIP_MSA_PREP}" -eq 0 ]]; then
  if [[ -n "${CUDA_DEVICES}" ]]; then
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PREP_ARGS[@]}"
  else
    "${PREP_ARGS[@]}"
  fi
elif [[ ! -f "${PREPARED_ROOT}/summary.json" ]]; then
  echo "Missing prepared RF3 inputs at ${PREPARED_ROOT}; cannot use --skip-msa-prep" >&2
  exit 1
else
  echo "[foundry-rf3-run] skipping MSA prep and reusing prepared inputs at ${PREPARED_ROOT}"
fi

if [[ "${EFFECTIVE_MSA_DIR}" != "${PREPARED_ROOT}/msas" && -d "${EFFECTIVE_MSA_DIR}" ]]; then
  mkdir -p "${PREPARED_ROOT}"
  ln -sfn "${EFFECTIVE_MSA_DIR}" "${PREPARED_ROOT}/msas"
fi

run_single_shard() {
  local shard_json="$1"
  local shard_out="$2"
  local gpu_id="$3"
  local log_path="$4"
  local override
  local -a shard_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/rf3/run_foundry_rf3_direct.py"
    --inputs "${shard_json}"
    --out-dir "${shard_out}"
    --ckpt-path "${CKPT_PATH}"
    --devices-per-node 1
    --num-nodes 1
    --n-recycles "${N_RECYCLES}"
    --diffusion-batch-size "${DIFFUSION_BATCH_SIZE}"
    --num-steps "${NUM_STEPS}"
  )
  if [[ -d "${EFFECTIVE_MSA_DIR}" ]]; then
    shard_cmd+=(--local-msa-dirs "${EFFECTIVE_MSA_DIR}")
  fi
  env \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    HYDRA_FULL_ERROR=1 \
    "${shard_cmd[@]}" >"${log_path}" 2>&1
}

run_state_sharded_single() {
  local state="$1"
  local shard_dir="${PREPARED_ROOT}/shards/${state}"
  local state_out="${OUT_ROOT}/${state}"
  local log_dir="${state_out}/logs"
  local -a shard_jsons=()
  local -a running_pids=()
  local -A pid_to_gpu=()
  local -A pid_to_shard=()
  local -A pid_to_log=()
  local total_shards=0
  local completed=0

  mkdir -p "${state_out}" "${log_dir}"
  if [[ -d "${shard_dir}" ]] && compgen -G "${shard_dir}/*.json" >/dev/null; then
    mapfile -t shard_jsons < <(printf '%s\n' "${shard_dir}"/*.json | sort)
  else
    local input_json="${PREPARED_ROOT}/${state}.json"
    if [[ ! -f "${input_json}" ]]; then
      echo "Prepared input JSON missing: ${input_json}" >&2
      exit 1
    fi
    shard_jsons=("${input_json}")
  fi
  total_shards="${#shard_jsons[@]}"

  prune_finished_jobs() {
    local -a still_running=()
    local pid exit_code gpu_id shard_name log_path
    for pid in "${running_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        still_running+=("${pid}")
        continue
      fi
      if wait "${pid}"; then
        exit_code=0
      else
        exit_code=$?
      fi
      gpu_id="${pid_to_gpu[${pid}]}"
      shard_name="${pid_to_shard[${pid}]}"
      log_path="${pid_to_log[${pid}]}"
      if (( exit_code != 0 )); then
        echo "[foundry-rf3-run] shard failed state=${state} shard=${shard_name} gpu=${gpu_id} log=${log_path}" >&2
        if [[ -f "${log_path}" ]]; then
          tail -n 80 "${log_path}" >&2 || true
        fi
        for pid in "${still_running[@]}"; do
          kill "${pid}" 2>/dev/null || true
        done
        exit "${exit_code}"
      fi
      completed=$((completed + 1))
      echo "[foundry-rf3-run] shard completed state=${state} shard=${shard_name} gpu=${gpu_id} completed=${completed}/${total_shards}"
      unset 'pid_to_gpu[$pid]' 'pid_to_shard[$pid]' 'pid_to_log[$pid]'
    done
    running_pids=("${still_running[@]}")
    if (( total_shards > 0 )); then
      echo "[foundry-rf3-run] state=${state} progress completed=${completed}/${total_shards} running=${#running_pids[@]}"
    fi
  }

  local idx=0 shard_json shard_name shard_out gpu_slot gpu_id log_path
  for shard_json in "${shard_jsons[@]}"; do
    while (( ${#running_pids[@]} >= RF3_GPUS )); do
      sleep 2
      prune_finished_jobs
    done
    shard_name="$(basename "${shard_json}" .json)"
    shard_out="${state_out}/${shard_name}"
    gpu_slot=$(( idx % RF3_GPUS ))
    gpu_id="${CUDA_DEVICE_ARRAY[${gpu_slot}]}"
    log_path="${log_dir}/${shard_name}.log"
    echo "[foundry-rf3-run] launching state=${state} shard=${shard_name} gpu=${gpu_id} log=${log_path}"
    run_single_shard "${shard_json}" "${shard_out}" "${gpu_id}" "${log_path}" &
    local pid=$!
    running_pids+=("${pid}")
    pid_to_gpu["${pid}"]="${gpu_id}"
    pid_to_shard["${pid}"]="${shard_name}"
    pid_to_log["${pid}"]="${log_path}"
    idx=$((idx + 1))
  done

  while (( ${#running_pids[@]} > 0 )); do
    sleep 2
    prune_finished_jobs
  done
}

run_state() {
  local state="$1"
  local input_json="${PREPARED_ROOT}/${state}.json"
  local shard_dir="${PREPARED_ROOT}/shards/${state}"
  local state_out="${OUT_ROOT}/${state}"
  local override

  if [[ "${RF3_LAUNCH_MODE}" == "sharded_single" ]]; then
    run_state_sharded_single "${state}"
    return
  fi

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
