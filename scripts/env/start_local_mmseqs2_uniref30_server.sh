#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURE_SCRIPT="${SCRIPT_DIR}/configure_local_mmseqs2_uniref30.sh"

MSA_ROOT="${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa"
UNIREF30_ROOT="/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302"
CONFIG_PATH=""
CLEANUP_UNIREF100=0
GPU_SERVER="1"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
FOREGROUND=0
HOST=""
PORT=""
LOCAL_WORKERS=""
PARALLEL_DATABASES=""
PARALLEL_STAGES=""
LOG_FILE=""
PID_FILE=""
MMSEQS_MAX_SEQS="4096"
MMSEQS_NUM_ITERATIONS="3"
DISABLE_MMSEQS_TUNE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/env/start_local_mmseqs2_uniref30_server.sh [options]

Generate a local MMSeqs2 server config that points at the existing UniRef30 DB,
then launch `mmseqs-server -local` with the GPU backend enabled.

Options:
  --msa-root PATH           Local MSA workspace root
  --uniref30-root PATH      Directory containing uniref30_2302_db*
  --config PATH             Output config path (default: <msa-root>/config.uniref30.json)
  --cleanup-uniref100       Remove accidental local uniref100_db* artifacts first
  --gpu-server N            Forward --gpu-server to sibling start script
  --cuda-devices LIST       Export CUDA_VISIBLE_DEVICES for mmseqs-server (default: 0,1,2,3)
  --foreground              Run attached instead of background mode
  --host HOST               Override host in generated config
  --port PORT               Override port in generated config and start command
  --local-workers N         Override generated local worker count
  --parallel-databases N    Override generated parallel-databases count
  --parallel-stages         Enable generated ColabFold parallel stages
  --no-parallel-stages      Disable generated ColabFold parallel stages
  --mmseqs-max-seqs N       Tune MMSeqs search/gpuserver max-seqs (default: 4096)
  --mmseqs-num-iterations N Tune MMSeqs search num-iterations (default: 3)
  --disable-mmseqs-tune     Disable the MMSeqs tuned wrapper and use the raw binary
  --log-file PATH           Log file for detached server mode (default: <msa-root>/logs/mmseqs-server.log)
  --pid-file PATH           PID file for detached server mode (default: <msa-root>/run/mmseqs-server.pid)
  -h, --help                Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --msa-root)
      MSA_ROOT="$2"; shift 2 ;;
    --uniref30-root)
      UNIREF30_ROOT="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --cleanup-uniref100)
      CLEANUP_UNIREF100=1; shift ;;
    --gpu-server)
      GPU_SERVER="$2"; shift 2 ;;
    --cuda-devices)
      CUDA_DEVICES="$2"; shift 2 ;;
    --foreground)
      FOREGROUND=1; shift ;;
    --host)
      HOST="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --local-workers)
      LOCAL_WORKERS="$2"; shift 2 ;;
    --parallel-databases)
      PARALLEL_DATABASES="$2"; shift 2 ;;
    --parallel-stages)
      PARALLEL_STAGES="true"; shift ;;
    --no-parallel-stages)
      PARALLEL_STAGES="false"; shift ;;
    --mmseqs-max-seqs)
      MMSEQS_MAX_SEQS="$2"; shift 2 ;;
    --mmseqs-num-iterations)
      MMSEQS_NUM_ITERATIONS="$2"; shift 2 ;;
    --disable-mmseqs-tune)
      DISABLE_MMSEQS_TUNE=1; shift ;;
    --log-file)
      LOG_FILE="$2"; shift 2 ;;
    --pid-file)
      PID_FILE="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIGURE_SCRIPT}" ]]; then
  echo "Missing configure script: ${CONFIGURE_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
  IFS=',' read -r -a _cuda_device_list <<< "${CUDA_DEVICES}"
  if [[ "${#_cuda_device_list[@]}" -ne 4 ]]; then
    echo "Expected exactly 4 CUDA devices for MMSeqs GPU server, got: ${CUDA_DEVICES}" >&2
    echo "Pass --cuda-devices explicitly if this host should use a different set." >&2
    exit 1
  fi
fi

cfg_cmd=(bash "${CONFIGURE_SCRIPT}" --msa-root "${MSA_ROOT}" --uniref30-root "${UNIREF30_ROOT}")
if [[ -n "${CONFIG_PATH}" ]]; then
  cfg_cmd+=(--config "${CONFIG_PATH}")
fi
if [[ "${CLEANUP_UNIREF100}" -eq 1 ]]; then
  cfg_cmd+=(--cleanup-uniref100)
fi
if [[ -n "${HOST}" ]]; then
  cfg_cmd+=(--host "${HOST}")
fi
if [[ -n "${PORT}" ]]; then
  cfg_cmd+=(--port "${PORT}")
fi
if [[ -n "${LOCAL_WORKERS}" ]]; then
  cfg_cmd+=(--local-workers "${LOCAL_WORKERS}")
fi
if [[ -n "${PARALLEL_DATABASES}" ]]; then
  cfg_cmd+=(--parallel-databases "${PARALLEL_DATABASES}")
fi
if [[ "${PARALLEL_STAGES}" == "true" ]]; then
  cfg_cmd+=(--parallel-stages)
elif [[ "${PARALLEL_STAGES}" == "false" ]]; then
  cfg_cmd+=(--no-parallel-stages)
fi
if [[ "${DISABLE_MMSEQS_TUNE}" -eq 1 ]]; then
  cfg_cmd+=(--disable-mmseqs-tune)
else
  cfg_cmd+=(--mmseqs-max-seqs "${MMSEQS_MAX_SEQS}" --mmseqs-num-iterations "${MMSEQS_NUM_ITERATIONS}")
fi

"${cfg_cmd[@]}"

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${MSA_ROOT}/config.uniref30.json"
fi
MMSEQS_SERVER_BIN="${MSA_ROOT}/ColabFold/MsaServer/bin/mmseqs-server"
MMSEQS_BIN="${MSA_ROOT}/ColabFold/MsaServer/bin/mmseqs"
LOG_FILE_DEFAULT="${MSA_ROOT}/logs/mmseqs-server.log"
PID_FILE_DEFAULT="${MSA_ROOT}/run/mmseqs-server.pid"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing MMSeqs server config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -x "${MMSEQS_SERVER_BIN}" ]]; then
  echo "Missing mmseqs-server binary: ${MMSEQS_SERVER_BIN}" >&2
  exit 1
fi
if [[ ! -x "${MMSEQS_BIN}" ]]; then
  echo "Missing mmseqs binary: ${MMSEQS_BIN}" >&2
  exit 1
fi

server_cmd=("${MMSEQS_SERVER_BIN}" -local -config "${CONFIG_PATH}" -paths.colabfold.gpu.gpu 1 -paths.colabfold.gpu.server "${GPU_SERVER}")
if [[ "${FOREGROUND}" -eq 1 ]]; then
  if [[ -n "${CUDA_DEVICES}" ]]; then
    echo "[msa-uniref30] CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
    exec env LOCAL=true CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${server_cmd[@]}"
  fi
  exec env LOCAL=true "${server_cmd[@]}"
fi

if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${LOG_FILE_DEFAULT}"
fi
if [[ -z "${PID_FILE}" ]]; then
  PID_FILE="${PID_FILE_DEFAULT}"
fi
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Error: mmseqs-server appears to already be running (pid=${old_pid})." >&2
    echo "Stop it first: kill ${old_pid}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

foreground_detached_cmd=("${server_cmd[@]}")
echo "[msa-uniref30] detached launch via setsid"
echo "[msa-uniref30] log=${LOG_FILE}"
echo "[msa-uniref30] pid_file=${PID_FILE}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "[msa-uniref30] CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  setsid env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${foreground_detached_cmd[@]}" </dev/null >"${LOG_FILE}" 2>&1 &
else
  setsid "${foreground_detached_cmd[@]}" </dev/null >"${LOG_FILE}" 2>&1 &
fi
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "[msa-uniref30] pid=${pid}"
echo "[msa-uniref30] waiting for localhost:${PORT:-8080}..."

check_port="${PORT:-8080}"
ok=0
for _ in $(seq 1 30); do
  if curl -sS --max-time 2 "http://127.0.0.1:${check_port}/api/" >/dev/null 2>&1 \
     || curl -sS --max-time 2 "http://127.0.0.1:${check_port}/" >/dev/null 2>&1; then
    if kill -0 "${pid}" 2>/dev/null; then
      ok=1
      break
    fi
  fi
  sleep 1
done
if [[ "${ok}" == "1" ]]; then
  echo "[msa-uniref30] server is reachable on http://127.0.0.1:${check_port}"
  exit 0
fi

echo "[msa-uniref30] error: detached server did not stay reachable; recent log:" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
rm -f "${PID_FILE}"
exit 1
