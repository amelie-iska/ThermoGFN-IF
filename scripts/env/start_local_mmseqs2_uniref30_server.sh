#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIBLING_START="${REPO_ROOT}/../enzyme-quiver/scripts/start_local_mmseqs2_server.sh"
CONFIGURE_SCRIPT="${SCRIPT_DIR}/configure_local_mmseqs2_uniref30.sh"

MSA_ROOT="${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa"
UNIREF30_ROOT="/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302"
CONFIG_PATH=""
CLEANUP_UNIREF100=0
GPU_SERVER=""
FOREGROUND=0
HOST=""
PORT=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/env/start_local_mmseqs2_uniref30_server.sh [options]

Generate a local MMSeqs2 server config that points at the existing UniRef30 DB,
then start the sibling MMseqs2 server wrapper with GPU backend enabled.

Options:
  --msa-root PATH           Local MSA workspace root
  --uniref30-root PATH      Directory containing uniref30_2302_db*
  --config PATH             Output config path (default: <msa-root>/config.uniref30.json)
  --cleanup-uniref100       Remove accidental local uniref100_db* artifacts first
  --gpu-server N            Forward --gpu-server to sibling start script
  --foreground              Run attached instead of background mode
  --host HOST               Override host in generated config
  --port PORT               Override port in generated config and start command
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
    --foreground)
      FOREGROUND=1; shift ;;
    --host)
      HOST="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
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

if [[ ! -f "${SIBLING_START}" ]]; then
  echo "Missing sibling start script: ${SIBLING_START}" >&2
  exit 1
fi
if [[ ! -f "${CONFIGURE_SCRIPT}" ]]; then
  echo "Missing configure script: ${CONFIGURE_SCRIPT}" >&2
  exit 1
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

"${cfg_cmd[@]}"

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${MSA_ROOT}/config.uniref30.json"
fi

start_cmd=(bash "${SIBLING_START}" --msa-root "${MSA_ROOT}" --config "${CONFIG_PATH}" --gpu-backend)
if [[ -n "${GPU_SERVER}" ]]; then
  start_cmd+=(--gpu-server "${GPU_SERVER}")
fi
if [[ -n "${PORT}" ]]; then
  start_cmd+=(--port "${PORT}")
fi
if [[ "${FOREGROUND}" -eq 1 ]]; then
  start_cmd+=(--foreground)
fi

exec "${start_cmd[@]}"
