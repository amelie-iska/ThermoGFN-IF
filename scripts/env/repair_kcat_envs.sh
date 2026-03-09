#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUT_JSON="${REPO_ROOT}/runs/env_status_kcat_health.json"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
KCATNET_ENV="KcatNet"
GRAPHKCAT_ENV="apodock"
ATTEMPTS=6
RETRY_SLEEP_SEC=20
KCATNET_SOLVER="libmamba"
GRAPHKCAT_SOLVER="libmamba"
CPU_ONLY=0
INCLUDE_LIGANDMPNN=1

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/repair_kcat_envs.sh [options]

Options:
  --out-json PATH              Health report path
                               (default: runs/env_status_kcat_health.json)
  --python VERSION             Python version for both envs (default: 3.10)
  --cuda VERSION               pytorch-cuda version for both envs (default: 12.1)
  --kcatnet-env NAME           KcatNet env name (default: KcatNet)
  --graphkcat-env NAME         GraphKcat env name (default: apodock)
  --kcatnet-solver NAME        Solver for KcatNet env (default: libmamba)
  --graphkcat-solver NAME      Solver for GraphKcat env (default: libmamba)
  --attempts N                 Retry attempts per solver (default: 6)
  --retry-sleep N              Sleep seconds between retries (default: 20)
  --cpu-only                   Build CPU-only torch stacks
  --no-ligandmpnn-check        Exclude ligandmpnn_env from final health check
  -h, --help                   Show help

Notes:
  - Repairs both Kcat oracle envs in place by re-running the curated env
    installers.
  - Runs strict health checks at the end.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-json)
      OUT_JSON="$2"; shift 2 ;;
    --python)
      PYTHON_VERSION="$2"; shift 2 ;;
    --cuda)
      CUDA_VERSION="$2"; shift 2 ;;
    --kcatnet-env)
      KCATNET_ENV="$2"; shift 2 ;;
    --graphkcat-env)
      GRAPHKCAT_ENV="$2"; shift 2 ;;
    --kcatnet-solver)
      KCATNET_SOLVER="$2"; shift 2 ;;
    --graphkcat-solver)
      GRAPHKCAT_SOLVER="$2"; shift 2 ;;
    --attempts)
      ATTEMPTS="$2"; shift 2 ;;
    --retry-sleep)
      RETRY_SLEEP_SEC="$2"; shift 2 ;;
    --cpu-only)
      CPU_ONLY=1; shift ;;
    --no-ligandmpnn-check)
      INCLUDE_LIGANDMPNN=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

mkdir -p "$(dirname "$OUT_JSON")"

cpu_flag=()
if [[ "$CPU_ONLY" -eq 1 ]]; then
  cpu_flag=(--cpu-only)
fi

echo "[repair-kcat-envs] repo_root=${REPO_ROOT}"
echo "[repair-kcat-envs] out_json=${OUT_JSON}"
echo "[repair-kcat-envs] python=${PYTHON_VERSION} cuda=${CUDA_VERSION} cpu_only=${CPU_ONLY}"
echo "[repair-kcat-envs] kcatnet_env=${KCATNET_ENV} graphkcat_env=${GRAPHKCAT_ENV}"
echo "[repair-kcat-envs] attempts=${ATTEMPTS} retry_sleep_sec=${RETRY_SLEEP_SEC}"

echo "[repair-kcat-envs] repairing ${KCATNET_ENV}"
"${REPO_ROOT}/scripts/env/create_kcatnet_env.sh" \
  --env-name "${KCATNET_ENV}" \
  --python "${PYTHON_VERSION}" \
  --cuda "${CUDA_VERSION}" \
  --solver "${KCATNET_SOLVER}" \
  --attempts "${ATTEMPTS}" \
  --retry-sleep "${RETRY_SLEEP_SEC}" \
  "${cpu_flag[@]}"

echo "[repair-kcat-envs] repairing ${GRAPHKCAT_ENV}"
"${REPO_ROOT}/scripts/env/create_graphkcat_env.sh" \
  --env-name "${GRAPHKCAT_ENV}" \
  --python "${PYTHON_VERSION}" \
  --cuda "${CUDA_VERSION}" \
  --solver "${GRAPHKCAT_SOLVER}" \
  --attempts "${ATTEMPTS}" \
  --retry-sleep "${RETRY_SLEEP_SEC}" \
  "${cpu_flag[@]}"

check_envs=("${KCATNET_ENV}" "${GRAPHKCAT_ENV}")
if [[ "$INCLUDE_LIGANDMPNN" -eq 1 ]]; then
  check_envs=("ligandmpnn_env" "${check_envs[@]}")
fi

echo "[repair-kcat-envs] running strict health checks: ${check_envs[*]}"
RUN_HEALTH_CHECKS=1 "${REPO_ROOT}/scripts/env/check_kcat_envs.sh" "${OUT_JSON}" "${check_envs[@]}"

echo "[repair-kcat-envs] done"
