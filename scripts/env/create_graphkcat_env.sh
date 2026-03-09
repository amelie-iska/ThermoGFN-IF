#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="apodock"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
CPU_ONLY=0
SOLVER="libmamba"
ATTEMPTS=4
RETRY_SLEEP_SEC=15
TORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCHAUDIO_VERSION="2.4.0"
PYG_VERSION="2.6.1"
PYG_SCATTER_VERSION="2.1.2"
PYG_CLUSTER_VERSION="1.6.3"

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/create_graphkcat_env.sh [options]

Options:
  --env-name NAME     Conda env name override (default: apodock)
  --python VERSION    Python version (default: 3.10)
  --cuda VERSION      pytorch-cuda version (default: 12.1)
  --cpu-only          Install CPU-only torch stack
  --solver NAME       Conda solver: libmamba|classic (default: libmamba)
  --attempts N        Retry attempts per solver for download/extract failures (default: 4)
  --retry-sleep N     Sleep seconds between retries (default: 15)
  -h, --help          Show help

Notes:
  This script intentionally does not use models/GraphKcat/apodock.yaml.
  That file is an exported workstation lockfile pinned to Python 3.8 and
  includes broken pip requirements for current package indexes. Instead, this
  script builds a curated GraphKcat inference/runtime environment.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"; shift 2 ;;
    --python)
      PYTHON_VERSION="$2"; shift 2 ;;
    --cuda)
      CUDA_VERSION="$2"; shift 2 ;;
    --cpu-only)
      CPU_ONLY=1; shift ;;
    --solver)
      SOLVER="$2"; shift 2 ;;
    --attempts)
      ATTEMPTS="$2"; shift 2 ;;
    --retry-sleep)
      RETRY_SLEEP_SEC="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

PYTHON_TAG="${PYTHON_VERSION/./}"
CUDA_TAG="${CUDA_VERSION/./}"

export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-120}"
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"
export CONDA_REMOTE_MAX_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-10}"
export CONDA_REMOTE_BACKOFF_FACTOR="${CONDA_REMOTE_BACKOFF_FACTOR:-2}"
export CONDA_FETCH_THREADS="${CONDA_FETCH_THREADS:-1}"
export CONDA_EXTRACT_THREADS="${CONDA_EXTRACT_THREADS:-1}"

echo "[graphkcat-env] env=${ENV_NAME} python=${PYTHON_VERSION} solver=${SOLVER} cpu_only=${CPU_ONLY}"
echo "[graphkcat-env] CONDA_REMOTE_READ_TIMEOUT_SECS=${CONDA_REMOTE_READ_TIMEOUT_SECS}"
echo "[graphkcat-env] CONDA_REMOTE_CONNECT_TIMEOUT_SECS=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS}"
echo "[graphkcat-env] CONDA_REMOTE_MAX_RETRIES=${CONDA_REMOTE_MAX_RETRIES}"
echo "[graphkcat-env] CONDA_FETCH_THREADS=${CONDA_FETCH_THREADS}"
echo "[graphkcat-env] CONDA_EXTRACT_THREADS=${CONDA_EXTRACT_THREADS}"
echo "[graphkcat-env] attempts=${ATTEMPTS} retry_sleep_sec=${RETRY_SLEEP_SEC}"

TMP_CONDARC="$(mktemp)"
cat >"$TMP_CONDARC" <<'YAML'
channel_priority: flexible
channels:
  - pytorch
  - nvidia
  - pyg
  - conda-forge
  - defaults
YAML
trap 'rm -f "$TMP_CONDARC"' EXIT
echo "[graphkcat-env] using temporary CONDARC with channel_priority=flexible"

run_conda_with_solver() {
  local label="$1"
  local solver_name="$2"
  shift 2
  local attempt
  local cmd=("$@" "--solver" "$solver_name")
  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    echo "[graphkcat-env] ${label} solver=${solver_name} attempt=${attempt}/${ATTEMPTS}"
    if CONDARC="$TMP_CONDARC" CONDA_CHANNEL_PRIORITY=flexible "${cmd[@]}"; then
      return 0
    fi
    if [[ "$attempt" -lt "$ATTEMPTS" ]]; then
      echo "[graphkcat-env] ${label} failed; cleaning tarballs and retrying after ${RETRY_SLEEP_SEC}s"
      conda clean -y --tarballs >/dev/null 2>&1 || true
      sleep "$RETRY_SLEEP_SEC"
    fi
  done
  return 1
}

run_conda_step() {
  local label="$1"
  shift
  if run_conda_with_solver "$label" "$SOLVER" "$@"; then
    return 0
  fi
  if [[ "$SOLVER" != "classic" ]]; then
    echo "[graphkcat-env] ${label} failed with ${SOLVER}; retrying via classic solver"
    run_conda_with_solver "$label" "classic" "$@"
    return $?
  fi
  return 1
}

run_pip_step() {
  local label="$1"
  shift
  local attempt
  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    echo "[graphkcat-env] ${label} attempt=${attempt}/${ATTEMPTS}"
    if conda run -n "$ENV_NAME" python -m pip install "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$ATTEMPTS" ]]; then
      sleep "$RETRY_SLEEP_SEC"
    fi
  done
  return 1
}

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  run_conda_step "create-base" conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
else
  echo "[graphkcat-env] reusing existing env ${ENV_NAME}"
fi

run_conda_step "ensure-python" conda install -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip

if [[ "$CPU_ONLY" -eq 1 ]]; then
  run_conda_step \
    "install-torch" \
    conda install -y -n "$ENV_NAME" -c pytorch \
    "pytorch::pytorch=${TORCH_VERSION}=py${PYTHON_VERSION}_cpu_*" \
    "pytorch::torchvision=${TORCHVISION_VERSION}=py${PYTHON_TAG}_cpu" \
    "pytorch::torchaudio=${TORCHAUDIO_VERSION}=py${PYTHON_TAG}_cpu" \
    cpuonly
else
  run_conda_step \
    "install-torch" \
    conda install -y -n "$ENV_NAME" -c pytorch -c nvidia \
    "pytorch::pytorch=${TORCH_VERSION}=py${PYTHON_VERSION}_cuda${CUDA_VERSION}*" \
    "pytorch::torchvision=${TORCHVISION_VERSION}=py${PYTHON_TAG}_cu${CUDA_TAG}" \
    "pytorch::torchaudio=${TORCHAUDIO_VERSION}=py${PYTHON_TAG}_cu${CUDA_TAG}" \
    "pytorch::pytorch-cuda=${CUDA_VERSION}" \
    "pytorch::pytorch-mutex=*=cuda"
fi

run_conda_step \
  "install-science" \
  conda install -y -n "$ENV_NAME" --freeze-installed -c conda-forge \
  numpy pandas scipy networkx biopython rdkit tqdm pyyaml click einops joblib pymol-open-source

run_conda_step \
  "install-pyg" \
  conda install -y -n "$ENV_NAME" -c pyg \
  "pyg::pyg=${PYG_VERSION}" \
  "pyg::pytorch-scatter=${PYG_SCATTER_VERSION}" \
  "pyg::pytorch-cluster=${PYG_CLUSTER_VERSION}"

echo "[graphkcat-env] upgrading pip"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

run_pip_step "install-pip-runtime" "fair-esm==2.0.0" "unimol-tools==0.1.4.post1"

echo "[graphkcat-env] validating imports"
conda run -n "$ENV_NAME" python - <<'PY'
import torch
import pandas
import esm
import pymol
from rdkit import Chem
from torch_geometric.data import HeteroData
import torch_cluster
import torch_scatter
from unimol_tools import UniMolRepr

print("cuda_available", torch.cuda.is_available())
print("ok")
PY

echo "[graphkcat-env] done"
