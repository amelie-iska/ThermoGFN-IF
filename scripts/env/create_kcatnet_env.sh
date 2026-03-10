#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="KcatNet"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
CPU_ONLY=0
SOLVER="classic"
ATTEMPTS=4
RETRY_SLEEP_SEC=15
TORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCHAUDIO_VERSION="2.4.0"
PYG_VERSION="2.6.1"
PYG_SCATTER_VERSION="2.1.2"
TORCH_RUNTIME_CANDIDATES=("2021.4.0:2021.4.0" "2023.1.0:2023.1.0" ":")

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/create_kcatnet_env.sh [options]

Options:
  --env-name NAME       Conda env name (default: KcatNet)
  --python VERSION      Python version (default: 3.10)
  --cuda VERSION        pytorch-cuda version (default: 12.1)
  --cpu-only            Install CPU-only torch stack
  --solver NAME         Conda solver: libmamba|classic (default: classic)
  --attempts N          Retry attempts per solver for download/extract failures (default: 4)
  --retry-sleep N       Sleep seconds between retries (default: 15)
  -h, --help            Show help

Notes:
  - Uses a temporary CONDARC with channel_priority=flexible to avoid strict-priority conflicts.
  - Re-running this script repairs an existing env in place.
  - Installs dependencies required by scripts/prep/oracles/kcatnet_score.py,
    including torch_scatter and a conda-forge Pillow/libtiff pair.
  - Repairs the GPU torch runtime by trying older MKL/OpenMP combinations from
    the checked-in env specs when newer solver choices trigger the libtorch
    `iJIT_NotifyEvent` import error.
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

echo "[kcatnet-env] env=${ENV_NAME} python=${PYTHON_VERSION} solver=${SOLVER} cpu_only=${CPU_ONLY}"
echo "[kcatnet-env] CONDA_REMOTE_READ_TIMEOUT_SECS=${CONDA_REMOTE_READ_TIMEOUT_SECS}"
echo "[kcatnet-env] CONDA_REMOTE_CONNECT_TIMEOUT_SECS=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS}"
echo "[kcatnet-env] CONDA_REMOTE_MAX_RETRIES=${CONDA_REMOTE_MAX_RETRIES}"
echo "[kcatnet-env] CONDA_FETCH_THREADS=${CONDA_FETCH_THREADS}"
echo "[kcatnet-env] CONDA_EXTRACT_THREADS=${CONDA_EXTRACT_THREADS}"
echo "[kcatnet-env] attempts=${ATTEMPTS} retry_sleep_sec=${RETRY_SLEEP_SEC}"

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
echo "[kcatnet-env] using temporary CONDARC with channel_priority=flexible"

run_conda_with_solver() {
  local label="$1"
  local solver_name="$2"
  shift 2
  local attempt
  local cmd=("$@" "--solver" "$solver_name")
  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    echo "[kcatnet-env] ${label} solver=${solver_name} attempt=${attempt}/${ATTEMPTS}"
    if CONDARC="$TMP_CONDARC" CONDA_CHANNEL_PRIORITY=flexible "${cmd[@]}"; then
      return 0
    fi
    if [[ "$attempt" -lt "$ATTEMPTS" ]]; then
      echo "[kcatnet-env] ${label} failed; cleaning tarballs and retrying after ${RETRY_SLEEP_SEC}s"
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
    echo "[kcatnet-env] ${label} failed with ${SOLVER}; retrying via classic solver"
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
    echo "[kcatnet-env] ${label} attempt=${attempt}/${ATTEMPTS}"
    if conda run -n "$ENV_NAME" python -m pip install "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$ATTEMPTS" ]]; then
      sleep "$RETRY_SLEEP_SEC"
    fi
  done
  return 1
}

repair_torch_runtime_with_versions() {
  local mkl_version="$1"
  local intel_openmp_version="$2"
  local runtime_label="${mkl_version:-solver-defaults}"
  local -a cmd=(
    conda install -y -n "$ENV_NAME"
    --override-channels
    -c pytorch
    -c nvidia
    -c defaults
    --force-reinstall
    "defaults::blas=*=mkl"
  )
  if [[ -n "$mkl_version" && -n "$intel_openmp_version" ]]; then
    cmd+=(
      "defaults::mkl=${mkl_version}"
      "defaults::intel-openmp=${intel_openmp_version}"
    )
  else
    cmd+=(
      defaults::mkl
      defaults::intel-openmp
    )
  fi
  if [[ "$CPU_ONLY" -eq 1 ]]; then
    cmd+=(
      "pytorch::pytorch=${TORCH_VERSION}=py${PYTHON_VERSION}_cpu_*"
      "pytorch::torchvision=${TORCHVISION_VERSION}=py${PYTHON_TAG}_cpu"
      "pytorch::torchaudio=${TORCHAUDIO_VERSION}=py${PYTHON_TAG}_cpu"
      cpuonly
    )
  else
    cmd+=(
      "pytorch::pytorch=${TORCH_VERSION}=py${PYTHON_VERSION}_cuda${CUDA_VERSION}*"
      "pytorch::torchvision=${TORCHVISION_VERSION}=py${PYTHON_TAG}_cu${CUDA_TAG}"
      "pytorch::torchaudio=${TORCHAUDIO_VERSION}=py${PYTHON_TAG}_cu${CUDA_TAG}"
      "pytorch::pytorch-cuda=${CUDA_VERSION}"
      "pytorch::pytorch-mutex=*=cuda"
    )
  fi

  echo "[kcatnet-env] repairing torch runtime (mkl=${runtime_label})"
  if ! run_conda_with_solver "repair-torch-runtime" "$SOLVER" "${cmd[@]}"; then
    if [[ "$SOLVER" != "classic" ]]; then
      echo "[kcatnet-env] repair-torch-runtime failed with ${SOLVER}; retrying via classic solver"
      run_conda_with_solver "repair-torch-runtime" "classic" "${cmd[@]}"
    else
      return 1
    fi
  fi
}

repair_torch_runtime() {
  local candidate
  local mkl_version
  local intel_openmp_version
  for candidate in "${TORCH_RUNTIME_CANDIDATES[@]}"; do
    IFS=":" read -r mkl_version intel_openmp_version <<<"$candidate"
    if repair_torch_runtime_with_versions "$mkl_version" "$intel_openmp_version"; then
      return 0
    fi
  done
  return 1
}

validate_imports() {
  conda run -n "$ENV_NAME" bash -lc "cd '$REPO_ROOT/models/KcatNet' && python -c \"import torch, esm, transformers, torch_scatter; from rdkit import Chem; from torch_geometric.data import Data; from models.model_kcat import KcatNet; from utils.protein_init import T5Tokenizer; print('cuda_available', torch.cuda.is_available()); print('ok')\""
}

validate_imports_with_runtime_repair() {
  local log_file
  log_file="$(mktemp)"
  if validate_imports >"$log_file" 2>&1; then
    cat "$log_file"
    rm -f "$log_file"
    return 0
  fi

  cat "$log_file" >&2
  if grep -Fq "iJIT_NotifyEvent" "$log_file"; then
    echo "[kcatnet-env] detected libtorch iJIT runtime failure; normalizing the torch + CUDA runtime"
    rm -f "$log_file"
    repair_torch_runtime
    validate_imports
    return $?
  fi

  rm -f "$log_file"
  return 1
}

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  run_conda_step "create-base" conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
else
  echo "[kcatnet-env] reusing existing env ${ENV_NAME}"
fi

run_conda_step "ensure-python" conda install -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip

echo "[kcatnet-env] installing torch stack"
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

echo "[kcatnet-env] installing scientific dependencies"
run_conda_step \
  "install-science" \
  conda install -y -n "$ENV_NAME" --freeze-installed -c conda-forge \
  numpy scipy pandas scikit-learn biopython rdkit tqdm h5py sentencepiece \
  conda-forge::pillow conda-forge::libtiff

echo "[kcatnet-env] installing torch-geometric stack and extensions"
run_conda_step \
  "install-pyg" \
  conda install -y -n "$ENV_NAME" -c pyg \
  "pyg::pyg=${PYG_VERSION}" "pyg::pytorch-scatter=${PYG_SCATTER_VERSION}"

echo "[kcatnet-env] normalizing Pillow/libtiff linkage"
if ! run_conda_with_solver \
  "repair-pillow" \
  "$SOLVER" \
  conda install -y -n "$ENV_NAME" --override-channels -c conda-forge --force-reinstall pillow libtiff; then
  if [[ "$SOLVER" != "classic" ]]; then
    echo "[kcatnet-env] repair-pillow failed with ${SOLVER}; retrying via classic solver"
    run_conda_with_solver \
      "repair-pillow" \
      "classic" \
      conda install -y -n "$ENV_NAME" --override-channels -c conda-forge --force-reinstall pillow libtiff
  else
    exit 1
  fi
fi

echo "[kcatnet-env] installing protein language model dependencies"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
run_pip_step "install-pip-runtime" "fair-esm==2.0.0" "transformers>=4.35,<4.46"

echo "[kcatnet-env] validating imports"
validate_imports_with_runtime_repair

echo "[kcatnet-env] done"
