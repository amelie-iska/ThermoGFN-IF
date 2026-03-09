#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="MMKcat"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
CPU_ONLY=0
INSTALL_DSSP=1
SOLVER="libmamba"

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/create_mmkcat_env.sh [options]

Options:
  --env-name NAME       Conda env name (default: MMKcat)
  --python VERSION      Python version (default: 3.10)
  --cuda VERSION        pytorch-cuda version (default: 12.1)
  --cpu-only            Install CPU-only torch stack
  --skip-dssp           Do not install mkdssp
  --solver NAME         Conda solver: libmamba|classic (default: libmamba)
  -h, --help            Show help

Notes:
  - Forces CONDA_CHANNEL_PRIORITY=flexible for compatibility with strict setups.
  - Installs MMKcat inference dependencies used by scripts/prep/oracles/mmkcat_score.py.
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
    --skip-dssp)
      INSTALL_DSSP=0; shift ;;
    --solver)
      SOLVER="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

echo "[mmkcat-env] env=${ENV_NAME} python=${PYTHON_VERSION} solver=${SOLVER} cpu_only=${CPU_ONLY}"
echo "[mmkcat-env] forcing CONDA_CHANNEL_PRIORITY=flexible"
export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-60}"
echo "[mmkcat-env] CONDA_REMOTE_READ_TIMEOUT_SECS=${CONDA_REMOTE_READ_TIMEOUT_SECS}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[mmkcat-env] creating base env"
  CONDA_CHANNEL_PRIORITY=flexible conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip --solver "$SOLVER"
else
  echo "[mmkcat-env] reusing existing env ${ENV_NAME}"
fi

echo "[mmkcat-env] installing torch stack"
if [[ "$CPU_ONLY" -eq 1 ]]; then
  CONDA_CHANNEL_PRIORITY=flexible conda install -y -n "$ENV_NAME" -c pytorch pytorch torchvision torchaudio cpuonly --solver "$SOLVER"
else
  CONDA_CHANNEL_PRIORITY=flexible conda install -y -n "$ENV_NAME" -c pytorch -c nvidia "pytorch-cuda=${CUDA_VERSION}" pytorch torchvision torchaudio --solver "$SOLVER"
fi

echo "[mmkcat-env] installing scientific deps"
CONDA_CHANNEL_PRIORITY=flexible conda install -y -n "$ENV_NAME" -c conda-forge \
  numpy scipy pandas scikit-learn biopython rdkit tqdm matplotlib seaborn networkx colorama \
  --solver "$SOLVER"

echo "[mmkcat-env] installing torch-geometric stack"
CONDA_CHANNEL_PRIORITY=flexible conda install -y -n "$ENV_NAME" -c pyg pyg --solver "$SOLVER"

echo "[mmkcat-env] installing ESM/ESMFold"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install "fair-esm[esmfold]==2.0.0"

if [[ "$INSTALL_DSSP" -eq 1 ]]; then
  echo "[mmkcat-env] installing mkdssp"
  CONDA_CHANNEL_PRIORITY=flexible conda install -y -n "$ENV_NAME" -c ostrokach dssp --solver "$SOLVER"
fi

echo "[mmkcat-env] validating imports"
if [[ "$INSTALL_DSSP" -eq 1 ]]; then
  conda run -n "$ENV_NAME" python - <<'PY'
import shutil
import torch
import esm
from torch_geometric.data import Data
from Bio.PDB import PDBParser

assert shutil.which("mkdssp"), "mkdssp not found in PATH"
print("ok")
PY
else
  conda run -n "$ENV_NAME" python - <<'PY'
import torch
import esm
from torch_geometric.data import Data
from Bio.PDB import PDBParser
print("ok")
PY
fi

echo "[mmkcat-env] done"
