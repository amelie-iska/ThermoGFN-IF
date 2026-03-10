#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/.venvs/foundry-rf3"
CHECKPOINT_PATH=""
LOCAL_MSA_ROOT="${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa"

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/check_foundry_rf3_env.sh [options]

Options:
  --env-dir PATH         Repo-local virtualenv directory (default: .venvs/foundry-rf3)
  --checkpoint PATH      Optional checkpoint file or registered checkpoint name
  --local-msa-root PATH  Optional MMSeqs2 local workspace path to check
  -h, --help             Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"; shift 2 ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"; shift 2 ;;
    --local-msa-root)
      LOCAL_MSA_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing python executable: ${ENV_DIR}/bin/python" >&2
  exit 1
fi

echo "[foundry-rf3-check] repo_root=${REPO_ROOT}"
echo "[foundry-rf3-check] env_dir=${ENV_DIR}"

"${ENV_DIR}/bin/python" - <<'PY'
import importlib
mods = ["foundry", "rf3"]
for mod in mods:
    importlib.import_module(mod)
print("imports_ok")
PY

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ "${CHECKPOINT_PATH}" == *.* || "${CHECKPOINT_PATH}" == */* ]]; then
    if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
      echo "Checkpoint path does not exist: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi
    echo "[foundry-rf3-check] checkpoint_exists=${CHECKPOINT_PATH}"
  else
    echo "[foundry-rf3-check] checkpoint_name=${CHECKPOINT_PATH}"
  fi
fi

if [[ -n "${LOCAL_MSA_ROOT}" ]]; then
  if [[ -L "${LOCAL_MSA_ROOT}" || -d "${LOCAL_MSA_ROOT}" ]]; then
    echo "[foundry-rf3-check] local_msa_root_present=${LOCAL_MSA_ROOT}"
  else
    echo "Local MSA root is missing: ${LOCAL_MSA_ROOT}" >&2
    exit 1
  fi
fi
