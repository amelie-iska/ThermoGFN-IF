#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/.venvs/foundry-rf3"
PYTHON_VERSION="3.12"
INSTALL_CHECKPOINTS=0
CHECKPOINT_DIR="${REPO_ROOT}/weights"
ENV_TOOL=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/create_foundry_rf3_env.sh [options]

Options:
  --env-dir PATH         Repo-local virtualenv directory (default: .venvs/foundry-rf3)
  --python VALUE         Python version, executable, or absolute path (default: 3.12)
  --env-tool TOOL        Force environment tool: auto|uv|venv (default: auto)
  --install-checkpoints  Also run `foundry install rf3`
  --checkpoint-dir PATH  Checkpoint directory for `foundry install` (default: ./weights)
  -h, --help             Show help

Notes:
  This creates a repo-local virtualenv and installs the local Foundry tree
  in editable mode with the `rf3` extra. It does not require conda.
  If `uv` is available it is used to create the environment; otherwise the
  script falls back to `python -m venv`.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"; shift 2 ;;
    --python)
      PYTHON_VERSION="$2"; shift 2 ;;
    --env-tool)
      ENV_TOOL="$2"; shift 2 ;;
    --install-checkpoints)
      INSTALL_CHECKPOINTS=1; shift ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

resolve_python_bin() {
  local requested="$1"
  if [[ "${requested}" == */* ]]; then
    if [[ -x "${requested}" ]]; then
      printf '%s\n' "${requested}"
      return 0
    fi
    echo "Requested python is not executable: ${requested}" >&2
    return 1
  fi
  if command -v "${requested}" >/dev/null 2>&1; then
    command -v "${requested}"
    return 0
  fi
  if [[ "${requested}" =~ ^[0-9]+(\.[0-9]+)?$ ]] && command -v "python${requested}" >/dev/null 2>&1; then
    command -v "python${requested}"
    return 0
  fi
  echo "Could not resolve requested python '${requested}'. Install it or pass --python /abs/path/to/python." >&2
  return 1
}

pick_env_tool() {
  local requested="$1"
  case "${requested}" in
    ""|auto)
      if command -v uv >/dev/null 2>&1; then
        printf '%s\n' "uv"
      else
        printf '%s\n' "venv"
      fi
      ;;
    uv)
      if ! command -v uv >/dev/null 2>&1; then
        echo "Requested --env-tool uv but uv was not found on PATH" >&2
        return 1
      fi
      printf '%s\n' "uv"
      ;;
    venv)
      printf '%s\n' "venv"
      ;;
    *)
      echo "Unknown --env-tool value: ${requested}" >&2
      return 1
      ;;
  esac
}

PKG_ROOT="${REPO_ROOT}/models/foundry"
mkdir -p "$(dirname "${ENV_DIR}")"

ENV_TOOL="$(pick_env_tool "${ENV_TOOL}")"
PYTHON_BIN="$(resolve_python_bin "${PYTHON_VERSION}")"

echo "[foundry-rf3-env] repo_root=${REPO_ROOT}"
echo "[foundry-rf3-env] pkg_root=${PKG_ROOT}"
echo "[foundry-rf3-env] env_dir=${ENV_DIR}"
echo "[foundry-rf3-env] python_request=${PYTHON_VERSION}"
echo "[foundry-rf3-env] python_bin=${PYTHON_BIN}"
echo "[foundry-rf3-env] env_tool=${ENV_TOOL}"

if [[ "${ENV_TOOL}" == "uv" ]]; then
  uv venv --python "${PYTHON_BIN}" "${ENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

(
  cd "${PKG_ROOT}"
  "${ENV_DIR}/bin/python" -m pip install -e ".[rf3]"
)

echo "[foundry-rf3-env] installed local Foundry RF3 package"
echo "[foundry-rf3-env] activate with: source ${ENV_DIR}/bin/activate"

if [[ "${INSTALL_CHECKPOINTS}" -eq 1 ]]; then
  mkdir -p "${CHECKPOINT_DIR}"
  export FOUNDRY_CHECKPOINT_DIRS="${CHECKPOINT_DIR}${FOUNDRY_CHECKPOINT_DIRS:+:${FOUNDRY_CHECKPOINT_DIRS}}"
  echo "[foundry-rf3-env] installing RF3 checkpoints into ${CHECKPOINT_DIR}"
  "${ENV_DIR}/bin/foundry" install rf3 --checkpoint-dir "${CHECKPOINT_DIR}"
fi
