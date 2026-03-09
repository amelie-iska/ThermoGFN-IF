#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="bioemu"
SKIP_REMOVE=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/rebuild_bioemu_env_py311.sh [options]

Options:
  --env-name NAME   Conda environment name (default: bioemu)
  --skip-remove     Do not remove existing environment before create
  -h, --help        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --skip-remove)
      SKIP_REMOVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[rebuild-bioemu] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[rebuild-bioemu] target_env=${ENV_NAME}"
if [[ "$SKIP_REMOVE" -ne 1 ]]; then
  echo "[rebuild-bioemu] removing existing env if present"
  conda env remove -n "${ENV_NAME}" -y || true
fi

echo "[rebuild-bioemu] creating python=3.11 env"
conda create -n "${ENV_NAME}" -y python=3.11 pip

echo "[rebuild-bioemu] installing bioemu"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install bioemu

echo "[rebuild-bioemu] health check"
conda run -n "${ENV_NAME}" python -V
conda run -n "${ENV_NAME}" python -m bioemu.sample --help >/dev/null
echo "[rebuild-bioemu] ready"
