#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BIOEMU_ENV="${BIOEMU_ENV_NAME:-bioemu}"
TARGET_DIR="${BIOEMU_COLABFOLD_DIR:-${REPO_ROOT}/.cache/bioemu/colabfold}"
FORCE=0
CHECK_ONLY=0
SKIP_NET_CHECK=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/setup_bioemu_colabfold_runtime.sh [options]

Options:
  --bioemu-env NAME     Conda env containing bioemu package and uv (default: bioemu)
  --target-dir PATH     ColabFold runtime directory (default: ./.cache/bioemu/colabfold)
  --force               Recreate target runtime even if present
  --check-only          Validate runtime only; do not install
  --skip-net-check      Skip DNS preflight for pypi hosts
  -h, --help            Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bioemu-env)
      BIOEMU_ENV="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-net-check)
      SKIP_NET_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bioemu-colabfold] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

log() {
  echo "[bioemu-colabfold] $*"
}

die() {
  echo "[bioemu-colabfold] ERROR: $*" >&2
  exit 1
}

check_dns() {
  local host="$1"
  python - "$host" <<'PY'
import socket
import sys

host = sys.argv[1]
try:
    socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
except Exception as exc:
    print(f"[bioemu-colabfold][net] DNS resolution failed for {host}: {exc}", flush=True)
    raise SystemExit(2)
print(f"[bioemu-colabfold][net] ok: {host}", flush=True)
PY
}

run_dns_preflight() {
  local failed=0
  for host in pypi.org files.pythonhosted.org storage.googleapis.com api.colabfold.com; do
    if ! check_dns "$host"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    die "network preflight failed for required BioEmu/ColabFold hosts"
  fi
}

resolve_env_python() {
  local env_name="$1"
  conda run -n "$env_name" python -c "import sys; print(sys.executable)" | awk 'NF{last=$0} END{print last}'
}

resolve_patch_dir() {
  local env_name="$1"
  conda run -n "$env_name" python -c "import os, bioemu.get_embeds as g; print(os.path.join(os.path.dirname(os.path.realpath(g.__file__)), 'colabfold_setup'))" | awk 'NF{last=$0} END{print last}'
}

check_af2_weights_ready() {
  local target="$1"
  "${target}/bin/python" - <<'PY' >/dev/null 2>&1
from pathlib import Path
from colabfold.download import default_data_dir
marker = Path(default_data_dir) / "params" / "download_finished.txt"
raise SystemExit(0 if marker.is_file() else 1)
PY
}

ensure_af2_weights() {
  local target="$1"
  "${target}/bin/python" - <<'PY'
from pathlib import Path
from colabfold.download import default_data_dir, download_alphafold_params
data_dir = Path(default_data_dir)
print(f"[bioemu-colabfold] ensuring alphafold2 weights in {data_dir}", flush=True)
download_alphafold_params("alphafold2", data_dir)
PY
}

check_target_ready() {
  local target="$1"
  [[ -x "${target}/bin/colabfold_batch" ]] || return 1
  [[ -f "${target}/.COLABFOLD_PATCHED" ]] || return 1
  "${target}/bin/python" - <<'PY' >/dev/null 2>&1
import colabfold.batch  # noqa: F401
PY
  check_af2_weights_ready "$target" || return 1
}

mkdir -p "${REPO_ROOT}/.cache/bioemu" "${REPO_ROOT}/.cache/uv" "${REPO_ROOT}/.cache/xdg" "${REPO_ROOT}/.cache/pip"
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache/xdg"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"

log "repo_root=${REPO_ROOT}"
log "target_dir=${TARGET_DIR}"
log "bioemu_env=${BIOEMU_ENV}"

if check_target_ready "$TARGET_DIR"; then
  if [[ "$FORCE" -eq 1 ]]; then
    log "existing runtime found; force enabled, rebuilding"
  else
    log "runtime already ready"
    exit 0
  fi
else
  if [[ "$CHECK_ONLY" -eq 1 ]]; then
    die "runtime not ready in check-only mode"
  fi
fi

if [[ "$SKIP_NET_CHECK" -ne 1 ]]; then
  run_dns_preflight
fi

BIOEMU_PY="$(resolve_env_python "$BIOEMU_ENV")"
PATCH_DIR="$(resolve_patch_dir "$BIOEMU_ENV")"

[[ -x "$BIOEMU_PY" ]] || die "bioemu env python not found: $BIOEMU_PY"
[[ -d "$PATCH_DIR" ]] || die "bioemu patch directory not found: $PATCH_DIR"
[[ -f "${PATCH_DIR}/modules.patch" ]] || die "missing modules.patch in ${PATCH_DIR}"
[[ -f "${PATCH_DIR}/batch.patch" ]] || die "missing batch.patch in ${PATCH_DIR}"
log "using bioemu python=${BIOEMU_PY}"

if [[ -d "$TARGET_DIR" ]]; then
  rm -rf "$TARGET_DIR"
fi

log "creating venv with ${BIOEMU_PY}"
"$BIOEMU_PY" -m venv --without-pip "$TARGET_DIR"

log "installing colabfold package set"
"$BIOEMU_PY" -m uv pip install --python "${TARGET_DIR}/bin/python" "colabfold[alphafold-minus-jax]==1.5.4"

log "installing CUDA/JAX pinned packages"
"$BIOEMU_PY" -m uv pip install --python "${TARGET_DIR}/bin/python" --force-reinstall \
  "jax[cuda12]==0.4.35" \
  "numpy==1.26.4" \
  "nvidia-cublas-cu12==12.8.4.1" \
  "nvidia-cuda-cupti-cu12==12.8.90" \
  "nvidia-cuda-nvcc-cu12==12.8.93" \
  "nvidia-cuda-runtime-cu12==12.8.90" \
  "nvidia-cudnn-cu12==9.8.0.87" \
  "nvidia-cufft-cu12==11.3.3.83" \
  "nvidia-cusolver-cu12==11.7.3.90" \
  "nvidia-cusparse-cu12==12.5.8.93" \
  "nvidia-nccl-cu12==2.26.2.post1" \
  "nvidia-nvjitlink-cu12==12.8.93"

SITE_DIR="$(echo "${TARGET_DIR}"/lib/python3.*/site-packages)"
[[ -d "$SITE_DIR" ]] || die "site-packages not found under ${TARGET_DIR}"

log "applying BioEmu colabfold patches"
patch "${SITE_DIR}/alphafold/model/modules.py" "${PATCH_DIR}/modules.patch"
patch "${SITE_DIR}/colabfold/batch.py" "${PATCH_DIR}/batch.patch"
touch "${TARGET_DIR}/.COLABFOLD_PATCHED"
ensure_af2_weights "$TARGET_DIR"

if ! check_target_ready "$TARGET_DIR"; then
  die "runtime verification failed after install"
fi

log "runtime ready at ${TARGET_DIR}"
