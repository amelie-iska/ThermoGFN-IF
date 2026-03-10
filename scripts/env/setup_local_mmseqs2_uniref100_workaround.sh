#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIBLING_SETUP="${REPO_ROOT}/../enzyme-quiver/scripts/setup_local_mmseqs2_uniref100.sh"

MSA_ROOT="${MSA_ROOT:-${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa}"
COLABFOLD_REPO_URL="${COLABFOLD_REPO_URL:-https://github.com/sokrypton/ColabFold.git}"
BOLTZ_CLIENT_REPO_URL="${BOLTZ_CLIENT_REPO_URL:-https://github.com/jwohlwend/boltz.git}"
PULL_UPDATES="${PULL_UPDATES:-false}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/env/setup_local_mmseqs2_uniref100_workaround.sh [options]

This wrapper works around a clone-order bug in the sibling
`../enzyme-quiver/scripts/setup_local_mmseqs2_uniref100.sh` by generating a
patched temporary copy of that script before running it.

Recognized options:
  --msa-root PATH  MMSeqs2 local workspace root
  -h, --help                  Show help

All other arguments are forwarded to the sibling setup script unchanged.
USAGE
}

abs_path() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

build_patched_setup() {
  local src="$1"
  local dst="$2"

  python3 - "$src" "$dst" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()

old_func = """clone_or_update_repo() {
  local repo_url="$1"
  local dst="$2"
  if [[ -d "${dst}/.git" ]]; then
    if [[ "${PULL_UPDATES}" == "true" ]]; then
      echo "[setup] updating repo: ${dst}"
      git -C "${dst}" pull --ff-only
    else
      echo "[setup] repo exists: ${dst} (skip pull; use --pull-updates to update)"
    fi
  else
    echo "[setup] cloning ${repo_url} -> ${dst}"
    git clone --depth 1 "${repo_url}" "${dst}"
  fi
}
"""

new_func = """clone_or_update_repo() {
  local repo_url="$1"
  local dst="$2"
  if [[ -d "${dst}/.git" ]]; then
    if [[ "${PULL_UPDATES}" == "true" ]]; then
      echo "[setup] updating repo: ${dst}"
      git -C "${dst}" pull --ff-only
    else
      echo "[setup] repo exists: ${dst} (skip pull; use --pull-updates to update)"
    fi
  elif [[ -d "${dst}" ]]; then
    echo "[setup] removing stale non-git directory before clone: ${dst}"
    rm -rf "${dst}"
    echo "[setup] cloning ${repo_url} -> ${dst}"
    git clone --depth 1 "${repo_url}" "${dst}"
  else
    echo "[setup] cloning ${repo_url} -> ${dst}"
    git clone --depth 1 "${repo_url}" "${dst}"
  fi
}
"""

old_mkdir = """mkdir -p "${MSA_ROOT}" "${DB_ROOT}" "${TMP_ROOT}"
mkdir -p "$(dirname "${MMSEQS_BIN}")" "$(dirname "${MMSEQS_SERVER_BIN}")"
"""

new_mkdir = """mkdir -p "${MSA_ROOT}" "${DB_ROOT}" "${TMP_ROOT}"
"""

old_clone_block = """clone_or_update_repo "${COLABFOLD_REPO_URL}" "${COLABFOLD_DIR}"
clone_or_update_repo "${BOLTZ_CLIENT_REPO_URL}" "${BOLTZ_CLIENT_DIR}"

if [[ ! -f "${MSA_SERVER_DIR}/config.json" ]]; then
"""

new_clone_block = """clone_or_update_repo "${COLABFOLD_REPO_URL}" "${COLABFOLD_DIR}"
clone_or_update_repo "${BOLTZ_CLIENT_REPO_URL}" "${BOLTZ_CLIENT_DIR}"
mkdir -p "$(dirname "${MMSEQS_BIN}")" "$(dirname "${MMSEQS_SERVER_BIN}")"

if [[ ! -f "${MSA_SERVER_DIR}/config.json" ]]; then
"""

if old_func not in text:
    raise SystemExit("expected clone_or_update_repo block not found")
if old_mkdir not in text:
    raise SystemExit("expected pre-clone mkdir block not found")
if old_clone_block not in text:
    raise SystemExit("expected clone block not found")

text = text.replace(old_func, new_func, 1)
text = text.replace(old_mkdir, new_mkdir, 1)
text = text.replace(old_clone_block, new_clone_block, 1)

dst.write_text(text)
PY

  chmod +x "${dst}"
}

forwarded_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --msa-root)
      MSA_ROOT="$2"
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${SIBLING_SETUP}" ]]; then
  echo "Missing sibling setup script: ${SIBLING_SETUP}" >&2
  exit 1
fi

MSA_ROOT="$(abs_path "${MSA_ROOT}")"
echo "[mmseqs2-setup-workaround] msa_root=${MSA_ROOT}"
patched_setup="$(mktemp /tmp/setup_local_mmseqs2_uniref100_workaround.XXXXXX.sh)"
trap 'rm -f "${patched_setup}"' EXIT

build_patched_setup "${SIBLING_SETUP}" "${patched_setup}"

echo "[mmseqs2-setup-workaround] delegating to patched copy ${patched_setup}"
exec bash "${patched_setup}" "${forwarded_args[@]}"
