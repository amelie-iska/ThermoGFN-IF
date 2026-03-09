#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_HOME_DEFAULT="${REPO_ROOT}/.cache/huggingface"
PREFETCH_TIMEOUT_SEC="${PREFETCH_TIMEOUT_SEC:-1800}"
PREFETCH_HEARTBEAT_SEC="${PREFETCH_HEARTBEAT_SEC:-15}"
PREFETCH_SKIP_NET_CHECK="${PREFETCH_SKIP_NET_CHECK:-0}"
BIOEMU_ENV_NAME="${BIOEMU_ENV_NAME:-bioemu}"
BIOEMU_COLABFOLD_DIR="${BIOEMU_COLABFOLD_DIR:-${REPO_ROOT}/.cache/bioemu/colabfold}"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

LOCK_FILE="${REPO_ROOT}/.cache/prefetch_production_oracles.lock"
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[prefetch] another prefetch process is already running (lock: $LOCK_FILE)" >&2
  echo "[prefetch] stop that process first, or wait for completion." >&2
  exit 3
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/env/prefetch_production_oracles.sh [options]

Options:
  --timeout-sec N      Per-step timeout in seconds (default: env PREFETCH_TIMEOUT_SEC or 1800)
  --heartbeat-sec N    Heartbeat print interval in seconds (default: env PREFETCH_HEARTBEAT_SEC or 15)
  --bioemu-env NAME    BioEmu conda env name (default: env BIOEMU_ENV_NAME or bioemu)
  --colabfold-dir PATH ColabFold runtime dir (default: env BIOEMU_COLABFOLD_DIR or ./.cache/bioemu/colabfold)
  --skip-net-check     Skip DNS preflight checks for external model hosts
  -h, --help           Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout-sec)
      PREFETCH_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --heartbeat-sec)
      PREFETCH_HEARTBEAT_SEC="$2"
      shift 2
      ;;
    --bioemu-env)
      BIOEMU_ENV_NAME="$2"
      shift 2
      ;;
    --colabfold-dir)
      BIOEMU_COLABFOLD_DIR="$2"
      shift 2
      ;;
    --skip-net-check)
      PREFETCH_SKIP_NET_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[prefetch] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

run_with_heartbeat() {
  local label="$1"
  shift
  local start_ts
  start_ts="$(date +%s)"
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$PREFETCH_HEARTBEAT_SEC"
    if kill -0 "$pid" 2>/dev/null; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      echo "[prefetch][$label] still running (${elapsed}s elapsed)"
    fi
  done
  wait "$pid"
  local rc=$?
  local end elapsed_total
  end="$(date +%s)"
  elapsed_total=$((end - start_ts))
  if [[ "$rc" -ne 0 ]]; then
    echo "[prefetch][$label] failed rc=${rc} after ${elapsed_total}s" >&2
  else
    echo "[prefetch][$label] completed in ${elapsed_total}s"
  fi
  return "$rc"
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
    print(f"[prefetch][net] DNS resolution failed for {host}: {exc}", flush=True)
    sys.exit(2)
print(f"[prefetch][net] ok: {host}", flush=True)
PY
}

run_network_preflight() {
  local failed=0
  local required_hosts=(
    "huggingface.co"
    "dl.fbaipublicfiles.com"
    "api.colabfold.com"
    "storage.googleapis.com"
  )
  if [[ ! -x "${BIOEMU_COLABFOLD_DIR}/bin/colabfold_batch" ]] || [[ ! -f "${BIOEMU_COLABFOLD_DIR}/.COLABFOLD_PATCHED" ]]; then
    required_hosts+=(
      "pypi.org"
      "files.pythonhosted.org"
    )
  fi
  echo "[prefetch] network preflight: DNS checks"
  for host in "${required_hosts[@]}"; do
    if ! check_dns "$host"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[prefetch][net] preflight failed. Required hosts are unavailable for checkpoint downloads and/or BioEmu ColabFold runtime setup." >&2
    echo "[prefetch][net] fix DNS/network and rerun. Use --skip-net-check only if hosts are intentionally blocked and all artifacts are already cached." >&2
    return 2
  fi
  return 0
}

echo "[prefetch] REPO_ROOT=$REPO_ROOT"
echo "[prefetch] HF_HOME=$HF_HOME"
echo "[prefetch] TORCH_HOME=$TORCH_HOME"
echo "[prefetch] timeout=${PREFETCH_TIMEOUT_SEC}s heartbeat=${PREFETCH_HEARTBEAT_SEC}s"
echo "[prefetch] bioemu_env=${BIOEMU_ENV_NAME}"
echo "[prefetch] colabfold_dir=${BIOEMU_COLABFOLD_DIR}"
if [[ "$PREFETCH_SKIP_NET_CHECK" -ne 1 ]]; then
  run_network_preflight
fi

echo "[prefetch] LigandMPNN model parameters"
echo "[prefetch][ligandmpnn] ensuring model_params directory is populated"
if [[ -d "${REPO_ROOT}/models/LigandMPNN" ]]; then
  run_with_heartbeat "ligandmpnn" timeout "${PREFETCH_TIMEOUT_SEC}" bash -lc "
    cd '${REPO_ROOT}/models/LigandMPNN'
    if [[ ! -f './model_params/ligandmpnn_v_32_010_25.pt' ]]; then
      bash get_model_params.sh './model_params'
    else
      echo '[prefetch][ligandmpnn] checkpoint already present: ./model_params/ligandmpnn_v_32_010_25.pt'
    fi
  "
else
  echo '[prefetch][ligandmpnn] skipped: models/LigandMPNN not found'
fi

echo "[prefetch] SPURS base model + multi model"
echo "[prefetch][spurs] launching prefetch worker"
run_with_heartbeat "spurs" timeout "${PREFETCH_TIMEOUT_SEC}" conda run --no-capture-output -n spurs bash -lc "
  export PYTHONUNBUFFERED=1
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export TORCH_HOME='$TORCH_HOME'
  export HF_HUB_ETAG_TIMEOUT='$HF_HUB_ETAG_TIMEOUT'
  export HF_HUB_DOWNLOAD_TIMEOUT='$HF_HUB_DOWNLOAD_TIMEOUT'
  python -u - <<'PY'
import sys
from time import perf_counter
import torch.hub as _torch_hub

_orig_load_state_dict_from_url = _torch_hub.load_state_dict_from_url
_orig_download_url_to_file = _torch_hub.download_url_to_file

def _download_url_to_file_progress(url, dst, *args, **kwargs):
    print(f'[prefetch][spurs][bytes] downloading url={url} -> {dst}', flush=True)
    # Force byte-level progress bar from torch hub.
    if len(args) >= 2:
        args = list(args)
        args[1] = True
        args = tuple(args)
    else:
        kwargs['progress'] = True
    return _orig_download_url_to_file(url, dst, *args, **kwargs)

def _load_state_dict_from_url_progress(url, *args, **kwargs):
    # ESM helper passes progress=False; override to show transfer progress.
    kwargs['progress'] = True
    return _orig_load_state_dict_from_url(url, *args, **kwargs)

_torch_hub.download_url_to_file = _download_url_to_file_progress
_torch_hub.load_state_dict_from_url = _load_state_dict_from_url_progress

from spurs.inference import get_SPURS_from_hub, get_SPURS_multi_from_hub

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

models = [
    ('spurs', get_SPURS_from_hub),
    ('spurs_multi', get_SPURS_multi_from_hub),
]
t0 = perf_counter()
for name, fn in tqdm(models, total=len(models), desc='spurs:prefetch', dynamic_ncols=True, disable=False, mininterval=0.5):
    print(f'[prefetch][spurs] fetching {name}', flush=True)
    fn(repo_id='cyclization9/SPURS')
    print(f'[prefetch][spurs] ready {name}', flush=True)
print(f'[prefetch][spurs] complete elapsed={perf_counter()-t0:.2f}s', flush=True)
PY
"

echo "[prefetch] BioEmu checkpoints"
echo "[prefetch][bioemu] launching prefetch worker"
run_with_heartbeat "bioemu" timeout "${PREFETCH_TIMEOUT_SEC}" conda run --no-capture-output -n "${BIOEMU_ENV_NAME}" bash -lc "
  export PYTHONUNBUFFERED=1
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export TORCH_HOME='$TORCH_HOME'
  export HF_HUB_ETAG_TIMEOUT='$HF_HUB_ETAG_TIMEOUT'
  export HF_HUB_DOWNLOAD_TIMEOUT='$HF_HUB_DOWNLOAD_TIMEOUT'
  python -u - <<'PY'
from time import perf_counter
from bioemu.model_utils import maybe_download_checkpoint
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

models = ('bioemu-v1.0', 'bioemu-v1.1', 'bioemu-v1.2')
t0 = perf_counter()
for model in tqdm(models, total=len(models), desc='bioemu:prefetch', dynamic_ncols=True, disable=False, mininterval=0.5):
    print(f'[prefetch][bioemu] fetching {model}', flush=True)
    ckpt, cfg = maybe_download_checkpoint(model_name=model)
    print(f'[prefetch][bioemu] ready {model} ckpt={ckpt} cfg={cfg}', flush=True)
print(f'[prefetch][bioemu] complete elapsed={perf_counter()-t0:.2f}s', flush=True)
PY
"

echo "[prefetch] BioEmu ColabFold runtime"
run_with_heartbeat "bioemu-colabfold" timeout "${PREFETCH_TIMEOUT_SEC}" bash -lc "
  cd '${REPO_ROOT}'
  ./scripts/env/setup_bioemu_colabfold_runtime.sh \
    --bioemu-env '${BIOEMU_ENV_NAME}' \
    --target-dir '${BIOEMU_COLABFOLD_DIR}'
"

echo "[prefetch] UMA model unit"
echo "[prefetch][uma-qc] launching prefetch worker"
run_with_heartbeat "uma-qc" timeout "${PREFETCH_TIMEOUT_SEC}" conda run --no-capture-output -n uma-qc bash -lc "
  export PYTHONUNBUFFERED=1
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export TORCH_HOME='$TORCH_HOME'
  export HF_HUB_ETAG_TIMEOUT='$HF_HUB_ETAG_TIMEOUT'
  export HF_HUB_DOWNLOAD_TIMEOUT='$HF_HUB_DOWNLOAD_TIMEOUT'
  python -u - <<'PY'
from time import perf_counter
from fairchem.core import pretrained_mlip
t0 = perf_counter()
print('[prefetch][uma] fetching uma-s-1p1', flush=True)
predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device='cpu')
print(f'[prefetch][uma] ready type={type(predictor).__name__}', flush=True)
print(f'[prefetch][uma] complete elapsed={perf_counter()-t0:.2f}s', flush=True)
PY
"

echo "[prefetch] done"
