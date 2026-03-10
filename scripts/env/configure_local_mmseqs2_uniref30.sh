#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MSA_ROOT="${REPO_ROOT}/../enzyme-quiver/MMseqs2/local_msa"
UNIREF30_ROOT="/opt/dlami/nvme/project-MORA/mmseqs2/databases/uniref30_2302"
CONFIG_PATH=""
WRITE_LEGACY_CONFIG=1
CLEANUP_UNIREF100=0
HOST="127.0.0.1"
PORT="8080"
LOCAL_WORKERS="4"
PARALLEL_DATABASES="2"
PARALLEL_STAGES="true"
INDEX_THREADS="${INDEX_THREADS:-$(nproc)}"
MMSEQS_TUNE_ENABLE="true"
MMSEQS_TUNE_MAX_SEQS="${MMSEQS_TUNE_MAX_SEQS:-4096}"
MMSEQS_TUNE_NUM_ITERATIONS="${MMSEQS_TUNE_NUM_ITERATIONS:-3}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/env/configure_local_mmseqs2_uniref30.sh [options]

Point the shared MMSeqs2 local workspace at an existing UniRef30 database
without downloading/building UniRef100 again.

Options:
  --msa-root PATH           Local MSA workspace root
  --uniref30-root PATH      Directory containing uniref30_2302_db*
  --config PATH             Output config path (default: <msa-root>/config.uniref30.json)
  --host HOST               Server host in generated config (default: 127.0.0.1)
  --port PORT               Server port in generated config (default: 8080)
  --local-workers N         Local worker count in generated config (default: 4)
  --parallel-databases N    Parallel databases in generated config (default: 2)
  --parallel-stages         Enable ColabFold parallel stages (default)
  --no-parallel-stages      Disable ColabFold parallel stages
  --index-threads N         Threads for one-time createindex repair if GPU .idx artifacts are missing (default: nproc)
  --mmseqs-max-seqs N       Tune MMSeqs search/gpuserver max-seqs (default: 4096)
  --mmseqs-num-iterations N Tune MMSeqs search num-iterations (default: 3)
  --disable-mmseqs-tune     Disable the MMSeqs tuned wrapper and use the raw binary
  --no-legacy-config        Do not also write <msa-root>/config.uniref100.json
  --cleanup-uniref100       Remove accidental local uniref100_db* artifacts under <msa-root>
  -h, --help                Show help
USAGE
}

abs_path() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --msa-root)
      MSA_ROOT="$2"; shift 2 ;;
    --uniref30-root)
      UNIREF30_ROOT="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --host)
      HOST="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --local-workers)
      LOCAL_WORKERS="$2"; shift 2 ;;
    --parallel-databases)
      PARALLEL_DATABASES="$2"; shift 2 ;;
    --parallel-stages)
      PARALLEL_STAGES="true"; shift ;;
    --no-parallel-stages)
      PARALLEL_STAGES="false"; shift ;;
    --index-threads)
      INDEX_THREADS="$2"; shift 2 ;;
    --mmseqs-max-seqs)
      MMSEQS_TUNE_MAX_SEQS="$2"; shift 2 ;;
    --mmseqs-num-iterations)
      MMSEQS_TUNE_NUM_ITERATIONS="$2"; shift 2 ;;
    --disable-mmseqs-tune)
      MMSEQS_TUNE_ENABLE="false"; shift ;;
    --no-legacy-config)
      WRITE_LEGACY_CONFIG=0; shift ;;
    --cleanup-uniref100)
      CLEANUP_UNIREF100=1; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MSA_ROOT="$(abs_path "${MSA_ROOT}")"
UNIREF30_ROOT="$(abs_path "${UNIREF30_ROOT}")"
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${MSA_ROOT}/config.uniref30.json"
fi
CONFIG_PATH="$(abs_path "${CONFIG_PATH}")"
LEGACY_CONFIG_PATH="${MSA_ROOT}/config.uniref100.json"

MSA_SERVER_DIR="${MSA_ROOT}/ColabFold/MsaServer"
TEMPLATE_CONFIG="${MSA_SERVER_DIR}/config.json"
MMSEQS_BIN="${MSA_SERVER_DIR}/bin/mmseqs"
MMSEQS_SERVER_BIN="${MSA_SERVER_DIR}/bin/mmseqs-server"
MMSEQS_FOR_CONFIG="${MMSEQS_BIN}"
UNIREF30_DB="${UNIREF30_ROOT}/uniref30_2302_db"
UNIREF30_DB_PAD="${UNIREF30_ROOT}/uniref30_2302_db_pad"
RESULTS_DIR="${MSA_ROOT}/jobs"

if [[ ! -f "${TEMPLATE_CONFIG}" ]]; then
  echo "Missing MsaServer template config: ${TEMPLATE_CONFIG}" >&2
  echo "Install the local workspace first, for example:" >&2
  echo "  bash scripts/env/setup_local_mmseqs2_uniref100_workaround.sh --msa-root \"${MSA_ROOT}\" --gpu-binary --skip-db-download" >&2
  exit 1
fi
if [[ ! -x "${MMSEQS_BIN}" ]]; then
  echo "Missing mmseqs binary: ${MMSEQS_BIN}" >&2
  exit 1
fi
if [[ ! -x "${MMSEQS_SERVER_BIN}" ]]; then
  echo "Missing mmseqs-server binary: ${MMSEQS_SERVER_BIN}" >&2
  exit 1
fi

SELECTED_DB="${UNIREF30_DB_PAD}"
if [[ ! -e "${UNIREF30_DB_PAD}" ]]; then
  SELECTED_DB="${UNIREF30_DB}"
fi
if [[ ! -e "${SELECTED_DB}" ]]; then
  echo "Could not find UniRef30 database under: ${UNIREF30_ROOT}" >&2
  echo "Expected ${UNIREF30_DB} or ${UNIREF30_DB_PAD}" >&2
  exit 1
fi
if [[ "${SELECTED_DB}" == "${UNIREF30_DB_PAD}" && ! -e "${UNIREF30_DB_PAD}.idx" && ! -e "${UNIREF30_DB_PAD}.idx.0" ]]; then
  echo "[mmseqs2-uniref30-config] missing GPU .idx artifacts for ${UNIREF30_DB_PAD}; running one-time createindex repair (threads=${INDEX_THREADS})"
  mkdir -p "${MSA_ROOT}/tmp"
  "${MMSEQS_BIN}" createindex "${UNIREF30_DB_PAD}" "${MSA_ROOT}/tmp" --threads "${INDEX_THREADS}"
fi
if [[ "${SELECTED_DB}" == "${UNIREF30_DB_PAD}" && ! -e "${UNIREF30_DB_PAD}.idx" && ! -e "${UNIREF30_DB_PAD}.idx.0" ]]; then
  echo "GPU padded UniRef30 DB is present but GPU .idx artifacts are still missing after createindex: ${UNIREF30_DB_PAD}" >&2
  exit 1
fi

if [[ "${CLEANUP_UNIREF100}" -eq 1 ]]; then
  echo "[mmseqs2-uniref30-config] removing accidental local UniRef100 artifacts under ${MSA_ROOT}"
  find "${MSA_ROOT}/databases" -maxdepth 1 \( -type f -o -type l \) -name 'uniref100_db*' -print -delete 2>/dev/null || true
  find "${MSA_ROOT}/tmp" -maxdepth 3 \( -type f -o -type l \) \( -name 'uniref100*' -o -name 'download.sh' -o -name 'version' \) -print -delete 2>/dev/null || true
fi

if [[ "${MMSEQS_TUNE_ENABLE}" == "true" ]]; then
  if ! [[ "${MMSEQS_TUNE_MAX_SEQS}" =~ ^[0-9]+$ ]] || (( MMSEQS_TUNE_MAX_SEQS <= 0 )); then
    echo "Invalid --mmseqs-max-seqs value: ${MMSEQS_TUNE_MAX_SEQS}" >&2
    exit 2
  fi
  if ! [[ "${MMSEQS_TUNE_NUM_ITERATIONS}" =~ ^[0-9]+$ ]] || (( MMSEQS_TUNE_NUM_ITERATIONS <= 0 )); then
    echo "Invalid --mmseqs-num-iterations value: ${MMSEQS_TUNE_NUM_ITERATIONS}" >&2
    exit 2
  fi
  MMSEQS_WRAPPER="${MSA_ROOT}/bin/mmseqs_tuned.sh"
  mkdir -p "$(dirname "${MMSEQS_WRAPPER}")"
  cat > "${MMSEQS_WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

REAL_MMSEQS="${MMSEQS_BIN}"
TUNE_MAX_SEQS="${MMSEQS_TUNE_MAX_SEQS}"
TUNE_NUM_ITERATIONS="${MMSEQS_TUNE_NUM_ITERATIONS}"

cmd="\${1:-}"
if [[ -z "\${cmd}" ]]; then
  exec "\${REAL_MMSEQS}"
fi
shift
args=("\$@")

rewrite_opt() {
  local key="\$1"
  local value="\$2"
  local -a out=()
  local seen=0
  local i=0
  while (( i < \${#args[@]} )); do
    if [[ "\${args[i]}" == "\${key}" ]]; then
      out+=("\${key}" "\${value}")
      seen=1
      i=\$((i + 2))
      continue
    fi
    out+=("\${args[i]}")
    i=\$((i + 1))
  done
  if (( seen == 0 )); then
    out+=("\${key}" "\${value}")
  fi
  args=("\${out[@]}")
}

case "\${cmd}" in
  search)
    rewrite_opt "--num-iterations" "\${TUNE_NUM_ITERATIONS}"
    rewrite_opt "--max-seqs" "\${TUNE_MAX_SEQS}"
    ;;
  ungappedprefilter|gpuserver)
    rewrite_opt "--max-seqs" "\${TUNE_MAX_SEQS}"
    ;;
esac

exec "\${REAL_MMSEQS}" "\${cmd}" "\${args[@]}"
EOF
  chmod +x "${MMSEQS_WRAPPER}"
  MMSEQS_FOR_CONFIG="${MMSEQS_WRAPPER}"
fi

mkdir -p "${RESULTS_DIR}" "${MSA_SERVER_DIR}/databases"
ln -sfn "${UNIREF30_DB}" "${MSA_SERVER_DIR}/databases/uniref30_2302_db"
if [[ -e "${UNIREF30_DB_PAD}" ]]; then
  ln -sfn "${UNIREF30_DB_PAD}" "${MSA_SERVER_DIR}/databases/uniref30_2302_db_pad"
fi

python3 - "$TEMPLATE_CONFIG" "$CONFIG_PATH" "$LEGACY_CONFIG_PATH" "$SELECTED_DB" "$MMSEQS_FOR_CONFIG" "$HOST" "$PORT" "$RESULTS_DIR" "$LOCAL_WORKERS" "$PARALLEL_DATABASES" "$PARALLEL_STAGES" "$WRITE_LEGACY_CONFIG" <<'PY'
import json
import os
import pathlib
import sys

def strip_json_comments(text: str) -> str:
    out = []
    i = 0
    in_string = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
                out.append(c)
            i += 1
            continue
        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)

def strip_trailing_commas(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)

src_cfg = pathlib.Path(sys.argv[1])
out_cfg = pathlib.Path(sys.argv[2])
legacy_cfg = pathlib.Path(sys.argv[3])
db_prefix = os.path.abspath(sys.argv[4])
mmseqs_bin = os.path.abspath(sys.argv[5])
host = sys.argv[6]
port = int(sys.argv[7])
results_dir = os.path.abspath(sys.argv[8])
local_workers = int(sys.argv[9])
parallel_databases = int(sys.argv[10])
parallel_stages = str(sys.argv[11]).strip().lower() in {"1", "true", "yes", "on"}
write_legacy = str(sys.argv[12]).strip() == "1"

raw = src_cfg.read_text(encoding="utf-8")
cfg = json.loads(strip_trailing_commas(strip_json_comments(raw)))

paths = cfg.setdefault("paths", {})
colabfold = paths.setdefault("colabfold", {})
colabfold["uniref"] = db_prefix
colabfold["environmental"] = ""
colabfold["pdb"] = ""
colabfold["pdb70"] = ""
colabfold["pdbdivided"] = ""
colabfold["pdbobsolete"] = ""
colabfold["parallelstages"] = bool(parallel_stages)
paths["mmseqs"] = mmseqs_bin
paths["results"] = results_dir

server = cfg.setdefault("server", {})
server["address"] = f"{host}:{port}"
server["pathprefix"] = "/api/"

local_cfg = cfg.setdefault("local", {})
local_cfg["workers"] = max(1, local_workers)

worker_cfg = cfg.setdefault("worker", {})
worker_cfg["paralleldatabases"] = max(1, parallel_databases)

payload = json.dumps(cfg, indent=2) + "\n"
out_cfg.write_text(payload, encoding="utf-8")
if write_legacy:
    legacy_cfg.write_text(payload, encoding="utf-8")
PY

echo "[mmseqs2-uniref30-config] msa_root=${MSA_ROOT}"
echo "[mmseqs2-uniref30-config] selected_uniref_db=${SELECTED_DB}"
echo "[mmseqs2-uniref30-config] wrote config=${CONFIG_PATH}"
if [[ "${WRITE_LEGACY_CONFIG}" -eq 1 ]]; then
  echo "[mmseqs2-uniref30-config] wrote legacy compatibility config=${LEGACY_CONFIG_PATH}"
fi
echo "[mmseqs2-uniref30-config] next:"
echo "  bash scripts/env/start_local_mmseqs2_uniref30_server.sh --msa-root \"${MSA_ROOT}\" --uniref30-root \"${UNIREF30_ROOT}\""
