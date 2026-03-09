#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SWITCH=""
TOTAL_DESIGNS=3000
LENGTH_RANGE="20-400"
OUT_ROOT="$REPO_ROOT/data/rfd3"
CKPT_PATH="rfd3"
RFD3_BIN="rfd3"
DIFFUSION_BATCH_SIZE=1
SKIP_EXISTING="True"
PREVALIDATE_INPUTS="True"
DUMP_TRAJECTORIES="False"
LOW_MEMORY_MODE="True"
AUTO_OOM_RETRY="True"
OOM_RETRIES=4
SHOW_PROGRESS="True"
PROGRESS_INTERVAL=15
DIMER_MODE="asymmetric"
LIGAND_LIST=""
BINDERS_PER_LIGAND=""
DRY_RUN=0
declare -a EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run RFdiffusion3 inference in one of three modes using --switch:
  1) unconditional monomer (20-400 aa by default)
  2) unconditional dimer (default: asymmetric 2-chain; optional C2 symmetry)
  3) small-molecule binder monomer generation from a ligand reference list

Usage:
  scripts/run_rfd3_inference.sh --switch <monomer|dimer|ligand> [options]

Common options:
  --switch MODE                 Required. One of: monomer, dimer, ligand
  --total-designs N             Target designs for monomer/dimer (default: 3000)
                                For ligand mode, used only when --binders-per-ligand
                                is not provided (split across ligands).
  --length-range MIN-MAX        Protein length range (default: 20-400)
  --out-root PATH               Root output dir (default: ./data/rfd3 relative to repo root)
  --ckpt-path PATH              RFD3 checkpoint path/name (default: rfd3)
  --rfd3-bin BIN                RFD3 CLI binary (default: rfd3)
  --diffusion-batch-size N      Preferred diffusion_batch_size (default: 1)
  --skip-existing True|False    Pass-through to rfd3 design (default: True)
  --prevalidate-inputs True|False
                                Pass-through to rfd3 design (default: True)
  --dump-trajectories True|False
                                Pass-through to rfd3 design (default: False)
  --low-memory-mode True|False  Pass-through to rfd3 design (default: True)
  --auto-oom-retry True|False   Retry with lower memory settings on CUDA OOM
                                (default: True)
  --oom-retries N               Max OOM retries (default: 4)
  --show-progress True|False    Show live progress tracking (default: True)
  --progress-interval SEC       Progress refresh interval (default: 15)
  --dimer-mode MODE             For --switch dimer: asymmetric|symmetric_c2
                                (default: asymmetric)
  --extra-arg KEY=VALUE         Additional Hydra override; can be repeated
  --dry-run                     Print commands only, do not run inference
  --help                        Show this help message

Ligand mode options:
  --ligand-list FILE            Required for --switch ligand.
                                Format per non-comment line:
                                  input_path ligand_code
                                OR
                                  name input_path ligand_code
  --binders-per-ligand N        Optional per-ligand designs. If omitted,
                                total designs are split across ligands.

Examples:
  scripts/run_rfd3_inference.sh --switch monomer
  scripts/run_rfd3_inference.sh --switch dimer --length-range 40-240
  scripts/run_rfd3_inference.sh --switch dimer --dimer-mode symmetric_c2
  scripts/run_rfd3_inference.sh --switch ligand --ligand-list data/ligands.tsv --binders-per-ligand 100
  scripts/run_rfd3_inference.sh --switch monomer --extra-arg inference_sampler.num_timesteps=100
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

log_info() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" >&2
}

format_seconds() {
  local total="$1"
  printf '%02d:%02d:%02d' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

json_escape() {
  local raw="$1"
  raw="${raw//\\/\\\\}"
  raw="${raw//\"/\\\"}"
  raw="${raw//$'\n'/\\n}"
  printf '%s' "$raw"
}

sanitize_name() {
  local raw="$1"
  local clean
  clean="$(printf '%s' "$raw" | tr -cs 'A-Za-z0-9._-' '_')"
  clean="${clean#_}"
  clean="${clean%_}"
  if [[ -z "$clean" ]]; then
    clean="entry"
  fi
  printf '%s' "$clean"
}

is_truthy() {
  local v="${1:-}"
  v="${v,,}"
  [[ "$v" == "true" || "$v" == "1" || "$v" == "yes" || "$v" == "y" ]]
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

count_output_designs() {
  local out_dir="$1"
  local cif_count
  cif_count="$(find "$out_dir" -maxdepth 1 -type f -name '*.cif.gz' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$cif_count" != "0" ]]; then
    printf '%s' "$cif_count"
    return 0
  fi

  find "$out_dir" -maxdepth 1 -type f -name '*.json' ! -name 'spec_*.json' ! -name '.*' 2>/dev/null | wc -l | tr -d ' '
}

monitor_progress_polling() {
  local run_pid="$1"
  local out_dir="$2"
  local baseline="$3"
  local target="$4"
  local label="$5"

  while kill -0 "$run_pid" 2>/dev/null; do
    local total_now done_now percent
    total_now="$(count_output_designs "$out_dir")"
    done_now=$((total_now - baseline))
    if (( done_now < 0 )); then
      done_now=0
    fi
    if (( done_now > target )); then
      done_now="$target"
    fi
    percent=$((100 * done_now / target))
    log_info "progress[$label]: ${done_now}/${target} (${percent}%)"
    sleep "$PROGRESS_INTERVAL"
  done
}

start_progress_monitor() {
  local run_pid="$1"
  local out_dir="$2"
  local baseline="$3"
  local target="$4"
  local label="$5"

  if ! is_truthy "$SHOW_PROGRESS"; then
    return 0
  fi

  if command -v python3 >/dev/null 2>&1 && python3 -c "import tqdm" >/dev/null 2>&1; then
    python3 - "$run_pid" "$out_dir" "$baseline" "$target" "$label" "$PROGRESS_INTERVAL" <<'PY' &
import glob
import os
import sys
import time
from tqdm import tqdm

run_pid = int(sys.argv[1])
out_dir = sys.argv[2]
baseline = int(sys.argv[3])
target = max(1, int(sys.argv[4]))
label = sys.argv[5]
interval = max(1.0, float(sys.argv[6]))

def count_outputs(path: str) -> int:
    cif_files = glob.glob(os.path.join(path, "*.cif.gz"))
    if cif_files:
        return len(cif_files)
    json_files = glob.glob(os.path.join(path, "*.json"))
    json_files = [p for p in json_files if not os.path.basename(p).startswith("spec_")]
    return len(json_files)

pbar = tqdm(total=target, desc=label, unit="design", dynamic_ncols=True, leave=False)
last = 0

while os.path.exists(f"/proc/{run_pid}"):
    done = max(0, count_outputs(out_dir) - baseline)
    done = min(done, target)
    if done > last:
        pbar.update(done - last)
        last = done
    time.sleep(interval)

done = max(0, count_outputs(out_dir) - baseline)
done = min(done, target)
if done > last:
    pbar.update(done - last)
pbar.close()
PY
    echo "$!"
    return 0
  fi

  monitor_progress_polling "$run_pid" "$out_dir" "$baseline" "$target" "$label" &
  echo "$!"
}

stop_progress_monitor() {
  local monitor_pid="$1"
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}

log_has_oom() {
  local log_file="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -qi 'outofmemoryerror|cuda out of memory' "$log_file"
  else
    grep -Eqi 'outofmemoryerror|cuda out of memory' "$log_file"
  fi
}

compute_batches() {
  local requested="$1"
  local preferred_batch="$2"
  local actual_batch
  local n_batches

  if ! is_positive_int "$requested"; then
    die "Design count must be a positive integer, got: $requested"
  fi
  if ! is_positive_int "$preferred_batch"; then
    die "Diffusion batch size must be a positive integer, got: $preferred_batch"
  fi

  if (( requested % preferred_batch == 0 )); then
    actual_batch="$preferred_batch"
    n_batches=$((requested / preferred_batch))
  else
    actual_batch=1
    n_batches="$requested"
    echo "Warning: $requested not divisible by diffusion_batch_size=$preferred_batch; using diffusion_batch_size=1 for exact count." >&2
  fi

  printf '%s %s\n' "$actual_batch" "$n_batches"
}

run_rfd3() {
  local inputs_file="$1"
  local out_dir="$2"
  local requested_designs="$3"
  shift 3
  local -a mode_overrides=("$@")

  local actual_batch n_batches
  read -r actual_batch n_batches < <(compute_batches "$requested_designs" "$DIFFUSION_BATCH_SIZE")
  local attempt=0
  local run_batch="$actual_batch"
  local run_low_memory_mode="$LOW_MEMORY_MODE"
  local run_start_ts
  run_start_ts="$(date +%s)"

  mkdir -p "$out_dir"
  local initial_outputs
  initial_outputs="$(count_output_designs "$out_dir")"
  log_info "launch: inputs=$inputs_file out_dir=$out_dir requested_designs=$requested_designs existing_outputs=$initial_outputs"

  # Helps CUDA allocator fragmentation on long runs unless user already set it.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  while true; do
    local attempt_id=$((attempt + 1))
    local -a cmd=(
      "$RFD3_BIN" design
      "out_dir=$out_dir"
      "inputs=$inputs_file"
      "ckpt_path=$CKPT_PATH"
      "skip_existing=$SKIP_EXISTING"
      "prevalidate_inputs=$PREVALIDATE_INPUTS"
      "dump_trajectories=$DUMP_TRAJECTORIES"
      "low_memory_mode=$run_low_memory_mode"
      "diffusion_batch_size=$run_batch"
      "n_batches=$n_batches"
    )

    local override
    for override in "${EXTRA_ARGS[@]}"; do
      cmd+=("$override")
    done
    for override in "${mode_overrides[@]}"; do
      cmd+=("$override")
    done

    log_info "running: requested=$requested_designs attempt=$attempt_id batch=$run_batch low_memory_mode=$run_low_memory_mode"
    echo "Running ($requested_designs designs, attempt $attempt_id): ${cmd[*]}"
    if (( DRY_RUN )); then
      return 0
    fi

    local run_log="$out_dir/.rfd3_run_attempt_${attempt_id}.log"
    local monitor_pid=""
    local attempt_start_ts
    attempt_start_ts="$(date +%s)"

    local target_new="$requested_designs"
    if (( target_new < 1 )); then
      target_new=1
    fi

    set +e
    (
      "${cmd[@]}" 2>&1 | tee "$run_log"
    ) &
    local run_pid="$!"
    monitor_pid="$(start_progress_monitor "$run_pid" "$out_dir" "$initial_outputs" "$target_new" "rfd3:$attempt_id")"
    wait "$run_pid"
    local status="$?"
    stop_progress_monitor "$monitor_pid"
    set -e

    local outputs_now produced_now attempt_secs
    outputs_now="$(count_output_designs "$out_dir")"
    produced_now=$((outputs_now - initial_outputs))
    if (( produced_now < 0 )); then
      produced_now=0
    fi
    attempt_secs=$(( $(date +%s) - attempt_start_ts ))
    log_info "attempt_done: attempt=$attempt_id status=$status produced_new=$produced_now elapsed=$(format_seconds "$attempt_secs") log=$run_log"

    if (( status == 0 )); then
      local total_secs
      total_secs=$(( $(date +%s) - run_start_ts ))
      log_info "run_complete: out_dir=$out_dir produced_new=$produced_now elapsed=$(format_seconds "$total_secs")"
      return 0
    fi

    local has_oom=0
    if log_has_oom "$run_log"; then
      has_oom=1
    fi

    if (( has_oom == 0 )) || ! is_truthy "$AUTO_OOM_RETRY"; then
      echo "Error: inference failed. See $run_log" >&2
      return "$status"
    fi

    if (( attempt >= OOM_RETRIES )); then
      echo "Error: CUDA OOM persisted after $((attempt + 1)) attempts. See $run_log" >&2
      echo "Hint: free GPU memory or reduce --length-range (e.g. 20-300)." >&2
      return "$status"
    fi

    if (( run_batch > 1 )); then
      run_batch=$((run_batch / 2))
      if (( run_batch < 1 )); then
        run_batch=1
      fi
      echo "OOM detected. Retrying with diffusion_batch_size=$run_batch" >&2
    elif ! is_truthy "$run_low_memory_mode"; then
      run_low_memory_mode="True"
      echo "OOM detected. Retrying with low_memory_mode=True" >&2
    else
      echo "Error: OOM with diffusion_batch_size=1 and low_memory_mode=True. Reduce --length-range or free GPU memory." >&2
      return "$status"
    fi

    attempt=$((attempt + 1))
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --switch)
      [[ $# -ge 2 ]] || die "Missing value for --switch"
      SWITCH="$2"
      shift 2
      ;;
    --total-designs)
      [[ $# -ge 2 ]] || die "Missing value for --total-designs"
      TOTAL_DESIGNS="$2"
      shift 2
      ;;
    --length-range)
      [[ $# -ge 2 ]] || die "Missing value for --length-range"
      LENGTH_RANGE="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "Missing value for --out-root"
      OUT_ROOT="$2"
      shift 2
      ;;
    --ckpt-path)
      [[ $# -ge 2 ]] || die "Missing value for --ckpt-path"
      CKPT_PATH="$2"
      shift 2
      ;;
    --rfd3-bin)
      [[ $# -ge 2 ]] || die "Missing value for --rfd3-bin"
      RFD3_BIN="$2"
      shift 2
      ;;
    --diffusion-batch-size)
      [[ $# -ge 2 ]] || die "Missing value for --diffusion-batch-size"
      DIFFUSION_BATCH_SIZE="$2"
      shift 2
      ;;
    --skip-existing)
      [[ $# -ge 2 ]] || die "Missing value for --skip-existing"
      SKIP_EXISTING="$2"
      shift 2
      ;;
    --prevalidate-inputs)
      [[ $# -ge 2 ]] || die "Missing value for --prevalidate-inputs"
      PREVALIDATE_INPUTS="$2"
      shift 2
      ;;
    --dump-trajectories)
      [[ $# -ge 2 ]] || die "Missing value for --dump-trajectories"
      DUMP_TRAJECTORIES="$2"
      shift 2
      ;;
    --low-memory-mode)
      [[ $# -ge 2 ]] || die "Missing value for --low-memory-mode"
      LOW_MEMORY_MODE="$2"
      shift 2
      ;;
    --auto-oom-retry)
      [[ $# -ge 2 ]] || die "Missing value for --auto-oom-retry"
      AUTO_OOM_RETRY="$2"
      shift 2
      ;;
    --oom-retries)
      [[ $# -ge 2 ]] || die "Missing value for --oom-retries"
      OOM_RETRIES="$2"
      shift 2
      ;;
    --show-progress)
      [[ $# -ge 2 ]] || die "Missing value for --show-progress"
      SHOW_PROGRESS="$2"
      shift 2
      ;;
    --progress-interval)
      [[ $# -ge 2 ]] || die "Missing value for --progress-interval"
      PROGRESS_INTERVAL="$2"
      shift 2
      ;;
    --dimer-mode)
      [[ $# -ge 2 ]] || die "Missing value for --dimer-mode"
      DIMER_MODE="$2"
      shift 2
      ;;
    --ligand-list)
      [[ $# -ge 2 ]] || die "Missing value for --ligand-list"
      LIGAND_LIST="$2"
      shift 2
      ;;
    --binders-per-ligand)
      [[ $# -ge 2 ]] || die "Missing value for --binders-per-ligand"
      BINDERS_PER_LIGAND="$2"
      shift 2
      ;;
    --extra-arg)
      [[ $# -ge 2 ]] || die "Missing value for --extra-arg"
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "$SWITCH" ]] || die "--switch is required (monomer|dimer|ligand)"
case "$SWITCH" in
  monomer|dimer|ligand) ;;
  *) die "--switch must be one of: monomer, dimer, ligand" ;;
esac
case "$DIMER_MODE" in
  asymmetric|symmetric_c2) ;;
  *) die "--dimer-mode must be one of: asymmetric, symmetric_c2" ;;
esac

is_positive_int "$TOTAL_DESIGNS" || die "--total-designs must be a positive integer"
is_positive_int "$DIFFUSION_BATCH_SIZE" || die "--diffusion-batch-size must be a positive integer"
is_positive_int "$OOM_RETRIES" || die "--oom-retries must be a positive integer"
is_positive_int "$PROGRESS_INTERVAL" || die "--progress-interval must be a positive integer"
if [[ -n "$BINDERS_PER_LIGAND" ]]; then
  is_positive_int "$BINDERS_PER_LIGAND" || die "--binders-per-ligand must be a positive integer"
fi

mkdir -p "$OUT_ROOT"
log_info "config: switch=$SWITCH total_designs=$TOTAL_DESIGNS length_range=$LENGTH_RANGE out_root=$OUT_ROOT ckpt_path=$CKPT_PATH"
log_info "config: diffusion_batch_size=$DIFFUSION_BATCH_SIZE low_memory_mode=$LOW_MEMORY_MODE auto_oom_retry=$AUTO_OOM_RETRY oom_retries=$OOM_RETRIES show_progress=$SHOW_PROGRESS interval=${PROGRESS_INTERVAL}s dimer_mode=$DIMER_MODE"
if ! command -v "$RFD3_BIN" >/dev/null 2>&1; then
  if (( DRY_RUN )); then
    log_info "warning: --rfd3-bin not found in PATH (dry-run mode): $RFD3_BIN"
  else
    die "--rfd3-bin not found in PATH: $RFD3_BIN"
  fi
fi

if [[ "$SWITCH" == "monomer" ]]; then
  RUN_DIR="$OUT_ROOT/unconditional_monomer"
  mkdir -p "$RUN_DIR"
  SPEC_PATH="$RUN_DIR/spec_unconditional_monomer.json"
  LENGTH_ESC="$(json_escape "$LENGTH_RANGE")"
  cat > "$SPEC_PATH" <<EOF
{
  "unconditional_monomer": {
    "length": "$LENGTH_ESC"
  }
}
EOF
  log_info "prepared_spec: $SPEC_PATH"
  run_rfd3 "$SPEC_PATH" "$RUN_DIR" "$TOTAL_DESIGNS"
  exit 0
fi

if [[ "$SWITCH" == "dimer" ]]; then
  if [[ "$DIMER_MODE" == "asymmetric" ]]; then
    RUN_DIR="$OUT_ROOT/unconditional_dimer_asymmetric"
    mkdir -p "$RUN_DIR"
    SPEC_PATH="$RUN_DIR/spec_unconditional_dimer_asymmetric.json"
    CONTIG_ESC="$(json_escape "${LENGTH_RANGE},/0,${LENGTH_RANGE}")"
    cat > "$SPEC_PATH" <<EOF
{
  "unconditional_dimer_asymmetric": {
    "dialect": 1,
    "contig": "$CONTIG_ESC",
    "length": null
  }
}
EOF
    log_info "prepared_spec: $SPEC_PATH"
    run_rfd3 "$SPEC_PATH" "$RUN_DIR" "$TOTAL_DESIGNS"
  else
    RUN_DIR="$OUT_ROOT/unconditional_dimer_c2"
    mkdir -p "$RUN_DIR"
    SPEC_PATH="$RUN_DIR/spec_unconditional_dimer_c2.json"
    LENGTH_ESC="$(json_escape "$LENGTH_RANGE")"
    cat > "$SPEC_PATH" <<EOF
{
  "unconditional_dimer_c2": {
    "length": "$LENGTH_ESC",
    "is_non_loopy": true,
    "symmetry": {
      "id": "C2"
    }
  }
}
EOF
    log_info "prepared_spec: $SPEC_PATH"
    run_rfd3 "$SPEC_PATH" "$RUN_DIR" "$TOTAL_DESIGNS" "inference_sampler.kind=symmetry"
  fi
  exit 0
fi

[[ -n "$LIGAND_LIST" ]] || die "--ligand-list is required when --switch ligand"
[[ -f "$LIGAND_LIST" ]] || die "Ligand list not found: $LIGAND_LIST"

RUN_DIR="$OUT_ROOT/small_molecule_binders"
mkdir -p "$RUN_DIR"

declare -a LIGAND_NAMES=()
declare -a LIGAND_INPUTS=()
declare -a LIGAND_CODES=()

LINE_NUM=0
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  LINE_NUM=$((LINE_NUM + 1))
  line="${raw_line%%#*}"
  if [[ -z "${line//[[:space:]]/}" ]]; then
    continue
  fi

  read -r c1 c2 c3 c4 <<< "$line"
  [[ -z "${c4:-}" ]] || die "Too many columns in $LIGAND_LIST:$LINE_NUM"

  name=""
  input_path=""
  ligand_code=""
  if [[ -n "${c3:-}" ]]; then
    name="$c1"
    input_path="$c2"
    ligand_code="$c3"
  else
    name="ligand_${LINE_NUM}"
    input_path="$c1"
    ligand_code="$c2"
  fi

  if [[ ! -f "$input_path" ]]; then
    candidate="$(cd "$(dirname "$LIGAND_LIST")" && pwd)/$input_path"
    if [[ -f "$candidate" ]]; then
      input_path="$candidate"
    else
      die "Missing input structure '$input_path' from $LIGAND_LIST:$LINE_NUM"
    fi
  fi

  # Canonicalize to absolute path so downstream JSON parsing is independent of
  # the spec file location.
  if command -v realpath >/dev/null 2>&1; then
    input_path="$(realpath "$input_path")"
  else
    input_path="$(cd "$(dirname "$input_path")" && pwd)/$(basename "$input_path")"
  fi

  LIGAND_NAMES+=("$name")
  LIGAND_INPUTS+=("$input_path")
  LIGAND_CODES+=("$ligand_code")
done < "$LIGAND_LIST"

N_LIGANDS="${#LIGAND_NAMES[@]}"
(( N_LIGANDS > 0 )) || die "No ligand entries found in $LIGAND_LIST"
log_info "ligand_list: entries=$N_LIGANDS file=$LIGAND_LIST"

if [[ -z "$BINDERS_PER_LIGAND" ]] && (( TOTAL_DESIGNS < N_LIGANDS )); then
  die "--total-designs ($TOTAL_DESIGNS) is smaller than number of ligands ($N_LIGANDS); increase total or set --binders-per-ligand"
fi

BASE_PER_LIGAND=0
REMAINDER=0
if [[ -z "$BINDERS_PER_LIGAND" ]]; then
  BASE_PER_LIGAND=$((TOTAL_DESIGNS / N_LIGANDS))
  REMAINDER=$((TOTAL_DESIGNS % N_LIGANDS))
fi

for ((i=0; i<N_LIGANDS; i++)); do
  count=0
  if [[ -n "$BINDERS_PER_LIGAND" ]]; then
    count="$BINDERS_PER_LIGAND"
  else
    count="$BASE_PER_LIGAND"
    if (( i < REMAINDER )); then
      count=$((count + 1))
    fi
  fi

  idx=$((i + 1))
  safe_name="$(sanitize_name "${LIGAND_NAMES[$i]}")"
  run_tag="$(printf 'ligand_%03d_%s' "$idx" "$safe_name")"
  ligand_subdir="$RUN_DIR/$run_tag"
  mkdir -p "$ligand_subdir"
  spec_file="$ligand_subdir/spec_${run_tag}.json"
  log_info "ligand_job[$idx/$N_LIGANDS]: tag=$run_tag ligand=${LIGAND_CODES[$i]} input=${LIGAND_INPUTS[$i]} target_designs=$count"

  key_esc="$(json_escape "$run_tag")"
  input_esc="$(json_escape "${LIGAND_INPUTS[$i]}")"
  ligand_esc="$(json_escape "${LIGAND_CODES[$i]}")"
  length_esc="$(json_escape "$LENGTH_RANGE")"
  cat > "$spec_file" <<EOF
{
  "$key_esc": {
    "input": "$input_esc",
    "ligand": "$ligand_esc",
    "length": "$length_esc",
    "select_fixed_atoms": {
      "$ligand_esc": "ALL"
    }
  }
}
EOF

  log_info "prepared_spec: $spec_file"
  run_rfd3 "$spec_file" "$ligand_subdir" "$count"
done
