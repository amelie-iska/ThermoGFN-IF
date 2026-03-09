#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "/home/iska/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /home/iska/miniconda3/etc/profile.d/conda.sh
else
  echo "ERROR: conda.sh not found at /home/iska/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate bioemu

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache/xdg}"
export BIOEMU_COLABFOLD_DIR="${BIOEMU_COLABFOLD_DIR:-$REPO_ROOT/.cache/bioemu/colabfold}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/bioemu_vram_profile}"
INPUT_JSONL="${INPUT_JSONL:-runs/thermogfn_ligandmpnn/bootstrap/D_0_train.jsonl}"
LENGTH_TARGETS="${LENGTH_TARGETS:-100,180,260,340,400}"
BATCH_SIZE_100="${BATCH_SIZE_100:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TARGET_VRAM_FRAC="${TARGET_VRAM_FRAC:-0.90}"
MODEL_NAME="${MODEL_NAME:-bioemu-v1.1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FILTER_SAMPLES="${FILTER_SAMPLES:-0}"

python "$REPO_ROOT/scripts/bench/bioemu_vram_profile.py" \
  --input-jsonl "$INPUT_JSONL" \
  --output-root "$OUTPUT_ROOT" \
  --length-targets "$LENGTH_TARGETS" \
  --batch-size-100 "$BATCH_SIZE_100" \
  --num-samples "$NUM_SAMPLES" \
  --target-vram-frac "$TARGET_VRAM_FRAC" \
  --model-name "$MODEL_NAME" \
  --log-level "$LOG_LEVEL" \
  $([[ "$FILTER_SAMPLES" == "1" ]] && echo "--filter-samples" || echo "--no-filter-samples") \
  "$@"

REPORT_JSON="$REPO_ROOT/$OUTPUT_ROOT/projection_report.json"
if [[ -f "$REPORT_JSON" ]]; then
  python - <<'PY'
import json
from pathlib import Path
p = Path("runs/bioemu_vram_profile/projection_report.json")
if not p.exists():
    raise SystemExit(0)
d = json.loads(p.read_text())
proj = d.get("projection", {})
print("projection.max_length_effective_batch_1 =", proj.get("projected_max_length_at_effective_batch_1"))
print("projection.max_batch_size_100_len100    =", proj.get("projected_max_batch_size_100_at_length_100"))
print("projection.total_vram_gib               =", round(float(proj.get("total_vram_bytes", 0)) / (1024**3), 3))
PY
fi
