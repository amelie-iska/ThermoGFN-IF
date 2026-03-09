#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${ROOT_DIR}/data/rfd3/unconditional_monomer"
OUTPUT_DIR="${ROOT_DIR}/data/rfd3_splits/unconditional_monomer_protrek35m"
FOLDSEEK_BIN="${ROOT_DIR}/models/ProTrek/bin/foldseek"
WEIGHTS_DIR="${ROOT_DIR}/models/ProTrek/weights/ProTrek_35M"

SEQ_THRESHOLD="${SEQ_THRESHOLD:-0.90}"
STRUCTURE_THRESHOLD="${STRUCTURE_THRESHOLD:-0.90}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
SAVE_EMBEDDINGS="${SAVE_EMBEDDINGS:-0}"
VERBOSE="${VERBOSE:-0}"

CMD=(
  python "${ROOT_DIR}/scripts/protrek_cluster_split.py"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --foldseek-bin "${FOLDSEEK_BIN}"
  --weights-dir "${WEIGHTS_DIR}"
  --seq-threshold "${SEQ_THRESHOLD}"
  --structure-threshold "${STRUCTURE_THRESHOLD}"
  --test-fraction "${TEST_FRACTION}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable in the current shell. "
        "Activate the protrek env directly (`conda activate protrek`) and do not use `conda run`."
    )
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${SAVE_EMBEDDINGS}" == "1" ]]; then
  CMD+=(--save-embeddings)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  CMD+=(--verbose)
fi

echo "Running ProTrek split pipeline:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"
