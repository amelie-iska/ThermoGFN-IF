#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARED_ROOT="${ROOT_DIR}/runs/rf3_reactzyme_inputs_smiles_full_with_msa_v7"
REACTANT_ROOT="${ROOT_DIR}/runs/rf3_reactzyme_out_smiles_full_sharded_v9/reactant"
PRODUCT_ROOT="${ROOT_DIR}/runs/rf3_reactzyme_out_smiles_full_sharded_v9/product"
OUTPUT_DIR="${ROOT_DIR}/rfd3-data/rfd3_splits/rf3_reactzyme_protrek35m"
FOLDSEEK_BIN="${ROOT_DIR}/models/ProTrek/bin/foldseek"
WEIGHTS_DIR="${ROOT_DIR}/models/ProTrek/weights/ProTrek_35M"

SEQ_THRESHOLD="${SEQ_THRESHOLD:-0.90}"
STRUCTURE_THRESHOLD="${STRUCTURE_THRESHOLD:-0.90}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
SAVE_EMBEDDINGS="${SAVE_EMBEDDINGS:-0}"
VERBOSE="${VERBOSE:-0}"
PROTREK_ENV_NAME="${PROTREK_ENV_NAME:-protrek}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepared-input-root)
      PREPARED_ROOT="$2"
      shift 2
      ;;
    --reactant-root)
      REACTANT_ROOT="$2"
      shift 2
      ;;
    --product-root)
      PRODUCT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --foldseek-bin)
      FOLDSEEK_BIN="$2"
      shift 2
      ;;
    --weights-dir)
      WEIGHTS_DIR="$2"
      shift 2
      ;;
    --seq-threshold)
      SEQ_THRESHOLD="$2"
      shift 2
      ;;
    --structure-threshold)
      STRUCTURE_THRESHOLD="$2"
      shift 2
      ;;
    --test-fraction)
      TEST_FRACTION="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --env-name)
      PROTREK_ENV_NAME="$2"
      shift 2
      ;;
    --save-embeddings)
      SAVE_EMBEDDINGS="1"
      shift
      ;;
    --verbose)
      VERBOSE="1"
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage: bash scripts/run_protrek_split_rf3_pairs.sh [options]

Options:
  --prepared-input-root PATH
  --reactant-root PATH
  --product-root PATH
  --output-dir PATH
  --foldseek-bin PATH
  --weights-dir PATH
  --seq-threshold FLOAT
  --structure-threshold FLOAT
  --test-fraction FLOAT
  --batch-size INT
  --seed INT
  --device {cuda,cpu}
  --limit INT
  --env-name NAME
  --save-embeddings
  --verbose
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run ProTrek split generation" >&2
  exit 1
fi

if ! conda run -n "${PROTREK_ENV_NAME}" python -c "import transformers, torch, sklearn" >/dev/null 2>&1; then
  echo "Conda env '${PROTREK_ENV_NAME}' is missing required ProTrek dependencies." >&2
  echo "Run: conda run -n ${PROTREK_ENV_NAME} pip install -r ${ROOT_DIR}/models/ProTrek/requirements.txt" >&2
  exit 1
fi

CMD=(
  conda run --no-capture-output -n "${PROTREK_ENV_NAME}"
  python "${ROOT_DIR}/scripts/rf3/protrek_cluster_split_rf3_pairs.py"
  --prepared-input-root "${PREPARED_ROOT}"
  --reactant-root "${REACTANT_ROOT}"
  --product-root "${PRODUCT_ROOT}"
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

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${SAVE_EMBEDDINGS}" == "1" ]]; then
  CMD+=(--save-embeddings)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  CMD+=(--verbose)
fi

echo "Running RF3 ProTrek split pipeline:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
