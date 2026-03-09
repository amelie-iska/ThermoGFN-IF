#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Decompress RFdiffusion3 output CIF archives (*.cif.gz).

Usage:
  scripts/decompress_rfd3_cifs.sh [OPTIONS] [DIR ...]

Defaults:
  If no DIR is provided, uses:
    ./data/rfd3/unconditional_monomer

Options:
  --keep-source            Keep .cif.gz files after decompression (default)
  --delete-source          Delete .cif.gz files after successful decompression
  --overwrite              Overwrite existing .cif files
  --dry-run                Print what would be done without writing files
  -h, --help               Show this help

Examples:
  scripts/decompress_rfd3_cifs.sh
  scripts/decompress_rfd3_cifs.sh ./data/rfd3/unconditional_dimer_asymmetric
  scripts/decompress_rfd3_cifs.sh --delete-source ./data/rfd3
EOF
}

log() {
  printf '%s\n' "$*" >&2
}

KEEP_SOURCE=true
OVERWRITE=false
DRY_RUN=false
declare -a TARGET_DIRS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-source)
      KEEP_SOURCE=true
      shift
      ;;
    --delete-source)
      KEEP_SOURCE=false
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      log "Error: unknown option '$1'"
      usage
      exit 1
      ;;
    *)
      TARGET_DIRS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  TARGET_DIRS+=(./data/rfd3/unconditional_monomer)
fi

declare -a FILES=()
for d in "${TARGET_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    log "Warning: directory not found, skipping: $d"
    continue
  fi
  while IFS= read -r -d '' f; do
    FILES+=("$f")
  done < <(find "$d" -type f -name '*.cif.gz' -print0)
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  log "No .cif.gz files found."
  exit 0
fi

log "Found ${#FILES[@]} .cif.gz file(s)."
log "keep_source=$KEEP_SOURCE overwrite=$OVERWRITE dry_run=$DRY_RUN"

processed=0
skipped=0
failed=0

for src in "${FILES[@]}"; do
  dst="${src%.gz}"

  if [[ -e "$dst" && "$OVERWRITE" == false ]]; then
    log "Skip (exists): $dst"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$DRY_RUN" == true ]]; then
    log "Would decompress: $src -> $dst"
    processed=$((processed + 1))
    continue
  fi

  tmp="${dst}.tmp.$$"
  if gzip -dc -- "$src" > "$tmp"; then
    mv -f -- "$tmp" "$dst"
    if [[ "$KEEP_SOURCE" == false ]]; then
      rm -f -- "$src"
    fi
    processed=$((processed + 1))
  else
    rm -f -- "$tmp" || true
    log "Failed: $src"
    failed=$((failed + 1))
  fi
done

log "Done. processed=$processed skipped=$skipped failed=$failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

