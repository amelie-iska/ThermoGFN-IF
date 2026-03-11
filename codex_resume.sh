#!/usr/bin/env bash
set -euo pipefail

# Always anchor to the directory containing this script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec codex resume --last \
  --cd "$ROOT_DIR" \
  --dangerously-bypass-approvals-and-sandbox \
  --search \
  "$@"
