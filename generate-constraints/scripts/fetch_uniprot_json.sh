#!/usr/bin/env bash
set -euo pipefail

# Fetch UniProt site-feature JSONs for all IDs in a uniprot_map.json and
# store them locally. Run on a machine with network access to rest.uniprot.org.
#
# Usage:
#   ./scripts/fetch_uniprot_json.sh reactzyme_boltz_ts_yaml/uniprot_map.json pocket_cache
#
# The map is a JSON object mapping sample keys (or sequences) to UniProt IDs.
# The script extracts unique IDs and downloads:
#   https://rest.uniprot.org/uniprotkb/<ID>.json?fields=ft_act_site,ft_binding,ft_site

MAP_PATH=${1:-reactzyme_boltz_ts_yaml/uniprot_map.json}
OUT_DIR=${2:-pocket_cache}

if [[ ! -f "${MAP_PATH}" ]]; then
  echo "Map file not found: ${MAP_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Extracting UniProt IDs from ${MAP_PATH}..."
IDS=$(python - <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
ids=set(data.values())
for uid in sorted(ids):
    if uid:
        print(uid)
PY
"${MAP_PATH}")

if [[ -z "${IDS}" ]]; then
  echo "No UniProt IDs found in map." >&2
  exit 1
fi

echo "Will fetch $(echo "${IDS}" | wc -l) UniProt entries into ${OUT_DIR}"

for uid in ${IDS}; do
  outfile="${OUT_DIR}/${uid}.json"
  if [[ -s "${outfile}" ]]; then
    echo "skip ${uid} (cached)"
    continue
  fi
  url="https://rest.uniprot.org/uniprotkb/${uid}.json?fields=ft_act_site,ft_binding,ft_site"
  echo "fetch ${uid}"
  curl -fsSL --retry 3 --retry-delay 2 "${url}" -o "${outfile}" || {
    echo "warn: failed to fetch ${uid}" >&2
    rm -f "${outfile}"
  }
done

echo "Done. Cached files in ${OUT_DIR}"
