#!/usr/bin/env bash
set -euo pipefail

# Build a UniProt mapping from the reactzyme splits and pull all pocket JSONs
# (ft_act_site, ft_binding, ft_site) for those IDs in one shot.
#
# Usage:
#   ./scripts/pull_all_uniprot_pockets.sh [DATA_ROOT] [OUT_ROOT] [CACHE_DIR]
# Defaults:
#   DATA_ROOT=data/enzyme_ligand/reactzyme_data_split
#   OUT_ROOT=reactzyme_boltz_ts_yaml
#   CACHE_DIR=pocket_cache
#
# Requirements: curl and python; run on a machine with network/DNS access to rest.uniprot.org.

DATA_ROOT=${1:-data/enzyme_ligand/reactzyme_data_split}
OUT_ROOT=${2:-reactzyme_boltz_ts_yaml}
CACHE_DIR=${3:-pocket_cache}
PYTHON_BIN=${PYTHON:-python}

MAP_PATH="${OUT_ROOT}/uniprot_map.json"
ID_PATH="${OUT_ROOT}/pocket_ids.txt"

mkdir -p "${OUT_ROOT}" "${CACHE_DIR}"

echo "[info] building UniProt map from ${DATA_ROOT}"
"${PYTHON_BIN}" - "${DATA_ROOT}" "${OUT_ROOT}" <<'PY'
import json
from pathlib import Path
import torch
import sys

data_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])
map_path = out_root / "uniprot_map.json"
id_path = out_root / "pocket_ids.txt"

def load_pt(path: Path):
    data = torch.load(path, map_location="cpu")
    return data.items()

tsv = data_root / "cleaned_uniprot_rhea.tsv"
seq_to_id = {}
if tsv.exists():
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            entry_idx = header.index("Entry")
            seq_idx = header.index("Sequence")
        except ValueError:
            entry_idx = seq_idx = None
        if entry_idx is not None and seq_idx is not None:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > max(entry_idx, seq_idx):
                    seq_to_id[parts[seq_idx]] = parts[entry_idx]

pt_paths = [
    data_root / "positive_train_val_seq_smi.pt",
    data_root / "positive_test_seq_smi.pt",
]

key_to_uid = {}
for pt in pt_paths:
    if not pt.exists():
        continue
    for k, v in load_pt(pt):
        try:
            smiles, seq = v
        except Exception:
            continue
        uid = seq_to_id.get(seq)
        if uid:
            key_to_uid[str(k)] = uid

with open(map_path, "w") as f:
    json.dump(key_to_uid, f, indent=2)

ids = sorted(set(key_to_uid.values()))
with open(id_path, "w") as f:
    for uid in ids:
        f.write(uid + "\n")

print(f"[info] wrote map with {len(key_to_uid)} entries to {map_path}")
print(f"[info] wrote {len(ids)} unique UniProt IDs to {id_path}")
PY

if [[ ! -s "${ID_PATH}" ]]; then
  echo "[error] no UniProt IDs found; aborting fetch"
  exit 1
fi

TOTAL_IDS=$(wc -l < "${ID_PATH}")
echo "[info] fetching ${TOTAL_IDS} UniProt JSONs into ${CACHE_DIR}"

idx=0
while read -r uid; do
  [[ -z "${uid}" ]] && continue
  idx=$((idx+1))
  outfile="${CACHE_DIR}/${uid}.json"
  if [[ -s "${outfile}" ]]; then
    echo "[${idx}/${TOTAL_IDS}] skip ${uid} (cached)"
    continue
  fi
  url="https://rest.uniprot.org/uniprotkb/${uid}.json?fields=ft_act_site,ft_binding,ft_site"
  echo "[${idx}/${TOTAL_IDS}] fetch ${uid}"
  if ! curl -fsSL --retry 3 --retry-delay 2 "${url}" -o "${outfile}"; then
    echo "[warn] failed to fetch ${uid}" >&2
    rm -f "${outfile}"
  fi
done < "${ID_PATH}"

echo "[info] done. Cached files in ${CACHE_DIR}"
