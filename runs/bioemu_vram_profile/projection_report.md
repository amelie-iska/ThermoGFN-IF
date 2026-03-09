# BioEmu VRAM Profile

- input: `/home/iska/amelie-ai/ThermoGFN/runs/thermogfn_ligandmpnn/bootstrap/D_0_train.jsonl`
- batch_size_100: `10`
- num_samples: `128`
- target_vram_frac: `0.9`
- filter_samples: `False`

## Measured

| length | effective_batch | peak_alloc_GiB | peak_reserved_GiB | peak_alloc_frac | peak_reserved_frac | candidate_id |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 10 | 0.799 | 1.018 | 0.034 | 0.043 | f433829b37250195 |
| 180 | 3 | 0.761 | 0.996 | 0.032 | 0.042 | 4263727932931c83 |
| 260 | 1 | 0.568 | 0.818 | 0.024 | 0.035 | 26ac8566d5b4fb21 |
| 340 | 1 | 0.869 | 1.271 | 0.037 | 0.054 | 769cab94e5312eec |
| 400 | 1 | 1.150 | 1.719 | 0.049 | 0.073 | 887ea2e2021ce1d7 |

## Projection

- projected max length at effective_batch=1 (reserved-memory model): **1322 residues**
- projected max batch_size_100 at length 100: **174**
- total_vram_GiB: **23.518**

_Model:_ conservative upper-envelope on `peak_reserved_bytes / (L^2 * effective_batch)` from measured rows.
