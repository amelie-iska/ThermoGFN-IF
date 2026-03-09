import argparse
import csv
import logging
import time
import shutil
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
import yaml
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

LOGGER = logging.getLogger("reactzyme_yaml")

class FlowList(list):
    """List that should be rendered in flow style (inline) in YAML."""
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)
yaml.add_representer(FlowList, flow_list_representer, Dumper=yaml.SafeDumper)


def setup_logging(level: str, log_file: Optional[Path] = None) -> None:
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(LOGGER.handlers):
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "NA"


def make_pair_id(split: str, row_id: str, rhea_id: str, counter: Dict[str, int]) -> str:
    base = f"{split}__{row_id}"
    if rhea_id:
        base = f"{base}__{sanitize_token(rhea_id)}"
    base = sanitize_token(base)
    idx = counter.get(base, 0)
    counter[base] = idx + 1
    if idx > 0:
        return f"{base}__{idx}"
    return base


def load_pt_dict(path: Path):
    data = torch.load(path, map_location="cpu")
    # Keys are stringified integers in the released splits
    keys = sorted(data.keys(), key=lambda x: int(x))
    return [(k, data[k]) for k in keys]


def load_seq_to_uniprot(data_root: Path):
    """Map full protein sequence to UniProt ID using cleaned_uniprot_rhea.tsv if present."""
    tsv_path = data_root / "cleaned_uniprot_rhea.tsv"
    if not tsv_path.exists():
        return {}
    seq_to_id = {}
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        try:
            entry_idx = header.index("Entry")
            seq_idx = header.index("Sequence")
        except ValueError:
            return {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(entry_idx, seq_idx):
                continue
            seq_to_id[parts[seq_idx]] = parts[entry_idx]
    return seq_to_id


def load_uniprot_to_rhea(tsv_path: Path) -> Dict[str, List[str]]:
    """Map UniProt -> list of Rhea IDs from uniprot_rhea.tsv."""
    mapping: Dict[str, List[str]] = {}
    if not tsv_path.exists():
        return mapping
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        try:
            entry_idx = header.index("Entry")
            rhea_idx = header.index("Rhea ID")
        except ValueError:
            return mapping
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(entry_idx, rhea_idx):
                continue
            entry = parts[entry_idx]
            rhea_ids = [r.strip() for r in parts[rhea_idx].split(";") if r.strip()]
            if entry and rhea_ids:
                mapping[entry] = rhea_ids
    return mapping


def load_rhea_molecules(tsv_path: Path) -> Dict[str, Dict[str, str]]:
    """Map Rhea ID -> dict(substrate=..., product=...)."""
    mapping: Dict[str, Dict[str, str]] = {}
    if not tsv_path.exists():
        return mapping
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        try:
            ridx = header.index("Rhea ID")
            s_idx = header.index("substrate")
            p_idx = header.index("product")
        except ValueError:
            return mapping
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(ridx, s_idx, p_idx):
                continue
            rid = parts[ridx].strip()
            sub = parts[s_idx].strip()
            prod = parts[p_idx].strip()
            if rid and (sub or prod):
                mapping[rid] = {"substrate": sub, "product": prod}
    return mapping


def fetch_uniprot_pockets(uniprot_id: str):
    """Fetch binding site positions from UniProt REST API."""
    LOGGER.debug("Fetching UniProt pockets for %s", uniprot_id)
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=ft_act_site,ft_binding,ft_site"
        LOGGER.debug("Requesting %s", url)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        positions = extract_positions_from_features(data.get("features", []))
        LOGGER.debug(
            "UniProt %s positions: %s%s",
            uniprot_id,
            positions[:10],
            "..." if len(positions) > 10 else "",
        )
        return sorted(set(positions))
    except Exception as e:
        LOGGER.warning("UniProt fetch failed for %s: %s", uniprot_id, e)
        return []


def write_yaml(
    out_path: Path,
    seq: str,
    smiles: str,
    state: str,
    pocket_positions=None,
    ligand_id: str = "L",
    sdf_path: Optional[Path] = None,
    template_chain_id: Optional[str] = None,
    template_threshold: float = 0.1,
):
    """
    state: 'reactant' or 'product'
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contacts = None
    if pocket_positions:
        contacts = FlowList([["A", int(pos)] for pos in pocket_positions])
    payload = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seq, "msa": 0}},
            {"ligand": {"id": ligand_id, "smiles": smiles}},
        ],
        "templates": [],
        "constraints": [],
        "properties": [{"affinity": {"binder": ligand_id}}],
    }
    if sdf_path:
        payload["templates"] = [
            {
                "sdf": str(Path(sdf_path).resolve()),
                "chain_id": template_chain_id or ligand_id,
                "atom_map": "identical",
                "force": True,
                "threshold": float(template_threshold),
            }
        ]
    if contacts:
        payload["constraints"].append(
            {
                "pocket": {
                    "binder": ligand_id,
                    "contacts": contacts,
                    "max_distance": 9.0,
                    "force": True,
                }
            }
        )
    text = yaml.safe_dump(payload, sort_keys=False)
    # Match example formatting: blank line after version
    text = text.replace("version: 1\nsequences:", "version: 1\n\nsequences:")
    with open(out_path, "w") as f:
        f.write(text)


def slice_entries(entries, start, count):
    return entries[start : start + count]


def extract_positions_from_features(features: List[Dict]) -> List[int]:
    positions = []
    for feat in features:
        ftype = feat.get("type", "").lower()
        if ftype not in {"binding site", "binding_site", "binding", "active site"}:
            continue
        loc = feat.get("location", {})
        pos = loc.get("position", {})
        if isinstance(pos, dict) and "value" in pos:
            try:
                positions.append(int(pos["value"]))
                continue
            except Exception:
                pass
        # ranges
        for start_key, end_key in (("begin", "end"), ("start", "end")):
            start_obj = loc.get(start_key, {})
            end_obj = loc.get(end_key, {})
            if not isinstance(start_obj, dict) or not isinstance(end_obj, dict):
                continue
            sv = start_obj.get("value")
            ev = end_obj.get("value")
            if sv is None or ev is None:
                continue
            try:
                sv = int(sv)
                ev = int(ev)
                positions.extend(range(sv, ev + 1))
                break
            except Exception:
                continue
    return positions


def load_pocket_cache(cache_dir: Path) -> Dict[str, List[int]]:
    cache: Dict[str, List[int]] = {}
    if not cache_dir or not cache_dir.exists():
        return cache
    for json_path in cache_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text())
            cache[json_path.stem] = sorted(set(extract_positions_from_features(data.get("features", []))))
        except Exception as e:
            LOGGER.warning("Failed to parse pocket cache %s: %s", json_path, e)
    LOGGER.info("Loaded pocket cache for %s UniProt IDs from %s", len(cache), cache_dir)
    return cache


def parse_pocket_positions(text: str) -> List[int]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    positions = []
    for part in parts:
        try:
            positions.append(int(part))
        except Exception:
            continue
    return positions


def convert_csv_split(
    csv_path: Path,
    split: str,
    out_dir: Path,
    sdf_root: Path,
    pocket_cache: Dict[str, List[int]],
    template_threshold: float,
    require_sdf: bool,
    use_csv_pockets: bool,
    max_rows: Optional[int] = None,
    log_every: int = 500,
):
    if not csv_path.exists():
        LOGGER.warning("Missing CSV: %s", csv_path)
        return
    split_t0 = time.perf_counter()
    counter: Dict[str, int] = {}
    missing_sdf = 0
    missing_pocket = 0
    missing_fields = 0
    written = 0
    total_rows = None
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            total_rows = max(0, sum(1 for _ in handle) - 1)
    except Exception:
        total_rows = None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        total = max_rows if max_rows is not None else total_rows
        iterator = tqdm(reader, desc=f"csv:{split}", total=total) if tqdm is not None else reader
        count = 0
        for row in iterator:
            if max_rows is not None and count >= max_rows:
                break
            count += 1
            row_id = (row.get("row_id") or "").strip() or "row"
            rhea_id = (row.get("rhea_id") or "").strip()
            uniprot_id = (row.get("uniprot_id") or "").strip()
            seq = (row.get("sequence") or "").strip()
            reactant_smiles = (row.get("reactant_smiles") or "").strip()
            product_smiles = (row.get("product_smiles") or "").strip()
            if not seq or not reactant_smiles or not product_smiles:
                missing_fields += 1
                LOGGER.warning("Missing fields for row_id=%s in %s", row_id, csv_path)
                continue

            pair_id = make_pair_id(split, row_id, rhea_id, counter)
            pocket_positions = pocket_cache.get(uniprot_id, [])
            if not pocket_positions and use_csv_pockets:
                pocket_positions = parse_pocket_positions(row.get("pocket_binding_positions") or "")
            if not pocket_positions:
                missing_pocket += 1

            reactant_sdf = sdf_root / split / f"{pair_id}__reactant.sdf"
            product_sdf = sdf_root / split / f"{pair_id}__product.sdf"

            reactant_template = reactant_sdf if reactant_sdf.exists() else None
            product_template = product_sdf if product_sdf.exists() else None

            if require_sdf and (reactant_template is None or product_template is None):
                missing_sdf += 1
                LOGGER.warning("Missing SDF for %s (split=%s)", pair_id, split)
                continue
            if reactant_template is None or product_template is None:
                missing_sdf += 1
                LOGGER.warning("Template missing for %s (split=%s) reactant=%s product=%s",
                               pair_id, split, reactant_template, product_template)

            write_yaml(
                out_dir / f"{pair_id}__reactant.yaml",
                seq,
                reactant_smiles,
                "reactant",
                pocket_positions,
                ligand_id="R",
                sdf_path=reactant_template,
                template_chain_id="R",
                template_threshold=template_threshold,
            )
            write_yaml(
                out_dir / f"{pair_id}__product.yaml",
                seq,
                product_smiles,
                "product",
                pocket_positions,
                ligand_id="P",
                sdf_path=product_template,
                template_chain_id="P",
                template_threshold=template_threshold,
            )
            written += 2
            if tqdm is not None:
                iterator.set_postfix(
                    written=written,
                    missing_sdf=missing_sdf,
                    missing_pocket=missing_pocket,
                    missing_fields=missing_fields,
                )
            if log_every and count % log_every == 0:
                LOGGER.info(
                    "Progress split=%s rows=%s written=%s missing_sdf=%s missing_pocket=%s missing_fields=%s",
                    split,
                    count,
                    written,
                    missing_sdf,
                    missing_pocket,
                    missing_fields,
                )

    if missing_sdf:
        LOGGER.info("%s: missing SDF templates for %s rows", split, missing_sdf)
    if missing_pocket:
        LOGGER.info("%s: missing pocket constraints for %s rows", split, missing_pocket)
    LOGGER.info(
        "Completed split=%s rows=%s written=%s missing_fields=%s missing_sdf=%s missing_pocket=%s",
        split,
        count,
        written,
        missing_fields,
        missing_sdf,
        missing_pocket,
    )
    split_dt = time.perf_counter() - split_t0
    LOGGER.info("Split=%s runtime=%.2fs", split, split_dt)


def convert_split(
    entries,
    out_dir: Path,
    max_items: int,
    reactant_products=None,
    pair_by_index=False,
    structure_root: Path = Path("."),
    uniprot_map=None,
    fetch_pockets=False,
    desc="",
    pocket_cache=None,
    seq_to_uniprot=None,
    uniprot_to_rhea=None,
    rhea_to_mols=None,
):
    iterable = entries[:max_items]
    if tqdm is not None:
        iterable = tqdm(iterable, desc=desc or str(out_dir.name))
    for idx, (key, value) in enumerate(iterable):
        if idx >= max_items:
            break
        try:
            smiles, seq = value
        except Exception:
            LOGGER.warning("Unable to unpack entry %s -> %s", key, value)
            continue

        if reactant_products is not None and key in reactant_products:
            rp = reactant_products[key]
            r_name = rp.get('reactant', f"reactant_state_template_model_{key}")
            p_name = rp.get('product', f"product_state_template_model_{key}")
        elif pair_by_index:
            r_name = f"reactant_state_template_model_{idx}"
            p_name = f"product_state_template_model_{idx}"
        else:
            r_name = f"{key}_initial"
            p_name = f"{key}_final"

        LOGGER.debug("Processing key=%s reactant=%s product=%s", key, r_name, p_name)

        pocket_positions = []
        if pocket_cache and uniprot_map:
            cache_key = uniprot_map.get(str(key)) or uniprot_map.get(key) or uniprot_map.get(seq)
            if cache_key and cache_key in pocket_cache:
                pocket_positions = pocket_cache[cache_key]
                LOGGER.debug("Pocket cache hit for %s: %s", cache_key, len(pocket_positions))
        if fetch_pockets:
            uniprot_id = None
            if uniprot_map:
                uniprot_id = uniprot_map.get(str(key)) or uniprot_map.get(key) or uniprot_map.get(seq)
            if uniprot_id:
                LOGGER.debug("Mapped %s to UniProt %s", key, uniprot_id)
                pocket_positions = fetch_uniprot_pockets(uniprot_id)
                LOGGER.debug("Pocket positions count for %s: %s", key, len(pocket_positions))
            else:
                LOGGER.warning("No UniProt mapping for %s; no pocket constraints", key)

        # Try to assign distinct reactant/product SMILES via UniProt -> Rhea -> rhea_molecules.tsv
        reactant_smiles = smiles
        product_smiles = smiles
        rhea_ids = []
        if seq_to_uniprot:
            uni = seq_to_uniprot.get(seq)
            if uni:
                if isinstance(uni, list):
                    uni_id = uni[0]
                else:
                    uni_id = uni
                if uniprot_to_rhea and uni_id in uniprot_to_rhea:
                    rhea_ids = uniprot_to_rhea[uni_id]
        if not rhea_ids and uniprot_map:
            uni_id = uniprot_map.get(str(key)) or uniprot_map.get(key) or uniprot_map.get(seq)
            if uni_id and uniprot_to_rhea and uni_id in uniprot_to_rhea:
                rhea_ids = uniprot_to_rhea[uni_id]
        chosen_rhea = None
        if rhea_ids and rhea_to_mols:
            for rid in rhea_ids:
                if rid in rhea_to_mols:
                    chosen_rhea = rid
                    break
        if chosen_rhea:
            entry = rhea_to_mols[chosen_rhea]
            reactant_smiles = entry.get("substrate", reactant_smiles) or reactant_smiles
            product_smiles = entry.get("product", product_smiles) or product_smiles
            LOGGER.debug(
                "Rhea match for %s: %s (R/P len %s, %s)",
                key,
                chosen_rhea,
                len(reactant_smiles),
                len(product_smiles),
            )

        LOGGER.debug("Writing YAMLs for %s -> reactant/product", key)
        write_yaml(out_dir / f"{key}_reactant.yaml", seq, reactant_smiles, "reactant", pocket_positions, "R")
        write_yaml(out_dir / f"{key}_product.yaml", seq, product_smiles, "product", pocket_positions, "P")


def main():
    parser = argparse.ArgumentParser(description="Convert Reactzyme .pt splits to Boltz-TS YAMLs.")
    parser.add_argument("--data-root", type=Path, default=Path("data/enzyme_ligand/reactzyme_data_split"))
    parser.add_argument("--output", type=Path, default=Path("input_rp_yamls"))
    parser.add_argument("--csv-dir", type=Path, default=None,
                        help="Directory with reactzyme_{train,validation,test}.csv")
    parser.add_argument("--sdf-root", type=Path, default=Path("output_sdf_templates"),
                        help="Root directory containing ETFlow SDF templates per split")
    parser.add_argument("--splits", type=str, default="train,validation,test",
                        help="Comma-separated splits for CSV mode")
    parser.add_argument("--template-threshold", type=float, default=0.1)
    parser.add_argument("--require-sdf", action="store_true",
                        help="Require SDF templates to exist; skip rows without them")
    parser.add_argument("--use-csv-pockets", action="store_true",
                        help="Fallback to pocket_binding_positions column when pocket_cache is missing")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows per split in CSV mode")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-dir", type=Path, default=Path("input_rp_yamls/logs"))
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Optional single log file for all splits (overrides --log-dir)")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=50)
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--reactant-product-map", type=Path, default=None,
                        help="Optional JSON mapping of key -> {reactant: <name>, product: <name>}")
    parser.add_argument("--pair-by-index", action="store_true",
                        help="Pair reactant/product as reactant_state_template_model_i / product_state_template_model_i")
    parser.add_argument("--structure-root", type=Path, default=Path("data/enzyme_ligand/structures/train"),
                        help="Root dir containing reactant_state_template_model_*.cif and product_state_template_model_*.cif (unused for now)")
    parser.add_argument("--uniprot-map", type=Path, default=None,
                        help="Optional JSON mapping key -> UniProt ID for pocket fetch")
    parser.add_argument("--fetch-pockets", action="store_true",
                        help="Fetch pocket annotations from UniProt REST API using uniprot-map")
    parser.add_argument("--pocket-cache", type=Path, default=Path("pocket_cache"),
                        help="Directory containing cached UniProt JSONs (*.json) to avoid live fetch")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove existing output directory before writing new YAMLs")
    parser.add_argument("--uniprot-rhea-tsv", type=Path, default=None,
                        help="TSV mapping UniProt IDs to Rhea IDs (columns Entry, Rhea ID)")
    parser.add_argument("--rhea-molecules-tsv", type=Path, default=None,
                        help="TSV mapping Rhea ID to substrate/product (columns Rhea ID, substrate, product)")
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)

    pocket_cache = load_pocket_cache(args.pocket_cache) if args.pocket_cache else {}

    if args.clean_output and args.output.exists():
        # Remove previous YAMLs to avoid mixing old and new pockets or splits.
        shutil.rmtree(args.output)

    if args.csv_dir:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        for split in splits:
            if args.log_file is None:
                setup_logging(args.log_level, args.log_dir / f"convert_reactzyme_{split}.log")
            csv_path = args.csv_dir / f"reactzyme_{split}.csv"
            convert_csv_split(
                csv_path=csv_path,
                split=split,
                out_dir=args.output / split,
                sdf_root=args.sdf_root,
                pocket_cache=pocket_cache,
                template_threshold=args.template_threshold,
                require_sdf=args.require_sdf,
                use_csv_pockets=args.use_csv_pockets,
                max_rows=args.max_rows,
                log_every=args.log_every,
            )
        LOGGER.info("Wrote CSV-based YAMLs to %s", args.output)
        return

    train_val_path = args.data_root / "positive_train_val_seq_smi.pt"
    test_path = args.data_root / "positive_test_seq_smi.pt"

    reactant_products = None
    if args.reactant_product_map and args.reactant_product_map.exists():
        reactant_products = json.loads(args.reactant_product_map.read_text())
    uniprot_map = None
    if args.uniprot_map and args.uniprot_map.exists():
        uniprot_map = json.loads(args.uniprot_map.read_text())
    elif args.fetch_pockets:
        uniprot_map = load_seq_to_uniprot(args.data_root)

    train_val = load_pt_dict(train_val_path)
    test_entries = load_pt_dict(test_path)

    # simple deterministic split: first max_train -> train, next max_val -> val
    train_entries = slice_entries(train_val, 0, args.max_train)
    val_entries = slice_entries(train_val, args.max_train, args.max_val)

    seq_to_uniprot = load_seq_to_uniprot(args.data_root)
    uni_to_rhea = load_uniprot_to_rhea(args.uniprot_rhea_tsv) if args.uniprot_rhea_tsv else load_uniprot_to_rhea(args.data_root / "uniprot_rhea.tsv")
    rhea_to_mols = load_rhea_molecules(args.rhea_molecules_tsv) if args.rhea_molecules_tsv else load_rhea_molecules(args.data_root / "rhea_molecules.tsv")

    convert_split(train_entries, args.output / "train", args.max_train, reactant_products,
                  pair_by_index=args.pair_by_index, structure_root=args.structure_root,
                  uniprot_map=uniprot_map, fetch_pockets=args.fetch_pockets, pocket_cache=pocket_cache,
                  seq_to_uniprot=seq_to_uniprot, uniprot_to_rhea=uni_to_rhea, rhea_to_mols=rhea_to_mols)
    convert_split(val_entries, args.output / "val", args.max_val, reactant_products,
                  pair_by_index=args.pair_by_index, structure_root=args.structure_root,
                  uniprot_map=uniprot_map, fetch_pockets=args.fetch_pockets, pocket_cache=pocket_cache,
                  seq_to_uniprot=seq_to_uniprot, uniprot_to_rhea=uni_to_rhea, rhea_to_mols=rhea_to_mols)
    convert_split(test_entries, args.output / "test", args.max_test, reactant_products,
                  pair_by_index=args.pair_by_index, structure_root=args.structure_root,
                  uniprot_map=uniprot_map, fetch_pockets=args.fetch_pockets, pocket_cache=pocket_cache,
                  seq_to_uniprot=seq_to_uniprot, uniprot_to_rhea=uni_to_rhea, rhea_to_mols=rhea_to_mols)

    LOGGER.info("Wrote train/val/test YAMLs to %s", args.output)


if __name__ == "__main__":
    main()
