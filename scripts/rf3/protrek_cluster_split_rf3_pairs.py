#!/usr/bin/env python3
"""Cluster RF3 reactant/product pair outputs with ProTrek and create a split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


LOGGER = logging.getLogger("protrek_rf3_pair_split")


@dataclass
class PairRecord:
    pair_id: str
    sequence: str
    protein_chain_id: str
    ligand_chain_id: str | None
    reactant_model: Path
    product_model: Path
    reactant_structure_sequence: str
    product_structure_sequence: str
    rhea_id: str | None
    uniprot_id: str | None
    substrate_smiles: str | None
    product_smiles: str | None
    pocket_positions: list[int]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def labels(self) -> list[int]:
        roots = [self.find(i) for i in range(len(self.parent))]
        reindex: dict[int, int] = {}
        labels: list[int] = []
        for root in roots:
            if root not in reindex:
                reindex[root] = len(reindex)
            labels.append(reindex[root])
        return labels


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _iter_shard_records(shard_root: Path, state: str):
    state_root = shard_root / "shards" / state
    for shard_path in sorted(state_root.glob("*.json")):
        obj = json.loads(shard_path.read_text())
        if isinstance(obj, dict):
            rows = obj.get("inputs") or obj.get("examples") or [obj]
        else:
            rows = obj
        for rec in rows:
            yield rec


def _collect_inputs(prepared_root: Path) -> dict[str, dict]:
    per_pair: dict[str, dict] = {}
    for state in ("reactant", "product"):
        for rec in _iter_shard_records(prepared_root, state):
            meta = rec.get("metadata", {})
            pair_id = str(meta.get("pair_id") or "").strip()
            if not pair_id:
                continue
            slot = per_pair.setdefault(pair_id, {})
            slot[state] = rec
    return per_pair


def _pair_id_from_job_name(job_name: str, state: str) -> str | None:
    suffix = f"__{state}"
    if not job_name.endswith(suffix):
        return None
    return job_name[: -len(suffix)]


def _collect_model_paths(state_root: Path, state: str) -> dict[str, Path]:
    best: dict[str, Path] = {}
    for cif_path in sorted(state_root.glob("**/*_model.cif")):
        job_name = cif_path.parent.name
        pair_id = _pair_id_from_job_name(job_name, state)
        if not pair_id:
            continue
        incumbent = best.get(pair_id)
        if incumbent is None or len(cif_path.parts) < len(incumbent.parts):
            best[pair_id] = cif_path.resolve()
    return best


def _component_seq(rec: dict) -> str:
    comps = rec.get("components") or []
    if not comps:
        return ""
    return str(comps[0].get("seq") or "").strip()


def _component_smiles(rec: dict) -> str:
    comps = rec.get("components") or []
    if len(comps) < 2:
        return ""
    return str(comps[1].get("smiles") or "").strip()


def import_protrek_components(repo_root: Path):
    protrek_root = repo_root / "models" / "ProTrek"
    if str(protrek_root) not in sys.path:
        sys.path.insert(0, str(protrek_root))
    from model.ProTrek.protein_encoder import ProteinEncoder  # pylint: disable=import-error
    from model.ProTrek.structure_encoder import StructureEncoder  # pylint: disable=import-error
    from utils.foldseek_util import get_struc_seq  # pylint: disable=import-error

    return ProteinEncoder, StructureEncoder, get_struc_seq


def load_checkpoint_components(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw.get("model", raw.get("state_dict", raw))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint structure in {checkpoint_path}")

    protein_state = {}
    structure_state = {}
    for key, tensor in state_dict.items():
        if key.startswith("1."):
            protein_state[key[2:]] = tensor
        elif key.startswith("3."):
            structure_state[key[2:]] = tensor
    if not protein_state:
        raise ValueError("Could not find sequence encoder weights under prefix '1.'")
    if not structure_state:
        raise ValueError("Could not find structure encoder weights under prefix '3.'")
    return protein_state, structure_state


def _extract_structure_sequence(cif_path: Path, chain_id: str, foldseek_bin: Path, get_struc_seq, process_id: int) -> str:
    seq_dict = get_struc_seq(
        str(foldseek_bin),
        str(cif_path),
        chains=[chain_id],
        process_id=process_id,
        foldseek_verbose=False,
    )
    if chain_id not in seq_dict:
        raise RuntimeError(f"Chain '{chain_id}' missing in Foldseek output for {cif_path}")
    struc_seq = str(seq_dict[chain_id][1]).lower().strip()
    if not struc_seq:
        raise RuntimeError(f"Empty structure sequence for {cif_path}")
    return struc_seq


def collect_pair_records(
    prepared_root: Path,
    reactant_root: Path,
    product_root: Path,
    foldseek_bin: Path,
    get_struc_seq,
    *,
    limit: int | None = None,
) -> list[PairRecord]:
    input_map = _collect_inputs(prepared_root)
    reactant_models = _collect_model_paths(reactant_root, "reactant")
    product_models = _collect_model_paths(product_root, "product")
    pair_ids = sorted(set(input_map) & set(reactant_models) & set(product_models))
    if limit is not None:
        pair_ids = pair_ids[:limit]

    records: list[PairRecord] = []
    failures = 0
    iterator = tqdm(pair_ids, desc="Extracting RF3 pair sequences", unit="pair")
    for idx, pair_id in enumerate(iterator):
        reactant_input = input_map[pair_id].get("reactant")
        product_input = input_map[pair_id].get("product")
        if not reactant_input or not product_input:
            failures += 1
            continue
        reactant_meta = reactant_input.get("metadata", {})
        product_meta = product_input.get("metadata", {})
        sequence = _component_seq(reactant_input)
        product_sequence = _component_seq(product_input)
        if not sequence or not product_sequence:
            LOGGER.warning("Skipping %s due to missing protein sequence", pair_id)
            failures += 1
            continue
        if sequence != product_sequence:
            LOGGER.warning("Pair %s has reactant/product sequence mismatch; using reactant sequence", pair_id)
        protein_chain_id = str(reactant_meta.get("protein_chain_id") or product_meta.get("protein_chain_id") or "A")
        ligand_chain_id = reactant_meta.get("ligand_chain_id") or product_meta.get("ligand_chain_id")
        reactant_model = reactant_models[pair_id]
        product_model = product_models[pair_id]
        try:
            reactant_struct_seq = _extract_structure_sequence(reactant_model, protein_chain_id, foldseek_bin, get_struc_seq, idx * 2)
            product_struct_seq = _extract_structure_sequence(product_model, protein_chain_id, foldseek_bin, get_struc_seq, idx * 2 + 1)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s due to structure-sequence extraction failure: %s", pair_id, exc)
            failures += 1
            continue
        records.append(
            PairRecord(
                pair_id=pair_id,
                sequence=sequence,
                protein_chain_id=protein_chain_id,
                ligand_chain_id=str(ligand_chain_id) if ligand_chain_id is not None else None,
                reactant_model=reactant_model,
                product_model=product_model,
                reactant_structure_sequence=reactant_struct_seq,
                product_structure_sequence=product_struct_seq,
                rhea_id=reactant_meta.get("rhea_id") or product_meta.get("rhea_id"),
                uniprot_id=reactant_meta.get("uniprot_id") or product_meta.get("uniprot_id"),
                substrate_smiles=reactant_meta.get("ligand_smiles") or _component_smiles(reactant_input) or None,
                product_smiles=product_meta.get("ligand_smiles") or _component_smiles(product_input) or None,
                pocket_positions=[int(x) for x in (reactant_meta.get("pocket_positions") or product_meta.get("pocket_positions") or [])],
            )
        )
    LOGGER.info(
        "Collected RF3 pairs for ProTrek split: ok=%d skipped=%d prepared_pairs=%d",
        len(records),
        failures,
        len(pair_ids),
    )
    return records


def compute_embeddings(
    records: list[PairRecord],
    protein_config: Path,
    structure_config: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    ProteinEncoder,
    StructureEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs = [r.sequence for r in records]
    reactant_struc = [r.reactant_structure_sequence for r in records]
    product_struc = [r.product_structure_sequence for r in records]

    protein_encoder = ProteinEncoder(str(protein_config), out_dim=1024, load_pretrained=False)
    structure_encoder = StructureEncoder(str(structure_config), out_dim=1024)
    protein_state, structure_state = load_checkpoint_components(checkpoint_path)
    protein_encoder.load_state_dict(protein_state, strict=False)
    structure_encoder.load_state_dict(structure_state, strict=False)
    protein_encoder.eval().to(device)
    structure_encoder.eval().to(device)

    with torch.no_grad():
        seq_emb = protein_encoder.get_repr(seqs, batch_size=batch_size, verbose=True)
        react_emb = structure_encoder.get_repr(reactant_struc, batch_size=batch_size, verbose=True)
        prod_emb = structure_encoder.get_repr(product_struc, batch_size=batch_size, verbose=True)

    return (
        seq_emb.detach().cpu().numpy().astype(np.float32),
        react_emb.detach().cpu().numpy().astype(np.float32),
        prod_emb.detach().cpu().numpy().astype(np.float32),
    )


def build_cluster_labels(
    seq_emb: np.ndarray,
    reactant_str_emb: np.ndarray,
    product_str_emb: np.ndarray,
    seq_threshold: float,
    structure_threshold: float,
) -> tuple[list[int], list[int], list[int]]:
    n = seq_emb.shape[0]
    seq_t = torch.from_numpy(seq_emb)
    react_t = torch.from_numpy(reactant_str_emb)
    prod_t = torch.from_numpy(product_str_emb)
    seq_sim = seq_t @ seq_t.T
    rr = react_t @ react_t.T
    rp = react_t @ prod_t.T
    pr = prod_t @ react_t.T
    pp = prod_t @ prod_t.T
    structure_sim = torch.maximum(torch.maximum(rr, rp), torch.maximum(pr, pp))

    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    seq_mask = seq_sim[tri_i, tri_j] >= seq_threshold
    structure_mask = structure_sim[tri_i, tri_j] >= structure_threshold
    combined_mask = seq_mask | structure_mask
    LOGGER.info(
        "Similarity edges above thresholds: seq=%d structure=%d combined=%d",
        int(seq_mask.sum().item()),
        int(structure_mask.sum().item()),
        int(combined_mask.sum().item()),
    )

    def labels_from_mask(mask: torch.Tensor) -> list[int]:
        uf = UnionFind(n)
        edges_i = tri_i[mask].cpu().numpy().tolist()
        edges_j = tri_j[mask].cpu().numpy().tolist()
        for i, j in zip(edges_i, edges_j):
            uf.union(int(i), int(j))
        return uf.labels()

    return labels_from_mask(seq_mask), labels_from_mask(structure_mask), labels_from_mask(combined_mask)


def split_by_clusters(cluster_labels: list[int], test_fraction: float, seed: int) -> list[str]:
    n = len(cluster_labels)
    cluster_to_indices: dict[int, list[int]] = {}
    for idx, cluster in enumerate(cluster_labels):
        cluster_to_indices.setdefault(cluster, []).append(idx)
    rng = random.Random(seed)
    groups = list(cluster_to_indices.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    target_test = int(round(n * test_fraction))
    test_indices: set[int] = set()
    current = 0
    for group in groups:
        if current >= target_test:
            break
        for idx in group:
            test_indices.add(idx)
        current += len(group)
    return ["test" if i in test_indices else "train" for i in range(n)]


def _cluster_map(records: list[PairRecord], labels: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i, cluster_id in enumerate(labels):
        out.setdefault(str(cluster_id), []).append(records[i].pair_id)
    return out


def _write_spec(path: Path, record: PairRecord, split_label: str, seq_cluster: int, structure_cluster: int, combined_cluster: int) -> None:
    spec = {
        "pair_id": record.pair_id,
        "rhea_id": record.rhea_id,
        "uniprot_id": record.uniprot_id,
        "sequence": record.sequence,
        "sequence_length": len(record.sequence),
        "substrate_smiles": record.substrate_smiles,
        "product_smiles": record.product_smiles,
        "ligand_smiles": record.substrate_smiles,
        "protein_chain_id": record.protein_chain_id,
        "ligand_chain_id": record.ligand_chain_id,
        "pocket_positions": record.pocket_positions,
        "reactant_complex_path": str(record.reactant_model),
        "product_complex_path": str(record.product_model),
        "reactant_protein_path": str(record.reactant_model),
        "product_protein_path": str(record.product_model),
        "representative_structure_path": str(record.reactant_model),
        "protrek": {
            "sequence_cluster": int(seq_cluster),
            "structure_cluster": int(structure_cluster),
            "combined_cluster": int(combined_cluster),
            "structure_similarity_mode": "max_cross_state",
        },
        "specification": {
            "length": str(len(record.sequence)),
            "extra": {
                "task_name": "rf3_reactzyme_protrek_pair_split",
                "example_id": record.pair_id,
                "sampled_contig": str(len(record.sequence)),
                "num_residues": len(record.sequence),
                "sequence_length": len(record.sequence),
                "split": split_label,
                "pair_id": record.pair_id,
                "rhea_id": record.rhea_id,
                "uniprot_id": record.uniprot_id,
                "protein_chain_id": record.protein_chain_id,
                "ligand_chain_id": record.ligand_chain_id,
                "substrate_smiles": record.substrate_smiles,
                "product_smiles": record.product_smiles,
                "pocket_positions": record.pocket_positions,
                "reactant_complex_path": str(record.reactant_model),
                "product_complex_path": str(record.product_model),
                "reactant_protein_path": str(record.reactant_model),
                "product_protein_path": str(record.product_model),
                "representative_structure_path": str(record.reactant_model),
            },
        },
    }
    path.write_text(json.dumps(spec, indent=2))


def write_split_output(
    output_dir: Path,
    records: list[PairRecord],
    seq_clusters: list[int],
    structure_clusters: list[int],
    combined_clusters: list[int],
    split_labels: list[str],
    *,
    seq_threshold: float,
    structure_threshold: float,
    test_fraction: float,
    seed: int,
    prepared_root: Path,
    reactant_root: Path,
    product_root: Path,
    save_embeddings: bool,
    seq_emb: np.ndarray | None,
    reactant_str_emb: np.ndarray | None,
    product_str_emb: np.ndarray | None,
) -> None:
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    metadata_dir = output_dir / "metadata"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / "pair_index.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "pair_id",
                "split",
                "sequence_length",
                "protein_chain_id",
                "ligand_chain_id",
                "rhea_id",
                "uniprot_id",
                "reactant_model",
                "product_model",
                "seq_cluster",
                "structure_cluster",
                "combined_cluster",
            ]
        )
        for idx, record in enumerate(records):
            split_label = split_labels[idx]
            writer.writerow(
                [
                    idx,
                    record.pair_id,
                    split_label,
                    len(record.sequence),
                    record.protein_chain_id,
                    record.ligand_chain_id or "",
                    record.rhea_id or "",
                    record.uniprot_id or "",
                    str(record.reactant_model),
                    str(record.product_model),
                    seq_clusters[idx],
                    structure_clusters[idx],
                    combined_clusters[idx],
                ]
            )
            target_dir = train_dir if split_label == "train" else test_dir
            _write_spec(
                target_dir / f"{record.pair_id}.json",
                record,
                split_label,
                seq_clusters[idx],
                structure_clusters[idx],
                combined_clusters[idx],
            )

    summary = {
        "n_pairs": len(records),
        "n_train": split_labels.count("train"),
        "n_test": split_labels.count("test"),
        "test_fraction_requested": test_fraction,
        "test_fraction_actual": split_labels.count("test") / max(1, len(split_labels)),
        "seq_threshold": seq_threshold,
        "structure_threshold": structure_threshold,
        "structure_similarity_mode": "max_cross_state",
        "seed": seed,
        "n_seq_clusters": len(set(seq_clusters)),
        "n_structure_clusters": len(set(structure_clusters)),
        "n_combined_clusters": len(set(combined_clusters)),
        "prepared_input_root": str(prepared_root),
        "reactant_root": str(reactant_root),
        "product_root": str(product_root),
    }
    (metadata_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (metadata_dir / "seq_clusters.json").write_text(json.dumps(_cluster_map(records, seq_clusters), indent=2))
    (metadata_dir / "structure_clusters.json").write_text(json.dumps(_cluster_map(records, structure_clusters), indent=2))
    (metadata_dir / "combined_clusters.json").write_text(json.dumps(_cluster_map(records, combined_clusters), indent=2))

    if save_embeddings and seq_emb is not None and reactant_str_emb is not None and product_str_emb is not None:
        np.save(metadata_dir / "seq_embeddings.npy", seq_emb)
        np.save(metadata_dir / "reactant_structure_embeddings.npy", reactant_str_emb)
        np.save(metadata_dir / "product_structure_embeddings.npy", product_str_emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProTrek RF3 train/test split for reactant/product pair outputs.")
    parser.add_argument("--prepared-input-root", required=True, type=Path)
    parser.add_argument("--reactant-root", required=True, type=Path)
    parser.add_argument("--product-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--foldseek-bin", type=Path, default=Path("models/ProTrek/bin/foldseek"))
    parser.add_argument("--weights-dir", type=Path, default=Path("models/ProTrek/weights/ProTrek_35M"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seq-threshold", type=float, default=0.90)
    parser.add_argument("--structure-threshold", type=float, default=0.90)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    repo_root = _repo_root()
    prepared_root = (repo_root / args.prepared_input_root).resolve() if not args.prepared_input_root.is_absolute() else args.prepared_input_root.resolve()
    reactant_root = (repo_root / args.reactant_root).resolve() if not args.reactant_root.is_absolute() else args.reactant_root.resolve()
    product_root = (repo_root / args.product_root).resolve() if not args.product_root.is_absolute() else args.product_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    foldseek_bin = (repo_root / args.foldseek_bin).resolve() if not args.foldseek_bin.is_absolute() else args.foldseek_bin.resolve()
    weights_dir = (repo_root / args.weights_dir).resolve() if not args.weights_dir.is_absolute() else args.weights_dir.resolve()
    checkpoint = (
        (weights_dir / "ProTrek_35M.pt")
        if args.checkpoint is None
        else ((repo_root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve())
    )
    protein_config = weights_dir / "esm2_t12_35M_UR50D"
    structure_config = weights_dir / "foldseek_t12_35M"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ProTrek RF3 splitting.")
    if not foldseek_bin.exists():
        raise FileNotFoundError(f"Foldseek binary not found: {foldseek_bin}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"ProTrek checkpoint not found: {checkpoint}")
    if not protein_config.exists() or not structure_config.exists():
        raise FileNotFoundError("Missing ProTrek model directories under weights.")

    ProteinEncoder, StructureEncoder, get_struc_seq = import_protrek_components(repo_root)
    records = collect_pair_records(
        prepared_root=prepared_root,
        reactant_root=reactant_root,
        product_root=product_root,
        foldseek_bin=foldseek_bin,
        get_struc_seq=get_struc_seq,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No RF3 pair records could be extracted for ProTrek splitting.")
    seq_emb, reactant_str_emb, product_str_emb = compute_embeddings(
        records=records,
        protein_config=protein_config,
        structure_config=structure_config,
        checkpoint_path=checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        ProteinEncoder=ProteinEncoder,
        StructureEncoder=StructureEncoder,
    )
    seq_clusters, structure_clusters, combined_clusters = build_cluster_labels(
        seq_emb=seq_emb,
        reactant_str_emb=reactant_str_emb,
        product_str_emb=product_str_emb,
        seq_threshold=args.seq_threshold,
        structure_threshold=args.structure_threshold,
    )
    split_labels = split_by_clusters(combined_clusters, test_fraction=args.test_fraction, seed=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_output(
        output_dir=output_dir,
        records=records,
        seq_clusters=seq_clusters,
        structure_clusters=structure_clusters,
        combined_clusters=combined_clusters,
        split_labels=split_labels,
        seq_threshold=args.seq_threshold,
        structure_threshold=args.structure_threshold,
        test_fraction=args.test_fraction,
        seed=args.seed,
        prepared_root=prepared_root,
        reactant_root=reactant_root,
        product_root=product_root,
        save_embeddings=args.save_embeddings,
        seq_emb=seq_emb,
        reactant_str_emb=reactant_str_emb,
        product_str_emb=product_str_emb,
    )
    LOGGER.info(
        "Done. RF3 pair split written to %s (train=%d test=%d)",
        output_dir,
        split_labels.count("train"),
        split_labels.count("test"),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
