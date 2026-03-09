#!/usr/bin/env python3
"""Cluster RFdiffusion outputs with ProTrek embeddings and create train/test copies.

This script:
1) extracts sequence and foldseek structural sequence from CIF/CIF.GZ files,
2) computes ProTrek sequence and structure embeddings,
3) clusters by cosine similarity (sequence, structure, and combined),
4) performs a cluster-aware train/test split, and
5) copies files into split directories without modifying originals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


LOGGER = logging.getLogger("protrek_cluster_split")


@dataclass
class DesignRecord:
    stem: str
    structure_path: Path
    selected_chain: str
    aa_sequence: str
    structure_sequence: str


def canonical_stem(file_path: Path) -> str:
    name = file_path.name
    if name.endswith(".cif.gz"):
        return name[: -len(".cif.gz")]
    return file_path.stem


def discover_file_groups(input_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        stem = canonical_stem(path)
        groups.setdefault(stem, []).append(path)
    return groups


def pick_structure_path(paths: Iterable[Path]) -> Path | None:
    cif = None
    cif_gz = None
    for p in paths:
        if p.name.endswith(".cif"):
            cif = p
        elif p.name.endswith(".cif.gz"):
            cif_gz = p
    return cif or cif_gz


def resolve_design_inputs(input_dir: Path, limit: int | None = None) -> Tuple[List[str], Dict[str, List[Path]], Dict[str, Path]]:
    groups = discover_file_groups(input_dir)
    stems = sorted(groups.keys())
    if limit is not None:
        stems = stems[:limit]

    structure_by_stem: Dict[str, Path] = {}
    filtered_stems: List[str] = []
    for stem in stems:
        structure_path = pick_structure_path(groups[stem])
        if structure_path is None:
            continue
        filtered_stems.append(stem)
        structure_by_stem[stem] = structure_path

    return filtered_stems, groups, structure_by_stem


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


def extract_sequences(
    stems: List[str],
    structure_by_stem: Dict[str, Path],
    foldseek_bin: Path,
    get_struc_seq,
) -> List[DesignRecord]:
    records: List[DesignRecord] = []
    with tempfile.TemporaryDirectory(prefix="protrek_extract_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        iterator = tqdm(stems, desc="Extracting AA + structure sequences", unit="design")
        for idx, stem in enumerate(iterator):
            structure_path = structure_by_stem[stem]
            active_path = structure_path
            tmp_cif: Path | None = None

            try:
                if structure_path.name.endswith(".cif.gz"):
                    tmp_cif = tmpdir_path / f"{stem}.cif"
                    with gzip.open(structure_path, "rb") as src, open(tmp_cif, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    active_path = tmp_cif

                seq_dict = get_struc_seq(
                    str(foldseek_bin),
                    str(active_path),
                    chains=None,
                    process_id=idx,
                    foldseek_verbose=False,
                )
                if not seq_dict:
                    LOGGER.warning("No chains found for %s", structure_path)
                    continue

                chain, values = max(seq_dict.items(), key=lambda item: len(item[1][0]))
                aa_seq, struc_seq = values[0], values[1].lower()
                if not aa_seq or not struc_seq:
                    LOGGER.warning("Empty extracted sequence for %s", structure_path)
                    continue

                records.append(
                    DesignRecord(
                        stem=stem,
                        structure_path=structure_path,
                        selected_chain=chain,
                        aa_sequence=aa_seq,
                        structure_sequence=struc_seq,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed extraction for %s: %s", structure_path, exc)
            finally:
                if tmp_cif and tmp_cif.exists():
                    tmp_cif.unlink(missing_ok=True)
    return records


def compute_embeddings(
    records: List[DesignRecord],
    protein_config: Path,
    structure_config: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    ProteinEncoder,
    StructureEncoder,
) -> Tuple[np.ndarray, np.ndarray]:
    seqs = [r.aa_sequence for r in records]
    struc_seqs = [r.structure_sequence for r in records]

    LOGGER.info("Loading ProTrek encoders on device=%s", device)
    protein_encoder = ProteinEncoder(str(protein_config), out_dim=1024, load_pretrained=False)
    structure_encoder = StructureEncoder(str(structure_config), out_dim=1024)

    protein_state, structure_state = load_checkpoint_components(checkpoint_path)
    protein_missing, protein_unexpected = protein_encoder.load_state_dict(protein_state, strict=False)
    structure_missing, structure_unexpected = structure_encoder.load_state_dict(structure_state, strict=False)
    if protein_missing or protein_unexpected:
        LOGGER.warning("Sequence encoder state mismatch: missing=%s unexpected=%s", protein_missing, protein_unexpected)
    if structure_missing or structure_unexpected:
        LOGGER.warning("Structure encoder state mismatch: missing=%s unexpected=%s", structure_missing, structure_unexpected)

    protein_encoder.eval().to(device)
    structure_encoder.eval().to(device)

    with torch.no_grad():
        seq_emb = protein_encoder.get_repr(seqs, batch_size=batch_size, verbose=True)
        str_emb = structure_encoder.get_repr(struc_seqs, batch_size=batch_size, verbose=True)

    seq_np = seq_emb.detach().cpu().numpy().astype(np.float32)
    str_np = str_emb.detach().cpu().numpy().astype(np.float32)
    return seq_np, str_np


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
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

    def labels(self) -> List[int]:
        roots = [self.find(i) for i in range(len(self.parent))]
        reindex: Dict[int, int] = {}
        labels: List[int] = []
        for root in roots:
            if root not in reindex:
                reindex[root] = len(reindex)
            labels.append(reindex[root])
        return labels


def build_cluster_labels(
    seq_emb: np.ndarray,
    str_emb: np.ndarray,
    seq_threshold: float,
    str_threshold: float,
) -> Tuple[List[int], List[int], List[int]]:
    n = seq_emb.shape[0]
    seq_t = torch.from_numpy(seq_emb)
    str_t = torch.from_numpy(str_emb)
    seq_sim = seq_t @ seq_t.T
    str_sim = str_t @ str_t.T

    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    seq_mask = seq_sim[tri_i, tri_j] >= seq_threshold
    str_mask = str_sim[tri_i, tri_j] >= str_threshold
    combined_mask = seq_mask | str_mask

    LOGGER.info(
        "Similarity edges above thresholds: seq=%d, structure=%d, combined=%d",
        int(seq_mask.sum().item()),
        int(str_mask.sum().item()),
        int(combined_mask.sum().item()),
    )

    def labels_from_mask(mask: torch.Tensor) -> List[int]:
        uf = UnionFind(n)
        edges_i = tri_i[mask].cpu().numpy().tolist()
        edges_j = tri_j[mask].cpu().numpy().tolist()
        for i, j in zip(edges_i, edges_j):
            uf.union(int(i), int(j))
        return uf.labels()

    return labels_from_mask(seq_mask), labels_from_mask(str_mask), labels_from_mask(combined_mask)


def split_by_clusters(cluster_labels: List[int], test_fraction: float, seed: int) -> List[str]:
    n = len(cluster_labels)
    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, cluster in enumerate(cluster_labels):
        cluster_to_indices.setdefault(cluster, []).append(idx)

    rng = random.Random(seed)
    groups = list(cluster_to_indices.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_test = int(round(n * test_fraction))
    test_indices = set()
    current = 0
    for group in groups:
        if current >= target_test:
            break
        for idx in group:
            test_indices.add(idx)
        current += len(group)

    split = ["test" if i in test_indices else "train" for i in range(n)]
    return split


def write_metadata(
    output_dir: Path,
    records: List[DesignRecord],
    seq_clusters: List[int],
    str_clusters: List[int],
    combined_clusters: List[int],
    split_labels: List[str],
    seq_threshold: float,
    str_threshold: float,
    test_fraction: float,
    seed: int,
):
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / "design_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "stem",
                "source_structure_path",
                "selected_chain",
                "aa_length",
                "structure_length",
                "seq_cluster",
                "structure_cluster",
                "combined_cluster",
                "split",
            ]
        )
        for idx, record in enumerate(records):
            writer.writerow(
                [
                    idx,
                    record.stem,
                    str(record.structure_path),
                    record.selected_chain,
                    len(record.aa_sequence),
                    len(record.structure_sequence),
                    seq_clusters[idx],
                    str_clusters[idx],
                    combined_clusters[idx],
                    split_labels[idx],
                ]
            )

    def cluster_map(labels: List[int]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for i, cluster_id in enumerate(labels):
            out.setdefault(str(cluster_id), []).append(records[i].stem)
        return out

    summary = {
        "n_designs": len(records),
        "n_train": split_labels.count("train"),
        "n_test": split_labels.count("test"),
        "test_fraction_requested": test_fraction,
        "test_fraction_actual": split_labels.count("test") / max(1, len(split_labels)),
        "seq_threshold": seq_threshold,
        "structure_threshold": str_threshold,
        "seed": seed,
        "n_seq_clusters": len(set(seq_clusters)),
        "n_structure_clusters": len(set(str_clusters)),
        "n_combined_clusters": len(set(combined_clusters)),
    }

    with open(metadata_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(metadata_dir / "seq_clusters.json", "w", encoding="utf-8") as handle:
        json.dump(cluster_map(seq_clusters), handle, indent=2)
    with open(metadata_dir / "structure_clusters.json", "w", encoding="utf-8") as handle:
        json.dump(cluster_map(str_clusters), handle, indent=2)
    with open(metadata_dir / "combined_clusters.json", "w", encoding="utf-8") as handle:
        json.dump(cluster_map(combined_clusters), handle, indent=2)


def copy_split_files(
    output_dir: Path,
    groups: Dict[str, List[Path]],
    stems: List[str],
    split_labels: List[str],
):
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        dest = train_dir if split_name == "train" else test_dir
        indices = [i for i, label in enumerate(split_labels) if label == split_name]
        for idx in tqdm(indices, desc=f"Copying {split_name} files", unit="design"):
            stem = stems[idx]
            files = groups.get(stem, [])
            for src in files:
                shutil.copy2(src, dest / src.name)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProTrek clustering split for RFdiffusion outputs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing RFdiffusion output files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for split copies and metadata.")
    parser.add_argument(
        "--foldseek-bin",
        type=Path,
        default=Path("models/ProTrek/bin/foldseek"),
        help="Path to foldseek binary.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models/ProTrek/weights/ProTrek_35M"),
        help="ProTrek_35M weights directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (defaults to <weights-dir>/ProTrek_35M.pt).",
    )
    parser.add_argument("--seq-threshold", type=float, default=0.90, help="Cosine threshold for sequence clustering.")
    parser.add_argument("--structure-threshold", type=float, default=0.90, help="Cosine threshold for structure clustering.")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Requested test set fraction.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda|cuda:0|auto (GPU-only).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of designs (debug).")
    parser.add_argument("--save-embeddings", action="store_true", help="Save raw embedding arrays to metadata.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir.resolve()
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

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not foldseek_bin.exists():
        raise FileNotFoundError(f"Foldseek binary not found: {foldseek_bin}")
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not protein_config.exists() or not structure_config.exists():
        raise FileNotFoundError("Missing protein/structure config directories under weights.")

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)

    stems, groups, structure_by_stem = resolve_design_inputs(input_dir=input_dir, limit=args.limit)
    if not stems:
        raise RuntimeError(f"No CIF/CIF.GZ designs found in {input_dir}")
    LOGGER.info("Discovered %d designs with structure files", len(stems))

    ProteinEncoder, StructureEncoder, get_struc_seq = import_protrek_components(repo_root)

    records = extract_sequences(stems, structure_by_stem, foldseek_bin, get_struc_seq)
    if not records:
        raise RuntimeError("No designs could be parsed into sequence/structure representations.")
    LOGGER.info("Successfully extracted sequences for %d/%d designs", len(records), len(stems))

    # Keep ordering consistent with records from this point forward.
    stems = [r.stem for r in records]
    device = args.device
    if device == "auto":
        device = "cuda"
    if not device.startswith("cuda"):
        raise RuntimeError(f"GPU-only pipeline: refusing non-CUDA device '{device}'.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the active environment. "
            "Activate 'protrek' directly (e.g. `conda activate protrek`) and avoid `conda run`."
        )

    seq_emb, str_emb = compute_embeddings(
        records=records,
        protein_config=protein_config,
        structure_config=structure_config,
        checkpoint_path=checkpoint,
        batch_size=args.batch_size,
        device=device,
        ProteinEncoder=ProteinEncoder,
        StructureEncoder=StructureEncoder,
    )
    LOGGER.info("Embeddings computed: sequence=%s structure=%s", seq_emb.shape, str_emb.shape)

    seq_clusters, str_clusters, combined_clusters = build_cluster_labels(
        seq_emb=seq_emb,
        str_emb=str_emb,
        seq_threshold=args.seq_threshold,
        str_threshold=args.structure_threshold,
    )
    split_labels = split_by_clusters(
        cluster_labels=combined_clusters,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    write_metadata(
        output_dir=output_dir,
        records=records,
        seq_clusters=seq_clusters,
        str_clusters=str_clusters,
        combined_clusters=combined_clusters,
        split_labels=split_labels,
        seq_threshold=args.seq_threshold,
        str_threshold=args.structure_threshold,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    if args.save_embeddings:
        np.save(output_dir / "metadata" / "seq_embeddings.npy", seq_emb)
        np.save(output_dir / "metadata" / "structure_embeddings.npy", str_emb)

    copy_split_files(
        output_dir=output_dir,
        groups=groups,
        stems=stems,
        split_labels=split_labels,
    )

    LOGGER.info(
        "Done. train=%d test=%d (design-level); files copied under %s",
        split_labels.count("train"),
        split_labels.count("test"),
        output_dir,
    )


if __name__ == "__main__":
    main()
