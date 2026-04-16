"""Shared helpers for real LigandMPNN sequence-generator training and sampling."""

from __future__ import annotations

import gzip
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def ensure_ligandmpnn_imports(repo_root: Path, ligandmpnn_root: str | Path) -> Path:
    """Put the LigandMPNN source directory on ``sys.path`` and return it."""

    root = Path(ligandmpnn_root)
    if not root.is_absolute():
        root = repo_root / root
    root = root.resolve()
    if not (root / "model_utils.py").exists():
        raise FileNotFoundError(f"LigandMPNN root missing model_utils.py: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def prepare_pdb(source: Path, dest: Path) -> None:
    """Materialize PDB input for LigandMPNN from PDB/mmCIF/cif.gz."""

    from prody import parseMMCIF, writePDB

    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".pdb":
        shutil.copy2(source, dest)
        return
    if suffix in {".cif", ".mmcif"}:
        structure = parseMMCIF(str(source))
        atom = structure.select("not water and not hydrogen")
        if atom is None:
            raise RuntimeError(f"failed to parse atoms from {source}")
        for chain in atom.getHierView():
            chain_id = chain.getChid().split(".")[-1]
            chain.setChids(chain_id)
        writePDB(str(dest), atom)
        return
    if suffix == ".gz" and source.name.lower().endswith(".cif.gz"):
        with _open_text(source) as fh, tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as tmp:
            tmp.write(fh.read())
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            prepare_pdb(tmp_path, dest)
        finally:
            tmp_path.unlink(missing_ok=True)
        return
    raise RuntimeError(f"unsupported structure extension for LigandMPNN: {source}")


def load_ligandmpnn_sequence_model(
    *,
    repo_root: Path,
    ligandmpnn_root: str | Path,
    checkpoint_path: str | Path,
    device: str,
    model_type: str = "ligand_mpnn",
    use_side_chain_context: int = 0,
):
    """Load a LigandMPNN sequence model from a standard checkpoint."""

    ensure_ligandmpnn_imports(repo_root, ligandmpnn_root)
    import torch
    from model_utils import ProteinMPNN

    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_path(repo_root, checkpoint_path)
    checkpoint = torch.load(str(ckpt_path), map_location=torch_device)
    atom_context_num = int(checkpoint.get("atom_context_num", 16 if model_type == "ligand_mpnn" else 1))
    k_neighbors = int(checkpoint.get("num_edges", 32))
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=torch_device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=int(use_side_chain_context),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    return model, checkpoint, torch_device, atom_context_num, ckpt_path


def set_trainable_scope(model, train_scope: str) -> list[str]:
    """Freeze/unfreeze LigandMPNN parameters and return trained parameter names."""

    scope = str(train_scope).lower()
    trained: list[str] = []
    for name, param in model.named_parameters():
        if scope == "full":
            train = True
        elif scope == "decoder":
            train = name.startswith("W_s") or name.startswith("decoder_layers") or name.startswith("W_out")
        elif scope == "output":
            train = name.startswith("W_out")
        else:
            raise ValueError(f"Unsupported LigandMPNN train_scope={train_scope!r}; expected decoder, output, or full")
        param.requires_grad = train
        if train:
            trained.append(name)
    if not trained:
        raise RuntimeError(f"No trainable LigandMPNN parameters selected for scope={train_scope}")
    return trained


def encode_sequence_for_structure(protein_dict: dict[str, Any], sequence: str, protein_chain_id: str | None):
    """Replace the design-chain sequence in a parsed LigandMPNN protein dict."""

    import torch
    from data_utils import restype_str_to_int

    seq = "".join(str(sequence).strip().upper().split())
    original = protein_dict["S"].clone().long()
    chain_letters = [str(x) for x in protein_dict["chain_letters"]]
    if protein_chain_id:
        design_idx = [i for i, chain in enumerate(chain_letters) if chain == protein_chain_id]
    else:
        unique_chains = sorted(set(chain_letters))
        if len(unique_chains) != 1:
            raise ValueError(f"sequence provided without protein_chain_id for multi-chain protein: {unique_chains}")
        design_idx = list(range(len(chain_letters)))
    if len(seq) != len(design_idx):
        raise ValueError(
            f"sequence length mismatch for design chain {protein_chain_id or '<all>'}: "
            f"len(sequence)={len(seq)} expected={len(design_idx)}"
        )
    for i, aa in zip(design_idx, seq, strict=False):
        if aa not in restype_str_to_int:
            raise ValueError(f"unsupported amino acid {aa!r} in target sequence")
        original[i] = int(restype_str_to_int[aa])
    protein_dict["S"] = original.to(dtype=torch.long, device=original.device)
    chain_mask = torch.zeros(len(chain_letters), dtype=torch.long, device=original.device)
    chain_mask[design_idx] = 1
    protein_dict["chain_mask"] = chain_mask
    return protein_dict, design_idx


def make_ligandmpnn_feature_dict(
    *,
    repo_root: Path,
    record: dict[str, Any],
    ligandmpnn_root: str | Path,
    work_dir: Path,
    device,
    atom_context_num: int,
    model_type: str,
    structure_key: str = "reactant_complex_path",
    sequence_key: str = "sequence",
    use_atom_context: int = 1,
    parse_atoms_with_zero_occupancy: int = 1,
    ligand_cutoff_a: float = 8.0,
    batch_size: int = 1,
    temperature: float = 0.2,
):
    """Build the exact feature dict consumed by LigandMPNN score/sample."""

    ensure_ligandmpnn_imports(repo_root, ligandmpnn_root)
    import torch
    from data_utils import featurize, parse_PDB

    structure_raw = record.get(structure_key) or record.get("complex_path") or record.get("cif_path")
    if not structure_raw:
        raise ValueError(f"record missing structure path key={structure_key}")
    sequence = record.get(sequence_key)
    if not sequence:
        raise ValueError(f"record missing target sequence key={sequence_key}")
    protein_chain_id = record.get("protein_chain_id") or record.get("chain_id")
    candidate_id = str(record.get("candidate_id") or record.get("backbone_id") or "record")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in candidate_id)[:96]
    prepared_pdb = work_dir / f"{safe_id}.pdb"
    prepare_pdb(resolve_path(repo_root, structure_raw), prepared_pdb)
    protein_dict, _backbone, _other_atoms, _icodes, _water_atoms = parse_PDB(
        str(prepared_pdb),
        device=str(device),
        chains=[],
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=int(parse_atoms_with_zero_occupancy),
    )
    protein_dict, design_idx = encode_sequence_for_structure(protein_dict, str(sequence), protein_chain_id)
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=float(ligand_cutoff_a),
        use_atom_context=bool(use_atom_context),
        number_of_ligand_atoms=int(atom_context_num),
        model_type=model_type,
    )
    _, length = feature_dict["S"].shape
    feature_dict["batch_size"] = int(batch_size)
    feature_dict["temperature"] = float(temperature)
    feature_dict["bias"] = torch.zeros((1, length, 21), dtype=torch.float32, device=device)
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    feature_dict["randn"] = torch.randn((int(batch_size), length), dtype=torch.float32, device=device)
    return feature_dict, protein_dict, design_idx


def compute_record_weight(
    record: dict[str, Any],
    *,
    mode: str = "reward",
    baseline_weight: float = 0.1,
    reward_power: float = 1.0,
    score_temperature: float = 1.0,
    max_weight: float = 20.0,
) -> float:
    """Convert an oracle-labeled row into a positive training weight."""

    source = str(record.get("source", ""))
    reward = record.get("reward")
    score = record.get("score")
    weight = float(baseline_weight)
    try:
        reward_f = float(reward)
        if math.isfinite(reward_f) and reward_f > 0.0:
            if mode == "reward":
                weight = reward_f ** float(reward_power)
            elif mode == "log_reward":
                weight = math.log1p(reward_f) ** float(reward_power)
            elif mode == "score_exp":
                score_f = float(score if score is not None else math.log(max(reward_f, 1e-12)))
                weight = math.exp(max(-8.0, min(8.0, score_f / max(float(score_temperature), 1e-6))))
            else:
                raise ValueError(f"Unsupported reward_weight_mode={mode!r}")
    except Exception:
        weight = float(baseline_weight)
    if source == "baseline" and reward is None:
        weight = float(baseline_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        weight = float(baseline_weight)
    return float(max(1e-6, min(float(max_weight), weight)))


def normalize_weights(weights: list[float]) -> list[float]:
    mean = sum(weights) / max(len(weights), 1)
    if mean <= 0.0:
        return [1.0 for _ in weights]
    return [float(w / mean) for w in weights]


def seed_key(record: dict[str, Any]) -> str:
    return str(record.get("seed_id") or record.get("backbone_id") or record.get("candidate_id") or "")


def build_seed_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick one baseline-like seed record for each seed key."""

    best: dict[str, tuple[tuple[int, int, str], dict[str, Any]]] = {}
    for rec in records:
        seq = str(rec.get("sequence", ""))
        if not seq:
            continue
        key = seed_key(rec)
        source = str(rec.get("source", ""))
        rank = (
            0 if source == "baseline" else 1,
            int(rec.get("K", 0) or 0),
            str(rec.get("candidate_id", "")),
        )
        current = best.get(key)
        if current is None or rank < current[0]:
            best[key] = (rank, rec)
    return {key: item[1] for key, item in best.items()}


def reconstruct_mutation_pairs(seed_sequence: str, target_sequence: str) -> list[tuple[int, str]]:
    seed = "".join(str(seed_sequence).strip().upper().split())
    target = "".join(str(target_sequence).strip().upper().split())
    if len(seed) != len(target):
        raise ValueError(f"equal-length edit reconstruction required: seed={len(seed)} target={len(target)}")
    return [(idx, new) for idx, (old, new) in enumerate(zip(seed, target, strict=False)) if old != new]


def positive_reward(
    record: dict[str, Any],
    *,
    reward_floor: float = 1.0,
    reward_clip_min: float = 1e-6,
    reward_clip_max: float = 1e6,
) -> float:
    try:
        reward = float(record.get("reward"))
        if not math.isfinite(reward) or reward <= 0.0:
            reward = float(reward_floor)
    except Exception:
        reward = float(reward_floor)
    return float(max(float(reward_clip_min), min(float(reward_clip_max), reward)))


def allowed_edit_positions(record: dict[str, Any], seed_sequence: str, mutations: list[tuple[int, str]]) -> list[int]:
    """Use pocket positions when present while always including replay mutations."""

    seq_len = len(seed_sequence)
    allowed: set[int] = set()
    for raw in record.get("pocket_positions") or []:
        try:
            pos = int(raw) - 1
        except Exception:
            continue
        if 0 <= pos < seq_len:
            allowed.add(pos)
    for pos, _aa in mutations:
        if 0 <= int(pos) < seq_len:
            allowed.add(int(pos))
    if not allowed:
        allowed = set(range(seq_len))
    return sorted(allowed)


def set_design_sequence(feature_dict: dict[str, Any], sequence: str, design_idx: list[int]) -> None:
    """Update ``feature_dict['S']`` in-place for the design chain only."""

    from data_utils import restype_str_to_int

    seq = "".join(str(sequence).strip().upper().split())
    if len(seq) != len(design_idx):
        raise ValueError(f"sequence length mismatch: len(sequence)={len(seq)} design_len={len(design_idx)}")
    target = feature_dict["S"].clone().long()
    for local_idx, aa in enumerate(seq):
        target[0, design_idx[local_idx]] = int(restype_str_to_int[aa])
    feature_dict["S"] = target


def make_design_chain_mask(feature_dict: dict[str, Any], design_idx: list[int]):
    import torch

    mask = torch.zeros_like(feature_dict["chain_mask"])
    for idx in design_idx:
        mask[0, idx] = 1
    return mask


def build_gflownet_trajectory_entries(
    records: list[dict[str, Any]],
    *,
    max_mutations: int = 64,
    reward_floor: float = 1.0,
    reward_clip_min: float = 1e-6,
    reward_clip_max: float = 1e6,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    seed_records = build_seed_records(records)
    entries: list[dict[str, Any]] = []
    for rec in records:
        key = seed_key(rec)
        seed_rec = seed_records.get(key)
        if seed_rec is None:
            continue
        seed_seq = "".join(str(seed_rec.get("sequence", "")).strip().upper().split())
        target_seq = "".join(str(rec.get("sequence", "")).strip().upper().split())
        if not seed_seq or not target_seq or len(seed_seq) != len(target_seq):
            continue
        mutations = reconstruct_mutation_pairs(seed_seq, target_seq)
        if len(mutations) > int(max_mutations):
            continue
        entries.append(
            {
                "record": rec,
                "seed_record": seed_rec,
                "seed_key": key,
                "seed_sequence": seed_seq,
                "target_sequence": target_seq,
                "mutations": sorted(mutations, key=lambda item: int(item[0])),
                "allowed_positions": allowed_edit_positions(rec, seed_seq, mutations),
                "reward": positive_reward(
                    rec,
                    reward_floor=reward_floor,
                    reward_clip_min=reward_clip_min,
                    reward_clip_max=reward_clip_max,
                ),
            }
        )
    return entries, seed_records


def design_chain_sequence_from_tensor(protein_dict: dict[str, Any], sequence_tensor, protein_chain_id: str | None) -> str:
    from data_utils import restype_int_to_str

    chain_letters = [str(x) for x in protein_dict["chain_letters"]]
    if protein_chain_id:
        design_idx = [i for i, chain in enumerate(chain_letters) if chain == protein_chain_id]
    else:
        design_idx = list(range(len(chain_letters)))
    seq_np = sequence_tensor.detach().cpu().numpy().tolist()
    return "".join(restype_int_to_str[int(seq_np[i])] for i in design_idx)


def mutation_list(seed_sequence: str, sampled_sequence: str) -> list[str]:
    seed = "".join(str(seed_sequence).strip().upper().split())
    sampled = "".join(str(sampled_sequence).strip().upper().split())
    if len(seed) != len(sampled):
        raise ValueError(f"cannot compute mutations for unequal lengths: seed={len(seed)} sampled={len(sampled)}")
    return [f"{old}{idx + 1}{new}" for idx, (old, new) in enumerate(zip(seed, sampled, strict=False)) if old != new]
