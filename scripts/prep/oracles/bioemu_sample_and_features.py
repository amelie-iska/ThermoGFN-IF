#!/usr/bin/env python3
"""BioEmu scoring wrapper using sampled ensembles only (no fallback proxies).

This scorer supports packed multi-protein batching per denoiser step to improve
GPU utilization. It dynamically adjusts `batch_size_100` to track a VRAM target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _load_reference_ca_coords(path: Path, preferred_chain: str | None) -> np.ndarray:
    from Bio.PDB import MMCIFParser, PDBParser

    suffix = path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif", ".gz"} else PDBParser(QUIET=True)

    if suffix == ".gz":
        with _open_text(path) as fh, tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=True) as tmp:
            tmp.write(fh.read())
            tmp.flush()
            structure = parser.get_structure(path.stem, tmp.name)
    else:
        structure = parser.get_structure(path.stem, str(path))

    model = next(structure.get_models())
    ca_by_chain: dict[str, np.ndarray] = {}
    for chain in model:
        coords = []
        for residue in chain:
            if residue.has_id("CA"):
                coords.append(residue["CA"].get_coord())
        if coords:
            ca_by_chain[str(chain.id)] = np.asarray(coords, dtype=np.float32)

    if not ca_by_chain:
        raise ValueError(f"no CA atoms found in reference structure: {path}")

    if preferred_chain:
        cid = str(preferred_chain)
        if cid in ca_by_chain:
            return ca_by_chain[cid]

    # Monomer default: use the longest chain.
    return max(ca_by_chain.values(), key=lambda arr: arr.shape[0])


def _build_native_contacts(ref_ca_nm: np.ndarray, min_seq_sep: int = 3, cutoff_nm: float = 0.8):
    n = int(ref_ca_nm.shape[0])
    pairs_i = []
    pairs_j = []
    d0 = []
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            dij = float(np.linalg.norm(ref_ca_nm[i] - ref_ca_nm[j]))
            if dij <= cutoff_nm:
                pairs_i.append(i)
                pairs_j.append(j)
                d0.append(dij)
    if not d0:
        raise ValueError(f"no native contacts built for n={n} cutoff_nm={cutoff_nm}")
    return np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64), np.asarray(d0, dtype=np.float32)


def _compute_features_from_samples(xtc_path: Path, top_pdb_path: Path, ref_ca_ang: np.ndarray) -> tuple[dict, float, float]:
    import mdtraj as md

    traj = md.load_xtc(str(xtc_path), top=str(top_pdb_path))
    ca_idx = traj.topology.select("name CA")
    if ca_idx.size == 0:
        raise ValueError("no CA atoms found in sampled topology")
    traj_ca = traj.atom_slice(ca_idx)

    n_frames = int(traj_ca.n_frames)
    n_ca = int(traj_ca.n_atoms)
    if ref_ca_ang.shape[0] != n_ca:
        raise ValueError(f"CA length mismatch ref={ref_ca_ang.shape[0]} sampled={n_ca}")

    ref_ca_nm = ref_ca_ang.astype(np.float32) / 10.0
    ref_traj = md.Trajectory(xyz=ref_ca_nm[None, :, :], topology=traj_ca.topology)
    rmsd_nm = md.rmsd(traj_ca, ref_traj)

    pair_i, pair_j, d0 = _build_native_contacts(ref_ca_nm, min_seq_sep=3, cutoff_nm=0.8)
    dij = np.linalg.norm(traj_ca.xyz[:, pair_i, :] - traj_ca.xyz[:, pair_j, :], axis=2)
    q_nat = (dij <= (1.2 * d0[None, :])).mean(axis=1).astype(np.float32)

    # Folded/disrupted thresholds aligned with manuscript defaults.
    folded = (q_nat >= 0.70) & (rmsd_nm <= 0.30)
    disrupted = (q_nat <= 0.40) | (rmsd_nm >= 0.70)

    p_fold = float(folded.mean())
    p_unfold = float(disrupted.mean())
    eps = 1e-4
    kbt_300 = 0.0019872041 * 300.0
    dg = float(kbt_300 * math.log((p_fold + eps) / (p_unfold + eps)))

    n = float(max(n_frames, 1))
    std = float(kbt_300 * math.sqrt((1.0 / (n * (p_fold + eps))) + (1.0 / (n * (p_unfold + eps)))))

    feats = {
        "n_frames": n_frames,
        "n_ca": n_ca,
        "n_native_contacts": int(d0.shape[0]),
        "q_nat_mean": float(q_nat.mean()),
        "q_nat_std": float(q_nat.std()),
        "rmsd_bb_mean_ang": float(rmsd_nm.mean() * 10.0),
        "rmsd_bb_std_ang": float(rmsd_nm.std() * 10.0),
        "p_fold": p_fold,
        "p_unfold": p_unfold,
    }
    return feats, dg, std


def _update_batch_size_100(
    current: int,
    control_vram_bytes: int | None,
    total_vram_bytes: int | None,
    *,
    target_frac: float,
    low_utilization_mult: float,
    max_growth_factor: float,
    max_shrink_factor: float,
    min_value: int,
    max_value: int,
) -> int:
    """Adaptive controller for BioEmu `batch_size_100`."""
    if control_vram_bytes is None or total_vram_bytes is None or total_vram_bytes <= 0:
        return current
    frac = float(control_vram_bytes) / float(total_vram_bytes)
    if frac <= 0.0:
        return current

    scaled = float(current) * (float(target_frac) / frac)
    if frac < float(target_frac) * float(low_utilization_mult):
        desired = scaled
    else:
        desired = 0.30 * float(current) + 0.70 * scaled

    hi = float(current) * max(1.0, float(max_growth_factor))
    lo = float(current) * max(0.0, float(max_shrink_factor))
    desired = min(desired, hi)
    desired = max(desired, lo)
    next_value = int(round(desired))
    return int(max(min_value, min(max_value, next_value)))


def _estimate_batch_size_100_from_alpha(
    *,
    alpha_bytes_per_l2b: float | None,
    total_vram_bytes: int | None,
    target_frac: float,
    min_value: int,
    max_value: int,
) -> int | None:
    if alpha_bytes_per_l2b is None or alpha_bytes_per_l2b <= 0.0:
        return None
    if total_vram_bytes is None or total_vram_bytes <= 0:
        return None
    budget = float(total_vram_bytes) * float(target_frac)
    # control_bytes ~= alpha * (batch_size_100 * 100^2)
    desired_b100 = budget / (float(alpha_bytes_per_l2b) * 10000.0)
    out = int(round(desired_b100))
    return int(max(min_value, min(max_value, out)))


@dataclass
class _CandidateState:
    row_idx: int
    rec: dict
    sequence: str
    seq_len: int
    reference_path: Path
    chain_id: str | None
    remaining: int
    produced: int
    base_seed: int
    context_chemgraph: Any | None
    tmpdir_obj: tempfile.TemporaryDirectory | None
    tmpdir: Path | None
    last_batch_size_100: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    total_vram_bytes: int | None
    vram_source: str | None
    cuda_device_index: int | None
    cuda_device_name: str | None

    @property
    def l2_cost(self) -> int:
        return int(self.seq_len * self.seq_len)



def _ensure_runtime(
    *,
    model_name: str,
    denoiser_type: str,
    denoiser_config_path: str | None,
    cache_so3_dir: Path,
):
    import hydra
    import torch
    import yaml
    from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
    from bioemu.sample import DEFAULT_DENOISER_CONFIG_DIR, SUPPORTED_DENOISERS

    ckpt_path, model_config_path = maybe_download_checkpoint(model_name=model_name, ckpt_path=None, model_config_path=None)
    score_model = load_model(ckpt_path, model_config_path)
    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=str(cache_so3_dir))

    if denoiser_config_path is None:
        if denoiser_type not in SUPPORTED_DENOISERS:
            raise ValueError(f"denoiser_type must be one of {SUPPORTED_DENOISERS}")
        denoiser_config_path = str(DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml")

    with open(denoiser_config_path, "r", encoding="utf-8") as f:
        denoiser_cfg = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return {
        "score_model": score_model,
        "sdes": sdes,
        "denoiser": denoiser,
        "device": device,
    }



def _ensure_context(
    state: _CandidateState,
    *,
    cache_embeds_dir: Path,
):
    if state.context_chemgraph is not None:
        return state.context_chemgraph
    from bioemu.sample import get_context_chemgraph

    state.context_chemgraph = get_context_chemgraph(
        sequence=state.sequence,
        cache_embeds_dir=str(cache_embeds_dir),
        msa_file=None,
        msa_host_url=None,
    )
    return state.context_chemgraph



def _ensure_tmpdir(state: _CandidateState) -> Path:
    if state.tmpdir is not None and state.tmpdir_obj is not None:
        return state.tmpdir
    tmp_obj = tempfile.TemporaryDirectory(prefix="bioemu_tmp_")
    state.tmpdir_obj = tmp_obj
    state.tmpdir = Path(tmp_obj.name)
    return state.tmpdir



def _pack_states_for_step(
    active: list[_CandidateState],
    *,
    batch_size_100: int,
    max_proteins_per_step: int,
) -> tuple[list[tuple[_CandidateState, int]], int, int]:
    """Pack multiple proteins into one denoiser step.

    Uses an L^2 budget in "sample-equivalent" units:
    budget_l2 = batch_size_100 * 100^2.
    """
    budget_l2 = max(1, int(batch_size_100 * 100 * 100))
    tasks: list[list[Any]] = []  # [state, n]
    used_l2 = 0

    # Pass 1: try assigning one sample per protein to increase proteins/step.
    for st in active:
        if st.remaining <= 0:
            continue
        if len(tasks) >= max_proteins_per_step:
            break
        cost = max(1, st.l2_cost)
        if used_l2 + cost <= budget_l2:
            tasks.append([st, 1])
            used_l2 += cost

    if not tasks:
        # Always make forward progress.
        st = max(active, key=lambda s: (s.l2_cost, s.remaining))
        tasks.append([st, 1])
        used_l2 = max(1, st.l2_cost)

    task_idx: dict[int, int] = {id(t[0]): i for i, t in enumerate(tasks)}

    # Pass 2: fill remaining budget, preferring already-selected states.
    for st in active:
        if st.remaining <= 0:
            continue
        cost = max(1, st.l2_cost)
        room = (budget_l2 - used_l2) // cost
        if room <= 0:
            continue
        if id(st) in task_idx:
            i = task_idx[id(st)]
            already = int(tasks[i][1])
            add = min(room, st.remaining - already)
            if add > 0:
                tasks[i][1] = already + int(add)
                used_l2 += int(add) * cost
        elif len(tasks) < max_proteins_per_step:
            add = min(room, st.remaining)
            if add > 0:
                task_idx[id(st)] = len(tasks)
                tasks.append([st, int(add)])
                used_l2 += int(add) * cost

    packed = [(t[0], int(t[1])) for t in tasks if int(t[1]) > 0]
    return packed, int(used_l2), int(budget_l2)



def _run_packed_step(
    *,
    runtime: dict,
    tasks: list[tuple[_CandidateState, int]],
    cache_embeds_dir: Path,
    require_torch_cuda_vram: bool,
    step_seed: int,
) -> dict:
    from bioemu.utils import format_npz_samples_filename
    import torch
    from torch_geometric.data.batch import Batch

    cuda_ok = bool(torch.cuda.is_available())
    if require_torch_cuda_vram and not cuda_ok:
        raise RuntimeError("Torch CUDA is unavailable; strict VRAM telemetry required")

    data_list = []
    slices: list[tuple[_CandidateState, int, int]] = []
    offset = 0
    for st, n in tasks:
        ctx = _ensure_context(st, cache_embeds_dir=cache_embeds_dir)
        data_list.extend([ctx] * int(n))
        slices.append((st, offset, int(n)))
        offset += int(n)

    telemetry: dict[str, Any] = {}
    total_vram_bytes: int | None = None
    if cuda_ok:
        device_index = torch.cuda.current_device()
        total_vram_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        telemetry.update(
            {
                "vram_source": "torch.cuda",
                "cuda_device_index": int(device_index),
                "cuda_device_name": str(torch.cuda.get_device_name(device_index)),
                "start_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
                "start_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            }
        )
        torch.cuda.reset_peak_memory_stats(device_index)

    torch.manual_seed(int(step_seed))
    with torch.no_grad():
        context_batch = Batch.from_data_list(data_list)
        sampled_batch = runtime["denoiser"](
            sdes=runtime["sdes"],
            device=runtime["device"],
            batch=context_batch,
            score_model=runtime["score_model"],
        )
        sampled_list = sampled_batch.to_data_list()

    if cuda_ok:
        device_index = torch.cuda.current_device()
        torch.cuda.synchronize()
        telemetry.update(
            {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
                "end_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
                "end_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            }
        )

    for st, start, count in slices:
        end = start + count
        sample_slice = sampled_list[start:end]
        pos = torch.stack([x.pos for x in sample_slice]).to("cpu").numpy()
        ori = torch.stack([x.node_orientations for x in sample_slice]).to("cpu").numpy()
        tmpdir = _ensure_tmpdir(st)
        out_npz = tmpdir / format_npz_samples_filename(st.produced, count)
        np.savez(out_npz, pos=pos, node_orientations=ori, sequence=st.sequence)
        st.produced += count
        st.remaining -= count

    return {
        "telemetry": telemetry,
        "total_vram_bytes": total_vram_bytes,
    }



def _finalize_state(
    state: _CandidateState,
    *,
    filter_samples: bool,
) -> tuple[dict, float, float]:
    from bioemu.convert_chemgraph import save_pdb_and_xtc
    import torch

    if state.tmpdir is None:
        raise RuntimeError("candidate temporary directory missing")

    sample_files = sorted(state.tmpdir.glob("batch_*.npz"))
    if not sample_files:
        raise RuntimeError("no sampled batches found to finalize candidate")

    pos_blocks = []
    ori_blocks = []
    seqs = set()
    for sf in sample_files:
        with np.load(sf, allow_pickle=False) as arr:
            pos_blocks.append(arr["pos"])
            ori_blocks.append(arr["node_orientations"])
            seqs.add(str(arr["sequence"].item()))
    if seqs != {state.sequence}:
        raise RuntimeError(f"sample sequence mismatch while finalizing candidate: {seqs}")

    positions = torch.tensor(np.concatenate(pos_blocks, axis=0))
    node_orientations = torch.tensor(np.concatenate(ori_blocks, axis=0))

    top_pdb = state.tmpdir / "topology.pdb"
    xtc = state.tmpdir / "samples.xtc"
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=top_pdb,
        xtc_path=xtc,
        sequence=state.sequence,
        filter_samples=bool(filter_samples),
    )

    ref_ca = _load_reference_ca_coords(state.reference_path, state.chain_id)
    feats, dg, std = _compute_features_from_samples(xtc, top_pdb, ref_ca)

    if state.tmpdir_obj is not None:
        state.tmpdir_obj.cleanup()
        state.tmpdir_obj = None
        state.tmpdir = None

    return feats, dg, std



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--model-name", default="bioemu-v1.1")
    parser.add_argument("--batch-size-100", type=int, default=10)
    parser.add_argument("--target-vram-frac", type=float, default=0.90)
    parser.add_argument("--batch-size-100-min", type=int, default=1)
    parser.add_argument("--batch-size-100-max", type=int, default=512)
    parser.add_argument(
        "--vram-control-metric",
        choices=["allocated", "reserved"],
        default="reserved",
        help="Which torch CUDA peak metric to control against target_vram_frac.",
    )
    parser.add_argument(
        "--low-utilization-mult",
        type=float,
        default=0.5,
        help="If utilization < target*mult, use aggressive scaling.",
    )
    parser.add_argument(
        "--batch-size-100-max-growth-factor",
        type=float,
        default=6.0,
        help="Maximum multiplicative growth per update.",
    )
    parser.add_argument(
        "--batch-size-100-max-shrink-factor",
        type=float,
        default=0.5,
        help="Maximum multiplicative shrink per update.",
    )
    parser.add_argument("--filter-samples", dest="filter_samples", action="store_true")
    parser.add_argument("--no-filter-samples", dest="filter_samples", action="store_false")
    parser.add_argument("--sort-by-length-desc", dest="sort_by_length_desc", action="store_true")
    parser.add_argument("--no-sort-by-length-desc", dest="sort_by_length_desc", action="store_false")
    parser.add_argument("--auto-batch-from-vram", dest="auto_batch_from_vram", action="store_true")
    parser.add_argument("--no-auto-batch-from-vram", dest="auto_batch_from_vram", action="store_false")
    parser.add_argument(
        "--max-proteins-per-step",
        type=int,
        default=8,
        help="Maximum number of proteins packed into one denoiser step.",
    )
    parser.add_argument(
        "--require-torch-cuda-vram",
        dest="require_torch_cuda_vram",
        action="store_true",
        help="Fail if torch CUDA VRAM telemetry is unavailable.",
    )
    parser.add_argument(
        "--no-require-torch-cuda-vram",
        dest="require_torch_cuda_vram",
        action="store_false",
        help="Allow running when torch CUDA VRAM telemetry is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(
        auto_batch_from_vram=True,
        require_torch_cuda_vram=True,
        filter_samples=True,
        sort_by_length_desc=True,
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    # Normalize runtime/cache paths so direct python invocations do not fall back
    # to ~/.bioemu_colabfold or ~/.cache/huggingface.
    hf_home = Path(os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))).resolve()
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(hf_home / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str((root / ".cache" / "xdg").resolve()))
    os.environ.setdefault("UV_CACHE_DIR", str((root / ".cache" / "uv").resolve()))
    os.environ.setdefault("PIP_CACHE_DIR", str((root / ".cache" / "pip").resolve()))
    os.environ.setdefault("BIOEMU_COLABFOLD_DIR", str((root / ".cache" / "bioemu" / "colabfold").resolve()))

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PIP_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["BIOEMU_COLABFOLD_DIR"]).mkdir(parents=True, exist_ok=True)

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging
    import torch

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    logger = configure_logging("oracle.bioemu", level=args.log_level)
    rows = read_records(root / args.candidate_path)
    logger.info(
        "BioEmu scoring start: candidates=%d model=%s samples=%d batch_size_100=%d auto_vram=%s target_vram=%.2f require_torch_cuda_vram=%s filter_samples=%s control_metric=%s sort_len_desc=%s max_proteins_per_step=%d",
        len(rows),
        args.model_name,
        args.num_samples,
        args.batch_size_100,
        args.auto_batch_from_vram,
        args.target_vram_frac,
        args.require_torch_cuda_vram,
        args.filter_samples,
        args.vram_control_metric,
        args.sort_by_length_desc,
        args.max_proteins_per_step,
    )
    if args.require_torch_cuda_vram and (not torch.cuda.is_available()):
        raise RuntimeError("Torch CUDA is unavailable; strict VRAM telemetry required for BioEmu scoring")

    cache_root = root / ".cache" / "bioemu"
    cache_so3_dir = cache_root / "sampling_so3_cache"
    cache_embeds_dir = cache_root / "embeds_cache"
    cache_so3_dir.mkdir(parents=True, exist_ok=True)
    cache_embeds_dir.mkdir(parents=True, exist_ok=True)
    logger.info("BioEmu cache dirs: so3=%s embeds=%s", cache_so3_dir, cache_embeds_dir)

    out: list[dict] = [dict(r) for r in rows]
    states: list[_CandidateState] = []
    for i, rec in enumerate(rows):
        if not rec.get("eligibility", {}).get("bioemu", False):
            out[i]["bioemu_features"] = None
            out[i]["bioemu_calibrated"] = None
            out[i]["bioemu_std"] = None
            out[i]["bioemu_status"] = "ineligible"
            continue
        seq = str(rec.get("sequence", ""))
        if not seq:
            raise RuntimeError(f"missing sequence for candidate_id={rec.get('candidate_id', 'unknown')}")
        state = _CandidateState(
            row_idx=i,
            rec=dict(rec),
            sequence=seq,
            seq_len=len(seq),
            reference_path=Path(str(rec["cif_path"])).resolve(),
            chain_id=(rec.get("chain_id") or rec.get("chain")),
            remaining=int(args.num_samples),
            produced=0,
            base_seed=int(args.seed + i),
            context_chemgraph=None,
            tmpdir_obj=None,
            tmpdir=None,
            last_batch_size_100=int(args.batch_size_100),
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            total_vram_bytes=None,
            vram_source=None,
            cuda_device_index=None,
            cuda_device_name=None,
        )
        states.append(state)

    if args.sort_by_length_desc:
        states.sort(key=lambda s: s.seq_len, reverse=True)
    else:
        states.sort(key=lambda s: s.seq_len)

    if not states:
        write_records(root / args.output_path, out)
        logger.info("BioEmu scoring complete: wrote=%d eligible=0 skipped=%d elapsed=%.2fs", len(out), len(out), time.perf_counter() - t0)
        print(root / args.output_path)
        return 0

    runtime = _ensure_runtime(
        model_name=args.model_name,
        denoiser_type="dpm",
        denoiser_config_path=None,
        cache_so3_dir=cache_so3_dir,
    )

    next_batch_size_100 = int(max(args.batch_size_100_min, min(args.batch_size_100_max, args.batch_size_100)))
    alpha_bytes_per_l2b: float | None = None
    step_id = 0
    completed = 0
    progress_candidates = None
    progress_samples = None
    total_samples = int(len(states) * int(args.num_samples))
    processed_samples = 0
    if tqdm is not None and not args.no_progress:
        progress_candidates = tqdm(
            total=len(states),
            desc="bioemu:score",
            dynamic_ncols=True,
            leave=True,
        )
        progress_samples = tqdm(
            total=total_samples,
            desc="bioemu:samples",
            dynamic_ncols=True,
            leave=True,
        )

    while completed < len(states):
        active = [s for s in states if s.remaining > 0]
        if not active:
            break

        step_id += 1
        packed, used_l2, budget_l2 = _pack_states_for_step(
            active,
            batch_size_100=next_batch_size_100,
            max_proteins_per_step=max(1, int(args.max_proteins_per_step)),
        )
        if not packed:
            raise RuntimeError("internal error: no packed tasks produced while candidates remain")
        step_samples = int(sum(int(c) for _, c in packed))

        try:
            step = _run_packed_step(
                runtime=runtime,
                tasks=packed,
                cache_embeds_dir=cache_embeds_dir,
                require_torch_cuda_vram=args.require_torch_cuda_vram,
                step_seed=args.seed + step_id,
            )
        except Exception as exc:  # noqa: BLE001
            bad_ids = [t[0].rec.get("candidate_id", "unknown") for t in packed]
            raise RuntimeError(f"BioEmu packed step failed for candidate_ids={bad_ids}: {exc}") from exc

        processed_samples += step_samples
        if progress_samples is not None:
            progress_samples.update(step_samples)
            progress_samples.set_postfix(
                step=step_id,
                packed=len(packed),
                b100=next_batch_size_100,
                l2=f"{used_l2}/{budget_l2}",
            )

        telemetry = step.get("telemetry") or {}
        total_vram_bytes = step.get("total_vram_bytes")
        peak_alloc = telemetry.get("peak_allocated_bytes")
        peak_reserved = telemetry.get("peak_reserved_bytes")
        control_vram_bytes = peak_reserved if args.vram_control_metric == "reserved" else peak_alloc

        if control_vram_bytes is not None and used_l2 > 0:
            local_alpha = float(control_vram_bytes) / float(used_l2)
            alpha_bytes_per_l2b = local_alpha if alpha_bytes_per_l2b is None else max(alpha_bytes_per_l2b, local_alpha)

        for st, _count in packed:
            st.last_batch_size_100 = int(next_batch_size_100)
            if peak_alloc is not None:
                st.peak_allocated_bytes = (
                    int(peak_alloc)
                    if st.peak_allocated_bytes is None
                    else max(int(st.peak_allocated_bytes), int(peak_alloc))
                )
            if peak_reserved is not None:
                st.peak_reserved_bytes = (
                    int(peak_reserved)
                    if st.peak_reserved_bytes is None
                    else max(int(st.peak_reserved_bytes), int(peak_reserved))
                )
            if total_vram_bytes is not None:
                st.total_vram_bytes = int(total_vram_bytes)
            st.vram_source = telemetry.get("vram_source")
            st.cuda_device_index = telemetry.get("cuda_device_index")
            st.cuda_device_name = telemetry.get("cuda_device_name")

        finalized_now = 0
        for st, _count in packed:
            if st.remaining > 0:
                continue
            feats, dg, std = _finalize_state(st, filter_samples=args.filter_samples)
            rec = out[st.row_idx]
            rec["bioemu_features"] = feats
            rec["bioemu_calibrated"] = dg
            rec["bioemu_std"] = std
            rec["bioemu_batch_size_100"] = int(st.last_batch_size_100)
            rec["bioemu_peak_vram_bytes"] = int(st.peak_allocated_bytes) if st.peak_allocated_bytes is not None else None
            rec["bioemu_total_vram_bytes"] = int(st.total_vram_bytes) if st.total_vram_bytes is not None else None
            rec["bioemu_peak_vram_frac"] = (
                float(st.peak_allocated_bytes) / float(st.total_vram_bytes)
                if st.peak_allocated_bytes is not None and st.total_vram_bytes is not None and st.total_vram_bytes > 0
                else None
            )
            rec["bioemu_vram_source"] = st.vram_source
            rec["bioemu_peak_reserved_vram_bytes"] = int(st.peak_reserved_bytes) if st.peak_reserved_bytes is not None else None
            rec["bioemu_start_allocated_vram_bytes"] = telemetry.get("start_allocated_bytes")
            rec["bioemu_start_reserved_vram_bytes"] = telemetry.get("start_reserved_bytes")
            rec["bioemu_end_allocated_vram_bytes"] = telemetry.get("end_allocated_bytes")
            rec["bioemu_end_reserved_vram_bytes"] = telemetry.get("end_reserved_bytes")
            rec["bioemu_cuda_device_index"] = st.cuda_device_index
            rec["bioemu_cuda_device_name"] = st.cuda_device_name
            rec["bioemu_vram_control_metric"] = args.vram_control_metric
            cand_control = st.peak_reserved_bytes if args.vram_control_metric == "reserved" else st.peak_allocated_bytes
            rec["bioemu_control_vram_bytes"] = int(cand_control) if cand_control is not None else None
            rec["bioemu_control_vram_frac"] = (
                float(cand_control) / float(st.total_vram_bytes)
                if cand_control is not None and st.total_vram_bytes is not None and st.total_vram_bytes > 0
                else None
            )
            rec["bioemu_step_id"] = int(step_id)
            rec["bioemu_packed_proteins_in_step"] = int(len(packed))
            rec["bioemu_packed_l2_used"] = int(used_l2)
            rec["bioemu_packed_l2_budget"] = int(budget_l2)
            rec["bioemu_status"] = "ok"

            completed += 1
            finalized_now += 1
            if progress_candidates is not None:
                progress_candidates.update(1)
                progress_candidates.set_postfix(
                    step=step_id,
                    packed=len(packed),
                    b100=next_batch_size_100,
                    done=f"{completed}/{len(states)}",
                    samples=f"{processed_samples}/{total_samples}",
                )

        if args.auto_batch_from_vram:
            next_from_obs = _update_batch_size_100(
                next_batch_size_100,
                int(control_vram_bytes) if control_vram_bytes is not None else None,
                int(total_vram_bytes) if total_vram_bytes is not None else None,
                target_frac=args.target_vram_frac,
                low_utilization_mult=args.low_utilization_mult,
                max_growth_factor=args.batch_size_100_max_growth_factor,
                max_shrink_factor=args.batch_size_100_max_shrink_factor,
                min_value=args.batch_size_100_min,
                max_value=args.batch_size_100_max,
            )
            next_batch_size_100 = int(next_from_obs)
            next_from_alpha = _estimate_batch_size_100_from_alpha(
                alpha_bytes_per_l2b=alpha_bytes_per_l2b,
                total_vram_bytes=int(total_vram_bytes) if total_vram_bytes is not None else None,
                target_frac=args.target_vram_frac,
                min_value=args.batch_size_100_min,
                max_value=args.batch_size_100_max,
            )
            if next_from_alpha is not None:
                next_batch_size_100 = int(next_from_alpha)

        if step_id == 1 or (step_id % 8) == 0 or finalized_now > 0:
            logger.info(
                "BioEmu packed step=%d completed=%d/%d proteins_in_step=%d finalized_now=%d batch_size_100=%d l2_used=%d l2_budget=%d control_frac=%s peak_alloc_gib=%s peak_reserved_gib=%s alpha_bytes_per_l2b=%s next_batch_size_100=%d",
                step_id,
                completed,
                len(states),
                len(packed),
                finalized_now,
                next_batch_size_100,
                used_l2,
                budget_l2,
                (
                    f"{(float(control_vram_bytes)/float(total_vram_bytes)):.3f}"
                    if control_vram_bytes is not None and total_vram_bytes is not None and total_vram_bytes > 0
                    else "n/a"
                ),
                (
                    f"{float(peak_alloc)/float(1024**3):.3f}"
                    if peak_alloc is not None
                    else "n/a"
                ),
                (
                    f"{float(peak_reserved)/float(1024**3):.3f}"
                    if peak_reserved is not None
                    else "n/a"
                ),
                (f"{alpha_bytes_per_l2b:.2f}" if alpha_bytes_per_l2b is not None else "n/a"),
                next_batch_size_100,
            )

    if progress_candidates is not None:
        progress_candidates.close()
    if progress_samples is not None:
        progress_samples.close()

    eligible = sum(1 for r in out if r.get("bioemu_status") == "ok")
    skipped = len(out) - eligible
    write_records(root / args.output_path, out)
    logger.info(
        "BioEmu scoring complete: wrote=%d eligible=%d skipped=%d elapsed=%.2fs",
        len(out),
        eligible,
        skipped,
        time.perf_counter() - t0,
    )
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
