#!/usr/bin/env python3
"""Pack candidate sequences onto reactant/product complexes with LigandMPNN."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile
import time
from pathlib import Path
import sys


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _ensure_env_site_packages() -> None:
    """Prepend the active env's site-packages to sys.path.

    This makes the packer resilient to parent-shell Python contamination when the
    script is launched via nested env dispatch.
    """

    prefix_raw = os.environ.get("CONDA_PREFIX")
    if prefix_raw:
        prefix = Path(prefix_raw).resolve()
    else:
        py = Path(sys.executable).resolve()
        prefix = py.parent.parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = prefix / "lib" / version / "site-packages"
    if site.exists():
        site_str = str(site)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _prepare_pdb(source: Path, dest: Path) -> None:
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
            try:
                _prepare_pdb(Path(tmp.name), dest)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return
    raise RuntimeError(f"unsupported structure extension: {source}")


def _write_protein_only_pdb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    keep = ("ATOM  ", "TER", "END", "MODEL ", "ENDMDL")
    with src.open("r", encoding="utf-8") as in_fh, dst.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            if line.startswith(keep):
                out_fh.write(line)


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_packer(checkpoint_sc: Path, device: str):
    import torch
    from sc_utils import Packer

    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model_sc = Packer(
        node_features=128,
        edge_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        top_k=32,
        dropout=0.0,
        augment_eps=0.0,
        atom37_order=False,
        device=torch_device,
        num_mix=3,
    )
    checkpoint = torch.load(str(checkpoint_sc), map_location=torch_device)
    model_sc.load_state_dict(checkpoint["model_state_dict"])
    model_sc.to(torch_device)
    model_sc.eval()
    return model_sc, torch_device


def _encode_sequence_for_structure(protein_dict: dict, sequence: str, protein_chain_id: str | None):
    import torch
    from data_utils import restype_str_to_int

    seq = sequence.strip()
    original = protein_dict["S"].clone()
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
            raise ValueError(f"unsupported amino acid '{aa}' in target sequence")
        original[i] = restype_str_to_int[aa]
    protein_dict["S"] = original.to(dtype=torch.long, device=original.device)
    chain_mask = torch.zeros(len(chain_letters), dtype=torch.long, device=original.device)
    chain_mask[design_idx] = 1
    protein_dict["chain_mask"] = chain_mask
    return protein_dict


def _pack_one_state(
    *,
    source_path: Path,
    out_dir: Path,
    sequence: str,
    protein_chain_id: str | None,
    model_sc,
    torch_device,
    pack_with_ligand_context: bool,
    repack_everything: bool,
    sc_num_denoising_steps: int,
    sc_num_samples: int,
    parse_atoms_with_zero_occupancy: int,
    force_hetatm: int,
) -> tuple[Path, Path]:
    from data_utils import featurize, parse_PDB, write_full_PDB
    from sc_utils import pack_side_chains

    prepared_pdb = out_dir / "source_complex.pdb"
    _prepare_pdb(source_path, prepared_pdb)

    protein_dict, _backbone, other_atoms, icodes, _ = parse_PDB(
        str(prepared_pdb),
        device=str(torch_device),
        chains=[],
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=int(parse_atoms_with_zero_occupancy),
    )
    protein_dict = _encode_sequence_for_structure(protein_dict, sequence, protein_chain_id)
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=bool(pack_with_ligand_context),
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )
    sc_dict = pack_side_chains(
        feature_dict,
        model_sc,
        int(sc_num_denoising_steps),
        int(sc_num_samples),
        bool(repack_everything),
    )

    packed_complex = out_dir / "packed_complex.pdb"
    write_full_PDB(
        str(packed_complex),
        sc_dict["X"][0].detach().cpu().numpy(),
        sc_dict["X_m"][0].detach().cpu().numpy(),
        sc_dict["b_factors"][0].detach().cpu().numpy(),
        feature_dict["R_idx_original"][0].detach().cpu().numpy(),
        protein_dict["chain_letters"],
        feature_dict["S"][0].detach().cpu().numpy(),
        other_atoms=other_atoms,
        icodes=icodes,
        force_hetatm=bool(force_hetatm),
    )
    packed_protein = out_dir / "packed_protein.pdb"
    _write_protein_only_pdb(packed_complex, packed_protein)
    return packed_complex, packed_protein


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ligandmpnn-root", default="models/LigandMPNN")
    parser.add_argument("--checkpoint-sc", default="models/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pack-with-ligand-context", type=int, default=1)
    parser.add_argument("--repack-everything", type=int, default=1)
    parser.add_argument("--sc-num-denoising-steps", type=int, default=3)
    parser.add_argument("--sc-num-samples", type=int, default=16)
    parser.add_argument("--parse-atoms-with-zero-occupancy", type=int, default=0)
    parser.add_argument("--force-hetatm", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    _ensure_env_site_packages()
    sys.path.insert(0, str(root))

    ligand_root = Path(args.ligandmpnn_root)
    if not ligand_root.is_absolute():
        ligand_root = root / ligand_root
    sys.path.insert(0, str(ligand_root.resolve()))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("oracle.ligandmpnn_pack", level=args.log_level)
    checkpoint_sc = Path(args.checkpoint_sc)
    if not checkpoint_sc.is_absolute():
        checkpoint_sc = root / checkpoint_sc
    model_sc, torch_device = _load_packer(checkpoint_sc.resolve(), args.device)

    input_path = _resolve_path(root, args.candidate_path)
    output_path = _resolve_path(root, args.output_path)
    output_root = _resolve_path(root, args.output_root)
    rows = read_records(input_path)
    logger.info(
        "LigandMPNN packing start: rows=%d output_root=%s device=%s ligand_context=%s",
        len(rows),
        output_root,
        torch_device,
        bool(args.pack_with_ligand_context),
    )

    out: list[dict] = []
    for rec in iter_progress(rows, total=len(rows), desc="pack:ligandmpnn", no_progress=args.no_progress):
        row = dict(rec)
        candidate_id = str(row.get("candidate_id") or "")
        sequence = str(row.get("sequence") or "").strip()
        protein_chain_id = row.get("protein_chain_id")
        candidate_dir = output_root / candidate_id
        try:
            if not candidate_id:
                raise ValueError("missing candidate_id")
            if not sequence:
                raise ValueError("missing sequence")

            for state in ("reactant", "product"):
                state_dir = candidate_dir / state
                source_key = f"{state}_complex_path"
                source_path = _resolve_path(root, row[source_key])
                packed_complex, packed_protein = _pack_one_state(
                    source_path=source_path,
                    out_dir=state_dir,
                    sequence=sequence,
                    protein_chain_id=str(protein_chain_id) if protein_chain_id else None,
                    model_sc=model_sc,
                    torch_device=torch_device,
                    pack_with_ligand_context=bool(args.pack_with_ligand_context),
                    repack_everything=bool(args.repack_everything),
                    sc_num_denoising_steps=int(args.sc_num_denoising_steps),
                    sc_num_samples=int(args.sc_num_samples),
                    parse_atoms_with_zero_occupancy=int(args.parse_atoms_with_zero_occupancy),
                    force_hetatm=int(args.force_hetatm),
                )
                row[f"{state}_complex_packed_path"] = str(packed_complex.resolve())
                row[f"{state}_protein_packed_path"] = str(packed_protein.resolve())

            row["protein_path"] = row["reactant_protein_packed_path"]
            row["complex_path"] = row["reactant_complex_packed_path"]
            row["packing_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["packing_status"] = "error"
            row["packing_error"] = str(exc)
            logger.error("Packing failed candidate_id=%s error=%s", candidate_id or "<unknown>", exc)
        out.append(row)

    write_records(output_path, out)
    logger.info("LigandMPNN packing complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
