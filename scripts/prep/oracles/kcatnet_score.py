#!/usr/bin/env python3
"""Score candidates with KcatNet for Kcat-mode active learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _pick_substrate_smiles(rec: dict) -> str | None:
    for key in ("substrate_smiles", "Smiles", "smiles", "ligand_smiles"):
        val = rec.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    return s
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-root", default="models/KcatNet")
    parser.add_argument("--checkpoint", default="models/KcatNet/RESULT/model_KcatNet.pt")
    parser.add_argument("--config-path", default="models/KcatNet/config_KcatNet.json")
    parser.add_argument("--degree-path", default="models/KcatNet/Dataset/degree.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--std-default", type=float, default=0.25)
    parser.add_argument("--prott5-model", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--prott5-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram, reset_peak_vram_tracking

    logger = configure_logging("oracle.kcatnet", level=args.log_level)

    model_root = Path(args.model_root)
    if not model_root.is_absolute():
        model_root = root / model_root
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    degree_path = Path(args.degree_path)
    if not degree_path.is_absolute():
        degree_path = root / degree_path

    if not model_root.exists():
        raise FileNotFoundError(f"KcatNet model root not found: {model_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"KcatNet checkpoint not found: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"KcatNet config not found: {config_path}")
    if not degree_path.exists():
        raise FileNotFoundError(f"KcatNet degree file not found: {degree_path}")

    if args.prott5_dir:
        os.environ["KCATNET_PROTT5_DIR"] = str(Path(args.prott5_dir).expanduser())
    os.environ["KCATNET_PROTT5_MODEL"] = str(args.prott5_model)

    rows = read_records(root / args.candidate_path)
    logger.info(
        "KcatNet scoring start: candidates=%d model_root=%s checkpoint=%s device=%s batch_size=%d",
        len(rows),
        model_root,
        checkpoint,
        args.device,
        args.batch_size,
    )

    # Validate and normalize scoring inputs.
    df_rows: list[dict[str, str]] = []
    for rec in iter_progress(rows, total=len(rows), desc="kcatnet:prepare", no_progress=args.no_progress):
        cid = rec.get("candidate_id")
        seq = str(rec.get("sequence") or "").strip()
        smi = _pick_substrate_smiles(rec)
        if not seq:
            raise RuntimeError(f"KcatNet candidate_id={cid} missing sequence")
        if not smi:
            raise RuntimeError(
                f"KcatNet candidate_id={cid} missing substrate smiles. "
                "Provide substrate_smiles/Smiles in dataset records."
            )
        if Chem.MolFromSmiles(smi) is None:
            raise RuntimeError(f"KcatNet candidate_id={cid} has invalid substrate smiles: {smi}")
        df_rows.append({"Pro_seq": seq, "Smile": smi})

    # Import KcatNet modules after env vars are set.
    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(model_root / "utils"))
    sys.path.insert(0, str(model_root / "models"))

    import torch
    from torch_geometric.loader import DataLoader
    from models.model_kcat import KcatNet
    from utils.Kcat_Dataset import EnzMolDataset
    from utils.ligand_init import ligand_init
    from utils.protein_init import protein_init
    from utils.trainer import pred

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    degree_dict = torch.load(str(degree_path), map_location="cpu")
    prot_deg = degree_dict["protein_deg"]

    model = KcatNet(
        prot_deg,
        mol_in_channels=config["params"]["mol_in_channels"],
        prot_in_channels=config["params"]["prot_in_channels"],
        prot_evo_channels=config["params"]["prot_evo_channels"],
        hidden_channels=config["params"]["hidden_channels"],
        pre_layers=config["params"]["pre_layers"],
        post_layers=config["params"]["post_layers"],
        aggregators=config["params"]["aggregators"],
        scalers=config["params"]["scalers"],
        total_layer=config["params"]["total_layer"],
        K=config["params"]["K"],
        heads=config["params"]["heads"],
        dropout=config["params"]["dropout"],
        dropout_attn_score=config["params"]["dropout_attn_score"],
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(str(checkpoint), map_location=device))
    model.eval()

    score_df = pd.DataFrame(df_rows, columns=["Pro_seq", "Smile"])
    unique_proteins = list(dict.fromkeys(score_df["Pro_seq"].tolist()))
    unique_smiles = list(dict.fromkeys(score_df["Smile"].tolist()))
    logger.info(
        "KcatNet feature init: unique_proteins=%d unique_smiles=%d",
        len(unique_proteins),
        len(unique_smiles),
    )

    protein_dict = protein_init(unique_proteins)
    ligand_dict = ligand_init(unique_smiles)

    dataset = EnzMolDataset(score_df, ligand_dict, protein_dict)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["mol_x", "prot_node_esm"],
    )

    reset_peak_vram_tracking()
    preds = pred(model, loader, device=str(args.device))
    log_peak_vram(logger, label="kcatnet:inference")
    if len(preds) != len(rows):
        raise RuntimeError(f"KcatNet prediction count mismatch: preds={len(preds)} rows={len(rows)}")

    out: list[dict] = []
    for rec, p in zip(
        iter_progress(rows, total=len(rows), desc="kcatnet:merge", no_progress=args.no_progress),
        preds,
    ):
        rec["kcatnet_log10"] = float(p)
        rec["kcatnet_kcat"] = float(10.0 ** float(p))
        rec["kcatnet_std"] = float(args.std_default)
        rec["kcatnet_status"] = "ok"
        out.append(rec)

    write_records(root / args.output_path, out)
    logger.info("KcatNet scoring complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
