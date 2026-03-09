#!/usr/bin/env python3
"""Score candidates with MMKcat for Kcat-mode active learning."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _parse_masks(text: str) -> list[list[bool]]:
    masks: list[list[bool]] = []
    for chunk in text.split(";"):
        tok = chunk.strip().replace(",", "")
        if not tok:
            continue
        if len(tok) != 4 or any(ch not in {"0", "1"} for ch in tok):
            raise ValueError(f"Invalid mask token '{chunk}' (expected 4-bit tokens, e.g. 1110)")
        masks.append([ch == "1" for ch in tok])
    if not masks:
        raise ValueError("No valid MMKcat masks parsed")
    return masks


def _as_smiles_list(value) -> list[str | None]:
    if value is None:
        return [None]
    if isinstance(value, list):
        out = []
        for v in value:
            if v is None:
                out.append(None)
            else:
                s = str(v).strip()
                out.append(s if s else None)
        return out or [None]
    s = str(value).strip()
    if not s:
        return [None]
    # Accept "a;b" / "a,b" for multi-product records.
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
        return [p for p in parts if p] or [None]
    return [s]


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


def _pick_products(rec: dict) -> list[str | None]:
    for key in ("product_smiles", "product"):
        if key in rec:
            return _as_smiles_list(rec.get(key))
    return [None]


def _cif_to_pdb(src: Path, out_dir: Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO

    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:12]
    dst = out_dir / f"{src.stem}_{digest}.pdb"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(src.stem, str(src))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-root", default="models/MMKcat")
    parser.add_argument("--checkpoint", default="models/MMKcat/ckpt/concat_best_checkpoint.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm2-device", default="cpu")
    parser.add_argument("--masks", default="1111;1110;1101;1100")
    parser.add_argument("--dssp-bin", default="mkdssp")
    parser.add_argument("--torch-home", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram, reset_peak_vram_tracking

    logger = configure_logging("oracle.mmkcat", level=args.log_level)
    model_root = Path(args.model_root)
    if not model_root.is_absolute():
        model_root = root / model_root
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    if not model_root.exists():
        raise FileNotFoundError(f"MMKcat model root not found: {model_root}")
    if not ckpt.exists():
        raise FileNotFoundError(
            "MMKcat checkpoint not found. Expected checkpoint at "
            f"{ckpt}. Download the released MMKcat checkpoint into models/MMKcat/ckpt."
        )

    masks = _parse_masks(args.masks)

    if args.torch_home:
        os.environ["TORCH_HOME"] = str(Path(args.torch_home).expanduser())
    dssp_path = str(Path(args.dssp_bin).expanduser())
    if "/" in dssp_path:
        dssp_dir = str(Path(dssp_path).resolve().parent)
        os.environ["PATH"] = f"{dssp_dir}:{os.environ.get('PATH', '')}"

    mm_model_dir = model_root / "model"
    mm_util_dir = model_root / "util"
    sys.path.insert(0, str(mm_model_dir))
    sys.path.insert(0, str(mm_util_dir))

    import torch
    import esm
    from basic_model_mm import mmKcatPrediction
    from generate_graph import pdb2graph

    candidates = read_records(root / args.candidate_path)
    logger.info(
        "MMKcat scoring start: candidates=%d model_root=%s checkpoint=%s device=%s masks=%s",
        len(candidates),
        model_root,
        ckpt,
        args.device,
        args.masks,
    )

    dev = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    esm2_dev = torch.device(args.esm2_device if str(args.esm2_device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    load_t0 = time.perf_counter()
    model = mmKcatPrediction(
        device=dev,
        batch_size=1,
        nhead=4,
        nhid=1024,
        nlayers=4,
        gcn_hidden=512,
        dropout=0.2,
        lambda_1=0.8,
        lambda_2=0.2,
        mode="test",
    ).to(dev)
    state = torch.load(str(ckpt), map_location=dev)
    model.load_state_dict(state)
    model.eval()

    esm2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2 = esm2.to(esm2_dev)
    esm2.eval()
    batch_converter = alphabet.get_batch_converter()

    esmfold = esm.pretrained.esmfold_v1()
    esmfold = esmfold.eval().to(dev)
    logger.info("MMKcat model stack loaded elapsed=%.2fs", time.perf_counter() - load_t0)

    mean_attr = mm_util_dir / "mean_attr.pt"
    if not mean_attr.exists():
        raise FileNotFoundError(f"MMKcat util mean_attr.pt missing: {mean_attr}")

    out: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="mmkcat_tmp_") as td:
        td_path = Path(td)
        pdb_cache = td_path / "pdb"
        pdb_cache.mkdir(parents=True, exist_ok=True)

        for i, rec in enumerate(
            iter_progress(candidates, total=len(candidates), desc="mmkcat:score", no_progress=args.no_progress),
            start=1,
        ):
            cid = rec.get("candidate_id")
            seq = str(rec.get("sequence") or "").strip()
            substrate = _pick_substrate_smiles(rec)
            products = _pick_products(rec)

            if not seq:
                raise RuntimeError(f"MMKcat candidate_id={cid} missing sequence")
            if not substrate:
                raise RuntimeError(
                    f"MMKcat candidate_id={cid} missing substrate smiles. "
                    "Provide substrate_smiles/Smiles in dataset records."
                )

            try:
                # ESM2 sequence embedding.
                data = [(str(cid or i), seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(esm2_dev)
                with torch.no_grad():
                    esm2_out = esm2(batch_tokens, repr_layers=[33], return_contacts=False)
                token_reps = esm2_out["representations"][33][0]
                seq_len = int((batch_tokens[0] != alphabet.padding_idx).sum().item())
                seq_rep = token_reps[1 : seq_len - 1].mean(0).detach().cpu().reshape(1, -1)

                # Build structure graph via ESMFold output + MMKcat graphizer.
                pdb_out = esmfold.infer_pdb(seq)
                pdb_path = pdb_cache / f"{cid or i}.pdb"
                pdb_path.write_text(pdb_out)
                graph = pdb2graph(str(pdb_path), str(mean_attr))
                if graph is None:
                    raise RuntimeError("pdb2graph returned None")

                substrate_list = [substrate]
                product_list = products if products else [None]
                data_mm = [
                    [substrate_list],
                    [seq_rep],
                    [(graph.x, graph.edge_index)],
                    [product_list],
                    [torch.tensor(0.0)],
                ]

                preds: list[float] = []
                reset_peak_vram_tracking()
                with torch.no_grad():
                    for mask in masks:
                        model.test_mask = np.array(mask, dtype=bool)
                        outputs = model(data_mm)
                        pred = float(outputs[-1].detach().cpu().reshape(-1)[0].item())
                        preds.append(pred)
                log_peak_vram(logger, label="mmkcat:inference")

                rec["mmkcat_log10"] = float(np.mean(preds))
                rec["mmkcat_std"] = float(np.std(preds))
                rec["mmkcat_mask_predictions"] = preds
                rec["mmkcat_num_masks"] = len(preds)
                rec["mmkcat_status"] = "ok"
                out.append(rec)

                if i == 1 or (i % 50) == 0 or i == len(candidates):
                    logger.info(
                        "MMKcat progress: %d/%d cid=%s mean=%.4f std=%.4f",
                        i,
                        len(candidates),
                        cid,
                        rec["mmkcat_log10"],
                        rec["mmkcat_std"],
                    )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"MMKcat scoring failed candidate_id={cid}: {exc}") from exc

    write_records(root / args.output_path, out)
    logger.info("MMKcat scoring complete: wrote=%d elapsed=%.2fs", len(out), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
