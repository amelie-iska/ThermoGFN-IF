#!/usr/bin/env python3
"""Batch ADFLIP sequence generation with one-time model load.

Run this inside the ADFLIP environment from `models/ADFLIP`:
  PYTHONPATH=. python ../../scripts/prep/oracles/adflip_batch_generate.py ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ema_pytorch import EMA
from tqdm.auto import tqdm

from data import all_atom_parse as aap
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN


class Config:
    """Checkpoint compatibility shim for pickled __main__.Config objects."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def build_denoiser(config, device: torch.device):
    cfg = config.zoidberg_denoiser
    model = Zoidberg_GNN(
        hidden_dim=cfg.hidden_dim,
        encoder_hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_layers,
        num_heads=cfg.num_heads,
        k=cfg.k_neighbors,
        num_positional_embeddings=cfg.num_positional_embeddings,
        num_rbf=cfg.num_rbf,
        augment_eps=cfg.augment_eps,
        backbone_diheral=cfg.backbone_diheral,
        dropout=cfg.dropout,
        denoiser=True,
        update_atom=cfg.update_atom,
        num_decoder_blocks=cfg.num_decoder_blocks,
        num_tfmr_heads=cfg.num_tfmr_heads,
        num_tfmr_layers=cfg.num_tfmr_layers,
        number_ligand_atom=cfg.number_ligand_atom,
        mpnn_cutoff=cfg.mpnn_cutoff,
    )
    return model.to(device)


def load_flow_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    config = ckpt["config"]

    denoiser = build_denoiser(config, device)
    flow = DiscreteFlow_AA(
        config,
        denoiser,
        min_t=0.0,
        sidechain_packing=True,
        sample_save_path="results/new_samples/",
    )
    flow.load_state_dict(ckpt["model"])

    ema_flow = EMA(
        flow,
        beta=config.training.ema_beta,
        update_every=config.training.ema_update_every,
    )
    ema_flow.load_state_dict(ckpt["ema"])
    ema_flow = ema_flow.to(device)
    ema_flow.eval()
    return ema_flow


def generate_sequence(
    ema_flow,
    cif_path: Path,
    method: str,
    dt: float,
    steps: int,
    threshold: float,
) -> str:
    with torch.inference_mode():
        if method == "fixed":
            samples, _ = ema_flow.ema_model.sample(str(cif_path), dt=dt, argmax_final=True)
        else:
            samples, _ = ema_flow.ema_model.adaptive_sample(
                str(cif_path),
                num_step=steps,
                threshold=threshold,
                argmax_final=False,
            )
    samples = samples.squeeze()
    return "".join(aap.restype_3to1[aap.index_to_token[i]] for i in samples.cpu().numpy())


def load_inputs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_outputs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--method", choices=["fixed", "adaptive"], default="adaptive")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    inp = Path(args.input_jsonl).resolve()
    outp = Path(args.output_jsonl).resolve()
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not inp.exists():
        raise FileNotFoundError(f"input jsonl not found: {inp}")

    rows = load_inputs(inp)
    print(f"[adflip-batch] loaded inputs={len(rows)}", flush=True)
    print(f"[adflip-batch] loading model ckpt={ckpt} device={args.device}", flush=True)
    model_t0 = time.perf_counter()
    device = torch.device(args.device)
    ema_flow = load_flow_model(ckpt, device)
    print(f"[adflip-batch] model ready in {time.perf_counter() - model_t0:.2f}s", flush=True)

    out_rows: list[dict] = []
    bar = tqdm(
        rows,
        total=len(rows),
        desc="adflip:infer",
        dynamic_ncols=True,
        disable=args.no_progress,
        mininterval=0.5,
    )
    for item in bar:
        idx = int(item["idx"])
        cif_path = Path(item["cif_path"]).resolve()
        one_t0 = time.perf_counter()
        try:
            seq = generate_sequence(
                ema_flow=ema_flow,
                cif_path=cif_path,
                method=args.method,
                dt=args.dt,
                steps=args.steps,
                threshold=args.threshold,
            )
            elapsed = time.perf_counter() - one_t0
            out_rows.append(
                {
                    "idx": idx,
                    "cif_path": str(cif_path),
                    "sequence": seq,
                    "ok": True,
                    "elapsed_sec": round(elapsed, 4),
                }
            )
            bar.set_postfix_str(f"idx={idx} sec={elapsed:.2f} len={len(seq)}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - one_t0
            out_rows.append(
                {
                    "idx": idx,
                    "cif_path": str(cif_path),
                    "ok": False,
                    "error": str(exc),
                    "elapsed_sec": round(elapsed, 4),
                }
            )
            write_outputs(outp, out_rows)
            print(
                f"[adflip-batch] failed idx={idx} cif={cif_path} elapsed={elapsed:.2f}s error={exc}",
                flush=True,
            )
            return 4

    write_outputs(outp, out_rows)
    total = time.perf_counter() - t0
    avg = (total / len(out_rows)) if out_rows else 0.0
    print(f"[adflip-batch] wrote outputs={len(out_rows)} path={outp}", flush=True)
    print(f"[adflip-batch] total={total:.2f}s avg={avg:.2f}s/item", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
